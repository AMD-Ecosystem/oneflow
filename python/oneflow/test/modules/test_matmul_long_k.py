"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import contextlib
import os
import subprocess
import sys
import unittest

import numpy as np

import oneflow as flow
import oneflow.unittest

# Bits in the significand of each type, so that 2 ** -bits bounds one rounding.
_SIGNIFICAND_BITS = {flow.float16: 11, flow.bfloat16: 8}

_K = 1 << 20


def _check(test_case, dtype, a, b, fn, desc):
    rng = np.random.RandomState(0)
    # Rounded on the host so that only the 16-bit operands reach the GPU.
    fa = flow.tensor(rng.uniform(-1, 1, a)).to(dtype)
    fb = flow.tensor(rng.uniform(-1, 1, b)).to(dtype)
    out = fn(fa.to("cuda"), fb.to("cuda"))
    test_case.assertEqual(out.dtype, dtype)
    # The reference multiplies exactly the values the kernel sees.
    ref = fn(fa.to(flow.float64), fb.to(flow.float64)).numpy()
    err = np.abs(out.cpu().to(flow.float64).numpy() - ref).max() / np.abs(ref).max()
    # Accumulated at float precision and rounded to dtype once; rounding the partial sums to
    # dtype along k gives several times this.
    tol = 4 * 2.0 ** -_SIGNIFICAND_BITS[dtype]
    test_case.assertLess(err, tol, "%s %s" % (desc, dtype))


def _check_grouped(test_case, problems, bias):
    # grouped_matmul_bias computes x * w^T + b per problem (m, n, k), in float16 only.
    rng = np.random.RandomState(0)
    half = lambda a: flow.tensor(a).to(flow.float16).to("cuda")
    xs = [half(rng.uniform(-1, 1, (m, k))) for m, n, k in problems]
    ws = [half(rng.uniform(-1, 1, (n, k))) for m, n, k in problems]
    bs = [half(rng.uniform(-1, 1, n)) for m, n, k in problems] if bias else None
    if bias:
        ys = flow._C.grouped_matmul_bias(xs, ws, bs)
    else:
        ys = flow._C.grouped_matmul(xs, ws)
    for i, (x, w, y) in enumerate(zip(xs, ws, ys)):
        test_case.assertEqual(y.dtype, flow.float16)
        test_case.assertEqual(tuple(y.shape), problems[i][:2])
        if y.numel() == 0:
            continue
        ref = x.cpu().to(flow.float64).numpy() @ w.cpu().to(flow.float64).numpy().T
        if bias:
            ref += bs[i].cpu().to(flow.float64).numpy()
        err = np.abs(y.cpu().to(flow.float64).numpy() - ref).max() / np.abs(ref).max()
        tol = 4 * 2.0 ** -_SIGNIFICAND_BITS[flow.float16]
        test_case.assertLess(err, tol, "grouped %s bias=%s" % (problems[i], bias))


def _check_all(test_case):
    for dtype in [flow.float16, flow.bfloat16]:
        _check(test_case, dtype, (2, _K), (_K, 16), flow.matmul, "matmul")
        _check(
            test_case,
            dtype,
            (2, _K),
            (16, _K),
            lambda x, y: flow.matmul(x, y, transpose_b=True),
            "matmul transpose_b",
        )
        # Neither operand has k as its leading dimension.
        _check(
            test_case,
            dtype,
            (_K, 2),
            (_K, 16),
            lambda x, y: flow.matmul(x, y, transpose_a=True),
            "matmul transpose_a",
        )
        _check(test_case, dtype, (2, 1, _K), (2, _K, 16), flow.bmm, "bmm")
        _check(test_case, dtype, (1, _K), (2, _K, 16), flow.matmul, "broadcast")
    problems = [(1, 16, _K), (3, 8, _K)]
    _check_grouped(test_case, problems, True)
    _check_grouped(test_case, problems, False)
    # Shorter k, including one that is not a multiple of 2^15.
    for k in [163840, 200000, 1 << 18]:
        for dtype in [flow.float16, flow.bfloat16]:
            _check(
                test_case,
                dtype,
                (2, k),
                (16, k),
                lambda x, y: flow.matmul(x, y, transpose_b=True),
                "matmul transpose_b k=%d" % k,
            )
        _check_grouped(test_case, [(3, 8, k)], False)
    # Empty outputs: a weight or an input without rows.
    for problems in [[(2, 0, _K)], [(0, 2, _K)]]:
        _check_grouped(test_case, problems, False)


def _check_row_tiles(test_case):
    # With tiles of 256 16-bit elements (128 float outputs), outputs with 300 cuBLAS rows (columns
    # of the oneflow output) take three tiles of 100 rows per column. The last two cases run one
    # GEMM per batch.
    gen = flow.Generator(device="cuda")
    gen.manual_seed(0)
    rand = lambda *s: flow.rand(*s, device="cuda", generator=gen) * 2 - 1
    k, n = 1 << 15, 300
    cases = [
        ((3, k), (k, n), {}, "matmul"),
        ((3, k), (n, k), {"transpose_b": True}, "matmul transpose_b"),
        (
            (k, 3),
            (n, k),
            {"transpose_a": True, "transpose_b": True},
            "matmul transpose_a transpose_b",
        ),
        ((3, k), (2, k, n), {}, "broadcast"),
        ((2, 3, k), (2, k, n), {}, "batched"),
    ]
    for dtype in [flow.float16, flow.bfloat16]:
        for a, b, kwargs, desc in cases:
            fa, fb = rand(*a).to(dtype), rand(*b).to(dtype)
            out = flow.matmul(fa, fb, **kwargs)
            test_case.assertEqual(out.dtype, dtype)
            # The reference multiplies exactly the values the kernel sees.
            ref = flow.matmul(fa.to(flow.float32), fb.to(flow.float32), **kwargs)
            err = ((out.to(flow.float32) - ref).abs().max() / ref.abs().max()).item()
            tol = 4 * 2.0 ** -_SIGNIFICAND_BITS[dtype]
            test_case.assertLess(err, tol, "%s, rows in pieces %s" % (desc, dtype))


def _check_narrow_last_tile(test_case):
    # A weight gradient x^T @ dy with an odd number of inputs, cut into output tiles 2 wide, so the
    # last tile is 1 wide: each tile must be written exactly once, whichever library runs it.
    with _env("ONEFLOW_EP_CUDA_LONG_K_GEMM_MAX_TILE_SIZE", "100"):
        _check(
            test_case,
            flow.float16,
            (40000, 1025),
            (40000, 8),
            lambda x, y: flow.matmul(x, y, transpose_a=True),
            "matmul transpose_a, 1-wide last tile",
        )


@contextlib.contextmanager
def _env(name, value):
    old = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if old is None:
            del os.environ[name]
        else:
            os.environ[name] = old


@flow.unittest.skip_unless_1n1d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestMatmulLongK(flow.unittest.TestCase):
    def test_matmul_long_k(test_case):
        _check_all(test_case)
        _check_narrow_last_tile(test_case)

    def test_matmul_long_k_without_hipblaslt(test_case):
        # On ROCm, the path taken when hipBLASLt has no algorithm for a shape. Elsewhere the
        # variable has no effect.
        with _env("ONEFLOW_EP_CUDA_LONG_K_GEMM_USE_HIPBLASLT", "0"):
            _check_all(test_case)
            # An output larger than the default 4 MiB workspace holds as float, so the float
            # tile is a separate allocation.
            _check(
                test_case,
                flow.float16,
                (1040, 1 << 15),
                (1024, 1 << 15),
                lambda x, y: flow.matmul(x, y, transpose_b=True),
                "matmul transpose_b, tile larger than the workspace",
            )
            with _env("ONEFLOW_EP_CUDA_LONG_K_GEMM_MAX_TILE_SIZE", "256"):
                _check_row_tiles(test_case)
            _check_narrow_last_tile(test_case)

    def test_matmul_long_k_without_workspace(test_case):
        # With no cuBLAS workspace the float tiles of the fallback need their own memory. A stream
        # reads the workspace size when it is created, so the test runs in a new process.
        env = dict(os.environ, ONEFLOW_EP_CUDA_CUBLAS_WORKSPACE_SIZE_MB="0")
        test = "TestMatmulLongK.test_matmul_long_k_without_hipblaslt"
        result = subprocess.run([sys.executable, __file__, test], env=env)
        test_case.assertEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
