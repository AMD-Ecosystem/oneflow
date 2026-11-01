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
import unittest

import numpy as np

import oneflow as flow
import oneflow.unittest

# Bits in the significand of each type, so that 2 ** -bits bounds one rounding.
_SIGNIFICAND_BITS = {flow.float16: 11, flow.bfloat16: 8}

_K = 1 << 20


def _check(test_case, dtype, a, b, fn, desc):
    rng = np.random.RandomState(0)
    fa = flow.tensor(rng.uniform(-1, 1, a), device="cuda").to(dtype)
    fb = flow.tensor(rng.uniform(-1, 1, b), device="cuda").to(dtype)
    out = fn(fa, fb)
    test_case.assertEqual(out.dtype, dtype)
    # The reference multiplies exactly the values the kernel sees.
    ref = fn(fa.to(flow.float64).cpu(), fb.to(flow.float64).cpu()).numpy()
    err = np.abs(out.to(flow.float64).numpy() - ref).max() / np.abs(ref).max()
    # Accumulated at float precision and rounded to dtype once; rounding the partial sums to
    # dtype along k gives several times this.
    tol = 4 * 2.0 ** -_SIGNIFICAND_BITS[dtype]
    test_case.assertLess(err, tol, "%s %s" % (desc, dtype))


def _check_grouped(test_case, problems, bias):
    # grouped_matmul_bias computes x * w^T + b per problem (m, n, k), in float16 only.
    rng = np.random.RandomState(0)
    half = lambda a: flow.tensor(a, device="cuda").to(flow.float16)
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
        ref = x.to(flow.float64).numpy() @ w.to(flow.float64).numpy().T
        if bias:
            ref += bs[i].to(flow.float64).numpy()
        err = np.abs(y.to(flow.float64).numpy() - ref).max() / np.abs(ref).max()
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
        _check(test_case, dtype, (2, 1, _K), (2, _K, 16), flow.bmm, "bmm")
        _check(test_case, dtype, (1, _K), (2, _K, 16), flow.matmul, "broadcast")
    problems = [(1, 16, _K), (3, 8, _K)]
    _check_grouped(test_case, problems, True)
    _check_grouped(test_case, problems, False)
    # Empty outputs: a weight or an input without rows.
    for problems in [[(2, 0, _K)], [(0, 2, _K)]]:
        _check_grouped(test_case, problems, False)


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

    def test_matmul_long_k_without_hipblaslt(test_case):
        # On ROCm, the path taken when hipBLASLt has no algorithm for a shape. Elsewhere the
        # variable has no effect.
        with _env("ONEFLOW_EP_CUDA_LONG_K_GEMM_USE_HIPBLASLT", "0"):
            _check_all(test_case)
            # An output larger than the default 4 MiB workspace holds as float, so the
            # product goes through it in two pieces.
            _check(
                test_case,
                flow.float16,
                (640, 1 << 15),
                (2048, 1 << 15),
                lambda x, y: flow.matmul(x, y, transpose_b=True),
                "matmul transpose_b, output in pieces",
            )


if __name__ == "__main__":
    unittest.main()
