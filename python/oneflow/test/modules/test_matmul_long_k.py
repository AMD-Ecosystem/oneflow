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


@flow.unittest.skip_unless_1n1d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestMatmulLongK(flow.unittest.TestCase):
    def test_matmul_long_k(test_case):
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


if __name__ == "__main__":
    unittest.main()
