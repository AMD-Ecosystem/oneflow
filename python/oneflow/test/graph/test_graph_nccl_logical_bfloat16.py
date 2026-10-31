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
import unittest
import os

import numpy as np

import oneflow as flow
import oneflow.unittest


@flow.unittest.skip_unless_1n2d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestGraphNcclLogicalBfloat16(flow.unittest.TestCase):
    def test_graph_nccl_logical_bfloat16(test_case):
        local_np = np.arange(4 * 8, dtype=np.float32).reshape(4, 8)
        placement = flow.placement("cuda", ranks=[0, 1])
        B = flow.sbp.broadcast()
        S0 = flow.sbp.split(0)
        S1 = flow.sbp.split(1)
        P = flow.sbp.partial_sum()

        flow.boxing.nccl.enable_use_compute_stream(True)

        class BoxingGraph(flow.nn.Graph):
            def build(self, p, s0, s1):
                return (
                    p.to_global(sbp=B, placement=placement),
                    p.to_global(sbp=S1, placement=placement),
                    s0.to_global(sbp=S1, placement=placement),
                    s1.to_global(sbp=B, placement=placement),
                    s1.to_global(sbp=S0, placement=placement),
                )

        x = flow.tensor(local_np, dtype=flow.bfloat16, device="cuda").to_global(
            sbp=B, placement=placement
        )
        p = flow.tensor(local_np / 2, dtype=flow.bfloat16, device="cuda").to_global(
            sbp=P, placement=placement
        )
        outs = BoxingGraph()(p, x.to_global(sbp=S0), x.to_global(sbp=S1))
        for out, sbp in zip(outs, [B, S1, S1, B, S0]):
            test_case.assertEqual(out.dtype, flow.bfloat16)
            test_case.assertEqual(out.sbp, (sbp,))
            test_case.assertTrue(np.array_equal(out.float().numpy(), local_np))


if __name__ == "__main__":
    unittest.main()
