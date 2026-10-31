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
from collections import OrderedDict

import numpy as np

from oneflow.test_utils.automated_test_util import *
from oneflow.test_utils.test_util import GenArgList

import oneflow as flow
import oneflow.unittest


def _test_batchnorm_float64(test_case, shape, channels_last, affine, track_running_stats):
    import torch

    rng = np.random.RandomState(sum(shape))
    np_x = rng.randn(*shape) * 3 + 0.5
    np_dy = rng.randn(*shape)
    channels = shape[1]
    np_weight = rng.uniform(0.5, 1.5, size=channels)
    np_bias = rng.randn(channels)
    # oneflow keeps epsilon and 1 - momentum as float32 attributes.
    momentum = 1 - float(np.float32(0.9))
    eps = float(np.float32(1e-5))
    torch_x = torch.tensor(np_x, requires_grad=True)
    torch_weight = torch.tensor(np_weight, requires_grad=True) if affine else None
    torch_bias = torch.tensor(np_bias, requires_grad=True) if affine else None
    torch_mean = torch.zeros(channels, dtype=torch.float64)
    torch_var = torch.ones(channels, dtype=torch.float64)
    # NHWC data with axis=3 is what oneflow runs for channels_last batch norm.
    to_flow = (lambda a: a.transpose(0, 2, 3, 1)) if channels_last else (lambda a: a)
    from_flow = (lambda a: a.transpose(0, 3, 1, 2)) if channels_last else (lambda a: a)
    x = flow.tensor(to_flow(np_x), device="cuda", requires_grad=True)
    weight = flow.tensor(np_weight, device="cuda", requires_grad=True) if affine else None
    bias = flow.tensor(np_bias, device="cuda", requires_grad=True) if affine else None
    mean = flow.zeros(channels, dtype=flow.float64, device="cuda")
    var = flow.ones(channels, dtype=flow.float64, device="cuda")
    for training in [True, False] if track_running_stats else [True]:
        torch_y = torch.nn.functional.batch_norm(
            torch_x,
            torch_mean if track_running_stats else None,
            torch_var if track_running_stats else None,
            torch_weight,
            torch_bias,
            training=training,
            momentum=momentum,
            eps=eps,
        )
        y = flow._C.normalization(
            x,
            mean if track_running_stats else None,
            var if track_running_stats else None,
            weight,
            bias,
            3 if channels_last else 1,
            1e-5,
            0.1,
            training,
        )
        torch_y.backward(torch.tensor(np_dy))
        y.backward(flow.tensor(to_flow(np_dy), device="cuda"))
        pairs = [(from_flow(y.numpy()), torch_y.detach().numpy())]
        pairs.append((from_flow(x.grad.numpy()), torch_x.grad.numpy()))
        if affine:
            pairs.append((weight.grad.numpy(), torch_weight.grad.numpy()))
            pairs.append((bias.grad.numpy(), torch_bias.grad.numpy()))
        if track_running_stats:
            pairs.append((mean.numpy(), torch_mean.numpy()))
            pairs.append((var.numpy(), torch_var.numpy()))
        for actual, expected in pairs:
            test_case.assertTrue(np.allclose(actual, expected, rtol=1e-10, atol=1e-10))

@flow.unittest.skip_unless_1n1d()
class TestBatchNormModule(flow.unittest.TestCase):
    @unittest.skip("skip for now, becase it failed 11 times in past week")
    @autotest(
        auto_backward=True, rtol=1e-3, atol=1e-3, check_grad_use_random_data=False
    )
    def test_batchnorm1d_module_with_random_data(test_case):
        device = random_device()
        channel = random(1, 4).to(int)
        m = torch.nn.BatchNorm1d(
            num_features=channel,
            track_running_stats=random().to(bool),
            affine=random().to(bool),
        ).to(device)
        m.train(random())
        x = random_tensor(
            ndim=3, dim0=random(1, 4), dim1=channel, requires_grad=True
        ).to(device)
        y = m(x)
        return y

    @autotest(
        auto_backward=True, rtol=1e-3, atol=1e-3, check_grad_use_random_data=False
    )
    def test_batchnorm2d_module_with_random_data(test_case):
        device = random_device()
        channel = random(1, 4).to(int)
        m = torch.nn.BatchNorm2d(
            num_features=channel,
            track_running_stats=random().to(bool),
            affine=random().to(bool),
        ).to(device)
        m.train(random())
        x = random_tensor(
            ndim=4, dim0=random(1, 4), dim1=channel, requires_grad=True
        ).to(device)
        y = m(x)
        return y

    @autotest(
        auto_backward=True, rtol=1e-3, atol=1e-3, check_grad_use_random_data=False
    )
    def test_batchnorm3d_module_with_random_data(test_case):
        device = random_device()
        channel = random(1, 4).to(int)
        m = torch.nn.BatchNorm3d(
            num_features=channel,
            track_running_stats=random().to(bool),
            affine=random().to(bool),
        ).to(device)
        m.train(random())
        x = random_tensor(ndim=5, dim1=channel, requires_grad=True).to(device)
        y = m(x)
        return y

    @autotest(rtol=1e-3, atol=1e-3, check_grad_use_random_data=False)
    def test_functional_batchnorm_with_random_data(test_case):
        device = random_device()
        channel = random(1, 4).to(int)
        x = random_tensor(ndim=5, dim1=channel, requires_grad=True).to(device)
        running_mean = random_tensor(ndim=1, dim0=channel, requires_grad=False)
        running_var = random_tensor(ndim=1, dim0=channel, low=0.0, requires_grad=False)
        weight = random_tensor(ndim=1, dim0=channel)
        bias = random_tensor(ndim=1, dim0=channel)
        result = torch.nn.functional.batch_norm(
            input=x,
            running_mean=running_mean,
            running_var=running_var,
            weight=weight,
            bias=bias,
            training=random_bool(),
        )
        return result

    @autotest(rtol=1e-3, atol=1e-3, auto_backward=False, check_graph=False)
    def test_batchnorm2d_module_with_half_random_data(test_case):
        device = random_device()
        channel = random(1, 4).to(int)
        m = torch.nn.BatchNorm2d(
            num_features=channel,
            track_running_stats=random().to(bool),
            affine=random().to(bool),
        ).to(device)
        m.train(random())
        m.half()
        x = random_tensor(
            ndim=4, dim0=random(1, 4), dim1=channel, requires_grad=True
        ).to(device)
        x.half()
        y = m(x)
        return y

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cuda cases")
    def test_batchnorm_float64(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(6, 5), (4, 3, 7), (4, 3, 5, 6), (2, 3, 4, 3, 2)]
        arg_dict["channels_last"] = [False]
        arg_dict["affine"] = [True, False]
        arg_dict["track_running_stats"] = [True, False]
        for arg in GenArgList(arg_dict):
            _test_batchnorm_float64(test_case, *arg)
        # Without gamma and beta the functor sizes them from dim 1, so channels_last needs affine.
        _test_batchnorm_float64(test_case, (4, 3, 5, 6), True, True, True)
        _test_batchnorm_float64(test_case, (4, 3, 5, 6), True, True, False)
        # Reductions longer than a 256-thread block, and more parameter slots (70000
        # per-activation features) than a launch has blocks.
        for shape in [(8, 3, 16, 16), (600, 700), (3, 70000)]:
            _test_batchnorm_float64(test_case, shape, False, True, True)
        _test_batchnorm_float64(test_case, (8, 3, 16, 16), True, True, True)

    @profile(torch.nn.functional.batch_norm)
    def profile_batchnorm(test_case):
        input = torch.ones(16, 128, 28, 28)
        running_mean = torch.randn(128)
        running_var = torch.randn(128)
        weight = torch.randn(128)
        bias = torch.randn(128)
        torch.nn.functional.batch_norm(
            input, running_mean, running_var, weight, bias, True
        )
        torch.nn.functional.batch_norm(
            input, running_mean, running_var, weight, bias, False
        )
        torch.nn.functional.batch_norm(
            input, running_mean, running_var, None, None, True
        )
        torch.nn.functional.batch_norm(
            input, running_mean, running_var, None, None, False
        )


if __name__ == "__main__":
    unittest.main()
