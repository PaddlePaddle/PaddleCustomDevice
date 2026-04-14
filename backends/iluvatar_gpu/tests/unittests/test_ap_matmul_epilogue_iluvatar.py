# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import unittest
import numpy as np
from pathlib import Path

import paddle
import paddle.incubate.cc as pcc
import paddle.incubate.cc.typing as pct
import paddle.profiler as profiler


os.environ["AP_WORKSPACE_DIR"] = "/tmp/paddle/ap_workspace"


def GetPirProgram(fused_func, tensor_args):
    dtypes = tuple(tensor.dtype for tensor in tensor_args)
    func = fused_func.func_overload_ctx.dtypes2func.get(dtypes, None)
    return str(func.infer_program.forward_program)


DT = "float16"
BS = 4
MS = 784
NS = 192
KS = 768


class TestMatmulEpilogue(unittest.TestCase):
    def setUp(self):
        dtype = DT
        x_shape = [BS, MS, KS]
        self.x = paddle.randn(x_shape, dtype=dtype)
        self.x.stop_gradient = True

        y_shape = [KS, NS]
        self.y = paddle.randn(y_shape, dtype=dtype)
        self.y.stop_gradient = True

        b_shape = [BS, MS, NS]
        self.b = paddle.randn(b_shape, dtype=dtype)
        self.b.stop_gradient = True

        bias_shape = [NS]
        self.bias = paddle.randn(bias_shape, dtype=dtype)
        self.bias.stop_gradient = True

        residual_shape = [BS, MS, NS]
        self.residual = paddle.randn(residual_shape, dtype=dtype)
        self.residual.stop_gradient = True

        mask_shape = [BS, MS, NS]
        self.mask = paddle.randn(mask_shape, dtype=dtype)
        self.mask.stop_gradient = True

    def get_matmul_add_act(self):
        B = pct.DimVar(BS)
        M = pct.DimVar(MS)
        K = pct.DimVar(KS)
        N = pct.DimVar(NS)
        T = pct.DTypeVar("T", DT)

        def matmul_add_act(
            x: pct.Tensor([B, M, K], T),
            y: pct.Tensor([K, N], T),
            b: pct.Tensor([B, M, N], T),
        ):

            out = paddle.matmul(x, y)
            out = out + b
            return paddle.nn.functional.relu(out)

        return matmul_add_act

    def get_matmul_add_divide_multipy_add(self):
        B = pct.DimVar(BS)
        M = pct.DimVar(MS)
        K = pct.DimVar(KS)
        N = pct.DimVar(NS)
        T = pct.DTypeVar("T", DT)

        def matmul_add_divide_multipy_add(
            x: pct.Tensor([B, M, K], T),
            y: pct.Tensor([K, N], T),
            bias: pct.Tensor([N], T),
            residual: pct.Tensor([B, M, N], T),
            mask: pct.Tensor([B, M, N], T),
        ):
            out = paddle.matmul(x, y)
            out = out + bias
            # out = out / 1.2
            out = out * mask
            return residual + out

        return matmul_add_divide_multipy_add

    def check_if_ap_variadic_exist(self, fused_foo, foo_args):
        generated_pir_program = GetPirProgram(fused_foo, foo_args)
        assert (
            "pd_op.ap_variadic" in generated_pir_program
        ), "AP fusion failed, none pd_op.ap_variadic found in the pir_program."

    def check_by_profiler(self, fused_foo, foo_args):
        paddle.device.synchronize()

        iters = 10
        with profiler.Profiler(
            targets=[profiler.ProfilerTarget.CPU, profiler.ProfilerTarget.GPU],
            on_trace_ready=profiler.export_chrome_tracing("./profiler_log"),
            timer_only=False,
        ) as prof:
            for _ in range(iters):
                _ = fused_foo(*foo_args)
                prof.step()
            prof.summary()

    def test_subgraph(self):
        foo = self.get_matmul_add_act()
        foo_args = (self.x, self.y, self.b)

        # foo = self.get_matmul_add_divide_multipy_add()
        # foo_args = (self.x, self.y, self.bias, self.residual, self.mask)

        iluvatar_gpu_dir = Path(__file__).resolve().parent.parent.parent
        fused_foo = pcc.compile(
            foo,
            ap_path=f"{iluvatar_gpu_dir}/apy/device",
            backend_device="custom_device",
        )

        self.check_if_ap_variadic_exist(fused_foo, foo_args)
        self.check_by_profiler(fused_foo, foo_args)

        ap_outs = fused_foo(*foo_args)
        dy_outs = foo(*foo_args)
        for dy_out, ap_out in zip(dy_outs, ap_outs):
            np.testing.assert_allclose(dy_out, ap_out, rtol=5e-2, atol=1e-1)


if __name__ == "__main__":
    unittest.main()
