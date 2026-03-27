import os
import subprocess
import unittest
import time
import numpy as np
import paddle
import paddle.incubate.cc as pcc
import paddle.incubate.cc.typing as pct


def GetPirProgram(fused_func, tensor_args):
    dtypes = tuple(tensor.dtype for tensor in tensor_args)
    func = fused_func.func_overload_ctx.dtypes2func.get(dtypes, None)
    return str(func.infer_program.forward_program)

DT = 'float16'
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

        b1_shape = [1]
        self.b1 = paddle.randn(b1_shape, dtype=dtype)
        self.b1.stop_gradient = True

        bias_shape = [NS]
        self.bias = paddle.randn(bias_shape, dtype=dtype)
        self.bias.stop_gradient = True

        residual_shape = [BS, MS, NS]
        self.residual = paddle.randn(residual_shape, dtype=dtype)
        self.residual.stop_gradient = True

        mask_shape = [BS, MS, NS]
        self.mask = paddle.randn(mask_shape, dtype=dtype)
        self.mask.stop_gradient = True


    def get_subgraph(self):
        B = pct.DimVar(BS)
        M = pct.DimVar(MS)
        K = pct.DimVar(KS)
        N = pct.DimVar(NS)
        T = pct.DTypeVar("T", DT)

        def matmul_add_act(
            x: pct.Tensor([B, M, K], T),
            y: pct.Tensor([K, N], T),
            b: pct.Tensor([B, M, N], T),
            b1: pct.Tensor([1], T),
        ):

            out = paddle.matmul(x, y)
            out = out + b
            return paddle.nn.functional.relu(out)


        def matmul_add_divide_multipy_add_S1(
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
        
        return matmul_add_divide_multipy_add_S1

    def test_subgraph(self):
        foo = self.get_subgraph()
        fused_foo = pcc.compile(
            foo, ap_path=f"{os.path.dirname(paddle.__file__)}/apy/matmul_pass"
        )

        ap_outs = fused_foo(self.x, self.y, self.bias, self.residual, self.mask)
        dy_outs = foo(self.x, self.y, self.bias, self.residual, self.mask)
        generated_pir_program = GetPirProgram(fused_foo, [x, y, b])
    
    
        assert 'pd_op.ap_variadic' in generated_pir_program, "fusion failed, excludes pd_op.ap_variadic"

        iters = 10
        # warmup
        _ = fused_foo(self.x, self.y, self.b, self.b1)
        _ = foo(self.x, self.y, self.b, self.b1)

        paddle.device.synchronize()
        start = time.time()
        for _ in range(iters):
            _ = fused_foo(self.x, self.y, self.b, self.b1)
        paddle.device.synchronize()
        end = time.time()
        avg_time = (end - start) / iters
        print(f"[Performance] Avg latency per run: {avg_time:.6f} s")

        for dy_out, ap_out in zip(dy_outs, ap_outs):
            np.testing.assert_allclose(dy_out, ap_out, rtol=5e-2, atol=1e-1)

if __name__ == "__main__":
    unittest.main()