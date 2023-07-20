import numpy as np
import paddle

x = np.random.uniform(-1, 1, [3, 3]).astype(np.float32)

paddle.set_device("SUPA")
x_supa = paddle.to_tensor(x, stop_gradient=False)
res_supa = paddle.abs(x_supa)
print(res_supa)

paddle.set_device("CPU")
x_cpu = paddle.to_tensor(x, stop_gradient=False)
res_cpu = paddle.abs(x_cpu)
print(res_cpu)

