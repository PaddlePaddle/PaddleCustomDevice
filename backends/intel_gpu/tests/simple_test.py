import paddle

paddle.set_device('intel_gpu')
x = paddle.to_tensor([1])
y = paddle.to_tensor([7])
z = x + y

print (z)
