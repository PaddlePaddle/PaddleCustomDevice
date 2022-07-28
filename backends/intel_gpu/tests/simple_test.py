import paddle

paddle.set_device('intel_gpu')
x = paddle.to_tensor([1.0])
y = paddle.to_tensor([7.0])
z = x + y

print (z)


print("----------------------------------------")
a = paddle.to_tensor([4.1,3.2,5.5])
b = paddle.to_tensor([5.3,7.7,5.3])

c = a*b

print(c)

print("----------------------------------------")
