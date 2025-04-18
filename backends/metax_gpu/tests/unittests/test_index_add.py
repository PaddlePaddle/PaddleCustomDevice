import paddle
import numpy as np
import unittest

class TestIndexAdd(unittest.TestCase):
    def setUp(self):
        self.custom_place = paddle.CustomPlace('GPGPU', 0)
        self.cpu_place = paddle.CPUPlace()


    def test_index_add_basic(self):

        x_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        index_np = np.array([0, 2], dtype=np.int64)
        value_np = np.array([4.0, 5.0], dtype=np.float32)
        
        x_cpu = paddle.to_tensor(x_np.copy(), place=self.cpu_place)
        index_cpu = paddle.to_tensor(index_np, place=self.cpu_place)
        value_cpu = paddle.to_tensor(value_np, place=self.cpu_place)
        out_cpu = paddle.index_add(x_cpu, index_cpu, 0, value_cpu)
        self.assertTrue(np.allclose(out_cpu.numpy(), np.array([5.0, 2.0, 8.0])))

        # x_custom = paddle.to_tensor(x_np.copy(), place=self.custom_place)
        # index_custom = paddle.to_tensor(index_np, place=self.custom_place)
        # value_custom = paddle.to_tensor(value_np, place=self.custom_place)
        # out_custom = paddle.index_add(x_custom, index_custom, 0, value_custom)
        

        # self.assertTrue(np.allclose(out_cpu.numpy(), out_custom.numpy(place=self.cpu_place)))


    # def test_index_add_2d(self):
    #     x_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    #     index_np = np.array([1, 0], dtype=np.int64)
    #     value_np = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
        
    #     # dim=1 测试
    #     x_custom = paddle.to_tensor(x_np.copy(), place=self.custom_place)
    #     out_custom = paddle.index_add(x_custom, 
    #                                  paddle.to_tensor(index_np, place=self.custom_place),
    #                                  paddle.to_tensor(value_np, place=self.custom_place),
    #                                  dim=1)
    #     expected = np.array([[31.0, 32.0, 3.0], [44.0, 25.0, 6.0]])
    #     self.assertTrue(np.allclose(out_custom.numpy(place=self.cpu_place), expected))


    # def test_index_out_of_range(self):
    #     x = paddle.to_tensor([1.0, 2.0, 3.0], place=self.custom_place)
    #     index = paddle.to_tensor([3], dtype='int64', place=self.custom_place)
    #     value = paddle.to_tensor([5.0], place=self.custom_place)
        
    #     with self.assertRaises(ValueError):
    #         paddle.index_add(x, index, value, dim=0)


    # def test_int64_dtype(self):
    #     x_np = np.array([1, 2, 3], dtype=np.int64)
    #     index_np = np.array([0, 2], dtype=np.int64)
    #     value_np = np.array([4, 5], dtype=np.int64)
        
    #     out_custom = paddle.index_add(
    #         paddle.to_tensor(x_np, place=self.custom_place),
    #         paddle.to_tensor(index_np, place=self.custom_place),
    #         paddle.to_tensor(value_np, place=self.custom_place),
    #         dim=0
    #     )
    #     self.assertTrue(np.array_equal(out_custom.numpy(place=self.cpu_place), np.array([5, 2, 8])))


    # def test_backward(self):
    #     x = paddle.to_tensor([1.0, 2.0, 3.0], place=self.custom_place, stop_gradient=False)
    #     index = paddle.to_tensor([0, 2], dtype='int64', place=self.custom_place)
    #     value = paddle.to_tensor([4.0, 5.0], place=self.custom_place)
        
    #     out = paddle.index_add(x, index, value, dim=0)
    #     out.backward()
        

    #     expected_grad = np.array([1.0, 1.0, 1.0])
    #     self.assertTrue(np.allclose(x.grad.numpy(place=self.cpu_place), expected_grad))

    # def test_random_data_consistency(self):
    #     np.random.seed(42)
    #     for _ in range(10):

    #         x_np = np.random.randn(5, 4).astype(np.float32)
    #         index_np = np.random.randint(0, 5, size=3).astype(np.int64)
    #         value_np = np.random.randn(3, 4).astype(np.float32)
            

    #         x_cpu = paddle.to_tensor(x_np.copy(), place=self.cpu_place)
    #         out_cpu = paddle.index_add(x_cpu, 
    #                                   paddle.to_tensor(index_np, place=self.cpu_place),
    #                                   paddle.to_tensor(value_np, place=self.cpu_place),
    #                                   dim=0)
            

    #         x_custom = paddle.to_tensor(x_np.copy(), place=self.custom_place)
    #         out_custom = paddle.index_add(x_custom,
    #                                     paddle.to_tensor(index_np, place=self.custom_place),
    #                                     paddle.to_tensor(value_np, place=self.custom_place),
    #                                     dim=0)

    #         self.assertTrue(np.allclose(out_cpu.numpy(), out_custom.numpy(place=self.cpu_place), atol=1e-6))

if __name__ == '__main__':
    unittest.main()