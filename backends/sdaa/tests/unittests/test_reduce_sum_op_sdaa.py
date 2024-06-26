# BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
# reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAMAGE.

from __future__ import print_function

import numpy as np
import unittest

from op_test import OpTest
import paddle
import paddle.base as base
from paddle.framework import convert_np_dtype_to_dtype_

paddle.enable_static()
SEED = 2021


class TestReduceSum(OpTest):
    def setUp(self):
        np.random.seed(SEED)
        self.set_sdaa()
        self.init_dtype()
        self.python_api = paddle.sum
        self.place = paddle.CustomPlace("sdaa", 0)
        self.init_op_type()
        self.initTestCase()

        self.use_mkldnn = False
        self.attrs = {
            "dim": self.axis,
            "keep_dim": self.keep_dim,
            "reduce_all": self.reduce_all,
        }
        self.inputs = {"X": np.random.random(self.shape).astype(self.dtype)}
        if self.attrs["reduce_all"]:
            self.outputs = {"Out": self.inputs["X"].sum()}
        else:
            self.outputs = {
                "Out": self.inputs["X"].sum(
                    axis=self.axis, keepdims=self.attrs["keep_dim"]
                )
            }

    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def init_dtype(self):
        self.dtype = np.float32

    def init_op_type(self):
        self.op_type = "reduce_sum"
        self.use_mkldnn = False
        self.keep_dim = False
        self.reduce_all = False

    def initTestCase(self):
        self.shape = (15, 16)
        self.axis = (0,)

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        dx = np.divide(
            np.ones_like(self.inputs["X"], dtype=self.dtype), self.outputs["Out"].size
        )
        self.check_grad_with_place(
            self.place,
            ["X"],
            "Out",
            user_defined_grads=[dx],
        )


class TestReduceSumFp16(TestReduceSum):
    def init_dtype(self):
        self.dtype = np.float16

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-2)


class TestReduceSumBool(TestReduceSum):
    def init_dtype(self):
        self.dtype = np.bool_

    def test_check_grad(self):
        pass


class TestReduceSumAxis3D(TestReduceSum):
    def initTestCase(self):
        self.shape = (15, 16, 17)
        self.axis = (1, 2)


class TestReduceSumAxisNeg(TestReduceSum):
    def initTestCase(self):
        self.shape = (15, 16, 17)
        self.axis = (-1,)


class TestReduceSum6D(TestReduceSum):
    def initTestCase(self):
        self.shape = (5, 7, 8, 12, 5, 5)
        self.axis = (1, 2)


class TestReduceAll(TestReduceSum):
    def initTestCase(self):
        self.shape = (5, 6, 2, 10)
        self.axis = None
        self.reduce_all = True


class TestReduceSumNet(unittest.TestCase):
    def set_reduce_sum_function(self, x):
        # keep_dim = False
        return paddle.sum(x, axis=-1)

    def _test(self, run_sdaa=True):
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        main_prog.random_seed = SEED
        startup_prog.random_seed = SEED
        np.random.seed(SEED)

        a_np = np.random.random(size=(2, 3, 4)).astype("float32")
        b_np = np.random.random(size=(2, 3, 4)).astype("float32")
        label_np = np.random.randint(2, size=(2, 1)).astype("int64")

        with paddle.static.program_guard(main_prog, startup_prog):
            a = paddle.static.data(name="a", shape=[2, 3, 4], dtype="float32")
            b = paddle.static.data(name="b", shape=[2, 3, 4], dtype="float32")
            label = paddle.static.data(name="label", shape=[2, 1], dtype="int64")

            a_1 = paddle.static.nn.fc(x=a, size=4, num_flatten_dims=2, activation=None)
            b_1 = paddle.static.nn.fc(x=b, size=4, num_flatten_dims=2, activation=None)
            z = paddle.add(a_1, b_1)
            z_1 = self.set_reduce_sum_function(z)

            prediction = paddle.static.nn.fc(x=z_1, size=2, activation="softmax")

            cost = paddle.nn.functional.cross_entropy(input=prediction, label=label)
            loss = paddle.mean(cost)
            sgd = paddle.optimizer.Momentum(learning_rate=0.01)
            sgd.minimize(loss)

        if run_sdaa:
            place = paddle.CustomPlace("sdaa", 0)
        else:
            place = paddle.CPUPlace()

        exe = paddle.static.Executor(place)
        exe.run(startup_prog)

        print("Start run on {}".format(place))
        for epoch in range(100):

            pred_res, loss_res = exe.run(
                main_prog,
                feed={"a": a_np, "b": b_np, "label": label_np},
                fetch_list=[prediction, loss],
            )
            if epoch % 10 == 0:
                print(
                    "Epoch {} | Prediction[0]: {}, Loss: {}".format(
                        epoch, pred_res[0], loss_res
                    )
                )

        return pred_res, loss_res

    def test_sdaa(self):
        cpu_pred, cpu_loss = self._test(False)
        sdaa_pred, sdaa_loss = self._test(True)

        self.assertTrue(np.allclose(sdaa_pred, cpu_pred))
        self.assertTrue(np.allclose(sdaa_loss, cpu_loss))


class TestReduceSumNet2(TestReduceSumNet):
    def set_reduce_sum_function(self, x):
        # keep_dim = True
        return paddle.sum(x, axis=-1, keepdim=True)


class TestReduceSumNet3(TestReduceSumNet):
    def _test(self, run_sdaa=True):
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        main_prog.random_seed = SEED
        startup_prog.random_seed = SEED
        np.random.seed(SEED)

        a_np = np.random.random(size=(2, 3, 4)).astype("float32")
        b_np = np.random.random(size=(2, 3, 4)).astype("float32")

        with paddle.static.program_guard(main_prog, startup_prog):
            a = paddle.static.data(name="a", shape=[2, 3, 4], dtype="float32")
            b = paddle.static.data(name="b", shape=[2, 3, 4], dtype="float32")

            z = paddle.add(a, b)
            loss = paddle.sum(z)
            sgd = paddle.optimizer.Momentum(learning_rate=0.01)
            sgd.minimize(loss)

        if run_sdaa:
            place = paddle.CustomPlace("sdaa", 0)
        else:
            place = paddle.CPUPlace()

        exe = paddle.static.Executor(place)
        exe.run(startup_prog)

        print("Start run on {}".format(place))
        for epoch in range(100):

            loss_res = exe.run(
                main_prog, feed={"a": a_np, "b": b_np}, fetch_list=[loss]
            )
            if epoch % 10 == 0:
                print("Epoch {} | Loss: {}".format(epoch, loss_res))

        return loss_res, loss_res


class TestReduceSumINT64(OpTest):
    def setUp(self):
        np.random.seed(SEED)
        self.set_sdaa()
        self.init_dtype()
        self.python_api = paddle.sum
        self.place = paddle.CustomPlace("sdaa", 0)
        self.init_op_type()
        self.initTestCase()

        self.use_mkldnn = False
        self.attrs = {
            "dim": self.axis,
            "keep_dim": self.keep_dim,
            "reduce_all": self.reduce_all,
        }
        self.inputs = {"X": np.random.uniform(-100, 100, self.shape).astype(self.dtype)}
        if self.attrs["reduce_all"]:
            self.outputs = {"Out": self.inputs["X"].sum()}
        else:
            self.outputs = {
                "Out": self.inputs["X"].sum(
                    axis=self.axis, keepdims=self.attrs["keep_dim"]
                )
            }

    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def init_dtype(self):
        self.dtype = np.int64

    def init_op_type(self):
        self.op_type = "reduce_sum"
        self.use_mkldnn = False
        self.keep_dim = False
        self.reduce_all = False

    def initTestCase(self):
        self.shape = (6, 8, 4)
        self.axis = (1,)

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestReduceSumINT32(TestReduceSumINT64):
    def init_dtype(self):
        self.dtype = np.int32


class API_TestSumOp(unittest.TestCase):
    def run_static(self, shape, x_dtype, attr_axis, attr_dtype=None, np_axis=None):
        if np_axis is None:
            np_axis = attr_axis

        places = [paddle.CustomPlace("sdaa", 0)]
        for place in places:
            with base.program_guard(base.Program(), base.Program()):
                data = paddle.static.data("data", shape=shape, dtype=x_dtype)
                result_sum = paddle.sum(x=data, axis=attr_axis, dtype=attr_dtype)

                exe = base.Executor(place)
                input_data = np.random.rand(*shape).astype(x_dtype)
                (res,) = exe.run(feed={"data": input_data}, fetch_list=[result_sum])

            np.testing.assert_allclose(
                res,
                np.sum(input_data.astype(attr_dtype), axis=np_axis),
                rtol=1e-05,
            )

    def test_static(self):
        paddle.device.set_device("sdaa")
        shape = [10, 10]
        axis = 1

        self.run_static(shape, "float32", axis, attr_dtype=None)

        self.run_static(shape, "bool", axis, attr_dtype=None)

        shape = [5, 5, 5]
        self.run_static(shape, "float32", (0, 1), attr_dtype="float32")
        self.run_static(shape, "float32", (), attr_dtype="float32", np_axis=(0, 1, 2))

    def test_dygraph(self):
        paddle.device.set_device("sdaa")
        np_x = np.random.random([2, 3, 4]).astype("bool")
        with base.dygraph.guard():
            x = base.dygraph.to_variable(np_x)
            out0 = paddle.sum(x).numpy()
            out1 = paddle.sum(x, axis=0).numpy()
            out2 = paddle.sum(x, axis=(0, 1)).numpy()
            out3 = paddle.sum(x, axis=(0, 1, 2)).numpy()

        self.assertTrue((out0 == np.sum(np_x, axis=(0, 1, 2))).all())
        self.assertTrue((out1 == np.sum(np_x, axis=0)).all())
        self.assertTrue((out2 == np.sum(np_x, axis=(0, 1))).all())
        self.assertTrue((out3 == np.sum(np_x, axis=(0, 1, 2))).all())


class TestReduceSumOpError(unittest.TestCase):
    def test_errors(self):
        def test_define_error_out_dtype():
            paddle.device.set_device("sdaa")
            x = paddle.to_tensor([1.2, 2.2], dtype="int32")
            paddle.sum(x, dtype="int16")

        paddle.disable_static()
        self.assertRaises(ValueError, test_define_error_out_dtype)
        paddle.enable_static()


def create_test_fp64_class(parent):
    class TestReduceSumOpFp64Case(parent):
        def init_dtype(self):
            self.dtype = np.float64

    cls_name = "{0}_{1}".format(parent.__name__, "FP64")
    TestReduceSumOpFp64Case.__name__ = cls_name
    globals()[cls_name] = TestReduceSumOpFp64Case


create_test_fp64_class(TestReduceSum)
create_test_fp64_class(TestReduceAll)
create_test_fp64_class(TestReduceSum6D)
create_test_fp64_class(TestReduceSumAxis3D)
create_test_fp64_class(TestReduceSumAxisNeg)


def create_test_int32_class(parent):
    class TestReduceSumOpInt32Case(parent):
        def init_dtype(self):
            self.dtype = np.int32

        # error in test due to mean kernel not supporting int32 in the CPU
        def test_check_grad(self):
            pass

    cls_name = "{0}_{1}".format(parent.__name__, "INT32")
    TestReduceSumOpInt32Case.__name__ = cls_name
    globals()[cls_name] = TestReduceSumOpInt32Case


create_test_int32_class(TestReduceSum)
create_test_int32_class(TestReduceAll)
create_test_int32_class(TestReduceSum6D)
create_test_int32_class(TestReduceSumAxis3D)
create_test_int32_class(TestReduceSumAxisNeg)


class TestReduceWithSpecifyOutDtype(OpTest):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.python_api = paddle.sum
        self.set_sdaa()
        self.place = paddle.CustomPlace("sdaa", 0)
        self.inputs = {"X": np.random.random((6, 2, 10)).astype("float64")}
        self.outputs = {"Out": self.inputs["X"].sum().astype("float64")}
        self.attrs = {"reduce_all": True}
        self.attrs.update(
            {
                "in_dtype": int(convert_np_dtype_to_dtype_(np.float32)),
                "out_dtype": int(convert_np_dtype_to_dtype_(np.float64)),
            }
        )

    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        dx = np.divide(
            np.ones_like(self.inputs["X"], dtype=self.dtype), self.outputs["Out"].size
        )
        self.check_grad_with_place(
            self.place,
            ["X"],
            "Out",
            user_defined_grads=[dx],
        )


class TestReduceWithSpecifyOutDtype1(TestReduceWithSpecifyOutDtype):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.python_api = paddle.sum
        self.set_sdaa()
        self.place = paddle.CustomPlace("sdaa", 0)
        self.inputs = {"X": np.random.random((6, 2, 10)).astype("float64")}
        self.outputs = {"Out": self.inputs["X"].sum(axis=1)}
        self.attrs = {"dim": [1]}
        self.attrs.update(
            {
                "in_dtype": int(convert_np_dtype_to_dtype_(np.float32)),
                "out_dtype": int(convert_np_dtype_to_dtype_(np.float64)),
            }
        )

    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        dx = np.divide(
            np.ones_like(self.inputs["X"], dtype=self.dtype), self.outputs["Out"].size
        )
        self.check_grad_with_place(
            self.place,
            ["X"],
            "Out",
            user_defined_grads=[dx],
        )


class TestReduceWithSpecifyOutDtype2(TestReduceWithSpecifyOutDtype):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.python_api = paddle.sum
        self.set_sdaa()
        self.place = paddle.CustomPlace("sdaa", 0)
        self.inputs = {"X": np.random.random((6, 2, 10)).astype("float64")}
        self.outputs = {"Out": self.inputs["X"].sum(axis=1, keepdims=True)}
        self.attrs = {"dim": [1], "keep_dim": True}
        self.attrs.update(
            {
                "in_dtype": int(convert_np_dtype_to_dtype_(np.float32)),
                "out_dtype": int(convert_np_dtype_to_dtype_(np.float64)),
            }
        )

    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        dx = np.divide(
            np.ones_like(self.inputs["X"], dtype=self.dtype), self.outputs["Out"].size
        )
        self.check_grad_with_place(
            self.place,
            ["X"],
            "Out",
            user_defined_grads=[dx],
        )


if __name__ == "__main__":
    unittest.main()
