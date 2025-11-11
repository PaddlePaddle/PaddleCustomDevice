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

paddle.enable_static()
SEED = 2021


class TestMul(OpTest):
    # case 1: (32, 5) * (5, 100) -> (32, 100)
    def config(self):
        self.x_shape = (32, 5)
        self.y_shape = (5, 100)

    def setUp(self):
        self.set_sdaa()
        self.op_type = "mul"
        self.init_dtype()
        self.config()
        np.random.seed(SEED)
        self.inputs = {
            "X": np.random.random(self.x_shape).astype(self.dtype),
            "Y": np.random.random(self.y_shape).astype(self.dtype),
        }
        self.outputs = {"Out": np.dot(self.inputs["X"], self.inputs["Y"])}

    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("sdaa", 0)

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-5)

    def test_check_grad_normal(self):
        self.check_grad_with_place(
            self.place,
            ["X", "Y"],
            "Out",
            max_relative_error=0.5,
        )

    def test_check_grad_ingore_x(self):
        self.check_grad_with_place(
            self.place,
            ["Y"],
            "Out",
            no_grad_set=set("X"),
            max_relative_error=0.5,
        )

    def test_check_grad_ingore_y(self):
        self.check_grad_with_place(
            self.place,
            ["X"],
            "Out",
            no_grad_set=set("Y"),
            max_relative_error=0.5,
        )

    def check_grad_with_place(
        self,
        place,
        inputs_to_check,
        output_names,
        no_grad_set=None,
        numeric_grad_delta=0.005,
        in_place=False,
        max_relative_error=0.005,
        user_defined_grads=None,
        user_defined_grad_outputs=None,
        check_dygraph=True,
        numeric_place=None,
    ):
        if self.dtype == np.float32:
            numeric_place = paddle.CPUPlace()
        super().check_grad_with_place(
            place,
            inputs_to_check,
            output_names,
            no_grad_set,
            numeric_grad_delta,
            in_place,
            max_relative_error,
            user_defined_grads,
            user_defined_grad_outputs,
            check_dygraph,
            numeric_place=numeric_place,
        )


class TestMul2(TestMul):
    # case 2: (20, 2, 5) * (10, 50) -> (20, 50), x_num_col_dims = 1
    def config(self):
        self.x_shape = (20, 2, 5)
        self.y_shape = (10, 50)

    def setUp(self):
        self.set_sdaa()
        self.op_type = "mul"
        self.init_dtype()
        self.config()
        np.random.seed(SEED)
        self.inputs = {
            "X": np.random.random(self.x_shape).astype(self.dtype),
            "Y": np.random.random(self.y_shape).astype(self.dtype),
        }
        self.outputs = {
            "Out": np.dot(self.inputs["X"].reshape(20, 10), self.inputs["Y"])
        }


class TestMul3(TestMul):
    # case 3: (20, 3, 4) * (4, 50) -> (20, 3, 50), x_num_col_dims = 2

    def config(self):
        self.x_shape = (20, 3, 4)
        self.y_shape = (4, 50)

    def setUp(self):
        self.set_sdaa()
        self.op_type = "mul"
        self.init_dtype()
        self.config()
        np.random.seed(SEED)
        self.inputs = {
            "X": np.random.random(self.x_shape).astype(self.dtype),
            "Y": np.random.random(self.y_shape).astype(self.dtype),
        }
        self.attrs = {"x_num_col_dims": 2}
        self.outputs = {"Out": np.matmul(self.inputs["X"], self.inputs["Y"])}


class TestMul4(TestMul):
    # case 4: (20, 2, 2, 3) * (12, 50) -> (20, 50), x_num_col_dims = 1
    def config(self):
        self.x_shape = (20, 2, 2, 3)
        self.y_shape = (12, 50)

    def setUp(self):
        self.set_sdaa()
        self.op_type = "mul"
        self.init_dtype()
        self.config()
        np.random.seed(SEED)
        self.inputs = {
            "X": np.random.random(self.x_shape).astype(self.dtype),
            "Y": np.random.random(self.y_shape).astype(self.dtype),
        }
        self.outputs = {
            "Out": np.dot(self.inputs["X"].reshape(20, 12), self.inputs["Y"])
        }


# --------------------test matmul fp16--------------------


def create_test_fp16_class(parent, atol=0.001, max_relative_error=0.5):
    class TestMulOpFp16Case(parent):
        def init_kernel_type(self):
            self.dtype = np.float16

        def test_check_output(self):
            self.check_output_with_place(self.place, atol=atol)

        def test_check_grad_normal(self):
            self.check_grad_with_place(
                self.place, ["X", "Y"], "Out", max_relative_error=max_relative_error
            )

        def test_check_grad_ingore_x(self):
            self.check_grad_with_place(
                self.place,
                ["Y"],
                "Out",
                no_grad_set=set("X"),
                max_relative_error=max_relative_error,
            )

        def test_check_grad_ingore_y(self):
            self.check_grad_with_place(
                self.place,
                ["X"],
                "Out",
                no_grad_set=set("Y"),
                max_relative_error=max_relative_error,
            )

    cls_name = "{0}_{1}".format(parent.__name__, "Fp16")
    TestMulOpFp16Case.__name__ = cls_name
    globals()[cls_name] = TestMulOpFp16Case


create_test_fp16_class(TestMul)
create_test_fp16_class(TestMul2)
create_test_fp16_class(TestMul3)
create_test_fp16_class(TestMul4)


class TestMulNet(unittest.TestCase):
    def init_dtype(self):
        self.dtype = np.float32

    def _test(self, run_sdaa=True):
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        main_prog.random_seed = SEED
        startup_prog.random_seed = SEED
        np.random.seed(SEED)

        a_np = np.random.random(size=(2, 3)).astype(self.dtype)
        b_np = np.random.random(size=(2, 3)).astype(self.dtype)
        c_np = np.random.random(size=(3, 2)).astype(self.dtype)
        d_np = np.random.random(size=(3, 2)).astype(self.dtype)
        label_np = np.random.randint(2, size=(2, 1)).astype("int64")

        with paddle.static.program_guard(main_prog, startup_prog):
            a = paddle.static.data(name="a", shape=[2, 3], dtype=self.dtype)
            b = paddle.static.data(name="b", shape=[2, 3], dtype=self.dtype)
            c = paddle.static.data(name="c", shape=[3, 2], dtype=self.dtype)
            d = paddle.static.data(name="d", shape=[3, 2], dtype=self.dtype)
            label = paddle.static.data(name="label", shape=[2, 1], dtype="int64")

            sum_1 = paddle.add(a, b)
            sum_2 = paddle.add(c, d)
            result = paddle.matmul(sum_1, sum_2)

            fc_1 = paddle.static.nn.fc(x=result, size=8)
            prediction = paddle.static.nn.fc(x=fc_1, size=2, activation="softmax")

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

        print("TestMulNet Start run on {} . ".format(place))
        for epoch in range(100):

            pred_res, loss_res = exe.run(
                main_prog,
                feed={"a": a_np, "b": b_np, "c": c_np, "d": d_np, "label": label_np},
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
        self.init_dtype()
        cpu_pred, cpu_loss = self._test(False)
        sdaa_pred, sdaa_loss = self._test(True)

        self.assertTrue(np.allclose(sdaa_pred, cpu_pred))
        self.assertTrue(np.allclose(sdaa_loss, cpu_loss))


class TestMulNet3_2(unittest.TestCase):
    def init_dtype(self):
        self.dtype = np.float32

    def _test(self, run_sdaa=True):
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        main_prog.random_seed = SEED
        startup_prog.random_seed = SEED
        np.random.seed(SEED)

        a_np = np.random.random(size=(2, 3, 4)).astype(self.dtype)
        b_np = np.random.random(size=(2, 3, 4)).astype(self.dtype)
        c_np = np.random.random(size=(12, 5)).astype(self.dtype)
        d_np = np.random.random(size=(12, 5)).astype(self.dtype)
        label_np = np.random.randint(2, size=(2, 1)).astype("int64")

        with paddle.static.program_guard(main_prog, startup_prog):
            a = paddle.static.data(name="a", shape=[2, 3, 4], dtype=self.dtype)
            b = paddle.static.data(name="b", shape=[2, 3, 4], dtype=self.dtype)
            c = paddle.static.data(name="c", shape=[12, 5], dtype=self.dtype)
            d = paddle.static.data(name="d", shape=[12, 5], dtype=self.dtype)
            label = paddle.static.data(name="label", shape=[2, 1], dtype="int64")

            sum_1 = paddle.add(a, b)
            sum_1_re_shape = paddle.reshape(sum_1, shape=[2, 12])
            sum_2 = paddle.add(c, d)
            result = paddle.matmul(sum_1_re_shape, sum_2)

            fc_1 = paddle.static.nn.fc(x=result, size=8)
            prediction = paddle.static.nn.fc(x=fc_1, size=2, activation="softmax")

            cost = paddle.nn.functional.cross_entropy(input=prediction, label=label)
            loss = paddle.mean(cost)
            # TODO(zhanggq): Use of the momentum optimizer results in a loss of precision
            sgd = paddle.optimizer.AdamW(learning_rate=0.1)
            sgd.minimize(loss)

        if run_sdaa:
            place = paddle.CustomPlace("sdaa", 0)
        else:
            place = paddle.CPUPlace()
        exe = paddle.static.Executor(place)
        exe.run(startup_prog)

        print("testMulNet3_2 tart run on {}".format(place))
        for epoch in range(100):

            pred_res, loss_res = exe.run(
                main_prog,
                feed={"a": a_np, "b": b_np, "c": c_np, "d": d_np, "label": label_np},
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
        self.init_dtype()
        cpu_pred, cpu_loss = self._test(False)
        sdaa_pred, sdaa_loss = self._test(True)

        self.assertTrue(np.allclose(sdaa_pred, cpu_pred, atol=1e-5))
        self.assertTrue(np.allclose(sdaa_loss, cpu_loss, atol=1e-5))


class TestMulNet3_2_xc2(unittest.TestCase):
    def init_dtype(self):
        self.dtype = np.float32

    def _test(self, run_sdaa=True):
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        main_prog.random_seed = SEED
        startup_prog.random_seed = SEED
        np.random.seed(SEED)

        a_np = np.random.random(size=(2, 3, 4)).astype(self.dtype)
        b_np = np.random.random(size=(2, 3, 4)).astype(self.dtype)
        c_np = np.random.random(size=(4, 5)).astype(self.dtype)
        d_np = np.random.random(size=(4, 5)).astype(self.dtype)
        label_np = np.random.randint(2, size=(2, 1)).astype("int64")

        with paddle.static.program_guard(main_prog, startup_prog):
            a = paddle.static.data(name="a", shape=[2, 3, 4], dtype=self.dtype)
            b = paddle.static.data(name="b", shape=[2, 3, 4], dtype=self.dtype)
            c = paddle.static.data(name="c", shape=[4, 5], dtype=self.dtype)
            d = paddle.static.data(name="d", shape=[4, 5], dtype=self.dtype)
            label = paddle.static.data(name="label", shape=[2, 1], dtype="int64")

            sum_1 = paddle.add(a, b)
            sum_2 = paddle.add(c, d)
            result = paddle.matmul(sum_1, sum_2)
            result_re = paddle.reshape(result, shape=[2, 15])

            fc_1 = paddle.static.nn.fc(x=result_re, size=8)
            prediction = paddle.static.nn.fc(x=fc_1, size=2, activation="softmax")

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

        print("TestMulNet3_2_xc2. Start run on {}".format(place))
        for epoch in range(100):

            pred_res, loss_res = exe.run(
                main_prog,
                feed={"a": a_np, "b": b_np, "c": c_np, "d": d_np, "label": label_np},
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
        self.init_dtype()
        cpu_pred, cpu_loss = self._test(False)
        sdaa_pred, sdaa_loss = self._test(True)

        self.assertTrue(np.allclose(sdaa_pred, cpu_pred))
        self.assertTrue(np.allclose(sdaa_loss, cpu_loss, rtol=4.0e-5))


class TestMulNet4_2(unittest.TestCase):
    def init_dtype(self):
        self.dtype = np.float32

    def _test(self, run_sdaa=True):
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        main_prog.random_seed = SEED
        startup_prog.random_seed = SEED
        np.random.seed(SEED)

        a_np = np.random.random(size=(12, 5)).astype(self.dtype)
        b_np = np.random.random(size=(12, 5)).astype(self.dtype)
        c_np = np.random.random(size=(12, 5)).astype(self.dtype)
        d_np = np.random.random(size=(12, 5)).astype(self.dtype)
        label_np = np.random.randint(2, size=(2, 1)).astype("int64")

        with paddle.static.program_guard(main_prog, startup_prog):
            a = paddle.static.data(name="a", shape=[12, 5], dtype=self.dtype)
            b = paddle.static.data(name="b", shape=[12, 5], dtype=self.dtype)
            c = paddle.static.data(name="c", shape=[12, 5], dtype=self.dtype)
            d = paddle.static.data(name="d", shape=[12, 5], dtype=self.dtype)
            label = paddle.static.data(name="label", shape=[2, 1], dtype="int64")

            sum_1 = paddle.add(a, b)  # [12, 5]
            sum_2 = paddle.add(c, d)  # [12, 5]
            fc_1 = paddle.static.nn.fc(x=sum_1, size=2)  # [12, 2]
            fc_1_re_shape = paddle.reshape(fc_1, shape=[2, 3, 2, 2])
            fc_1_re_shape = paddle.reshape(fc_1_re_shape, shape=[2, 12])
            fc_2 = paddle.static.nn.fc(x=sum_2, size=2)  # [12, 2]
            result = paddle.matmul(fc_1_re_shape, fc_2)  # [2, 12] * [12, 2]

            prediction = paddle.static.nn.fc(x=result, size=2, activation="softmax")

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

        print("testMulNet4_2 tart run on {}".format(place))
        for epoch in range(100):

            pred_res, loss_res = exe.run(
                main_prog,
                feed={"a": a_np, "b": b_np, "c": c_np, "d": d_np, "label": label_np},
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
        self.init_dtype()
        cpu_pred, cpu_loss = self._test(False)
        sdaa_pred, sdaa_loss = self._test(True)

        self.assertTrue(np.allclose(sdaa_pred, cpu_pred, atol=1e-5))
        self.assertTrue(np.allclose(sdaa_loss, cpu_loss, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
