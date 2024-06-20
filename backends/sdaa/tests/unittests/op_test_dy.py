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
# STRICT LIABILITY,OR TORT (INCLUDINGEargs NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAMAGE.

import struct
import unittest
import warnings
import numpy as np
from paddle.base import core
import paddle
from paddle import base
import copy
from utils import static_guard

from paddle.base import core
from paddle.base.framework import (
    Program,
)

SEED = 1234


def run_in_dy(func):
    def wrapper(*args, **kw):
        paddle.disable_static()
        func(*args, **kw)
        paddle.enable_static()

    return wrapper


class TestDygraphInplace(unittest.TestCase):
    def setUp(self):
        self.place = paddle.CustomPlace("sdaa", 0)()
        self.op_type = ""
        self.dtype = np.float32
        self.init_data()
        self.set_np_compare_func()
        self.inputs = {}
        self.attrs = {}
        self.outputs = {}
        self.python_api = None
        self.python_inplace_api = None

    def set_np_compare_func(self):
        self.np_compare = np.array_equal

    def init_data(self):
        self.input_var_numpy = np.random.uniform(-5, 5, [10, 20, 1])
        self.dtype = "float32"

    def assumption_assert_and_transform(self, args, inp_num):
        """
        transform inputs by the following rules:
            Note: it may not be possible to distinguish list with one Tensor,you should use wrapper to distinguish.
            1. [Tensor] -> Tensor
            2. [Tensor, Tensor, ...] -> list of Tensors
            3. None -> None
            4. Others: raise Error

        only support "X" is list of Tensor, currently don't support other structure like dict.
        """
        # temp no use (Because now the test dont support default parameter)
        return args
        inp_args = [
            [inp] if inp is None else inp for inp in args[:inp_num]
        ]  # convert None -> [None]
        for inp in inp_args:
            assert isinstance(
                inp, list
            ), "currently only support `X` is [Tensor], don't support other structure."
        args = [inp[0] if len(inp) == 1 else inp for inp in inp_args] + args[inp_num:]
        return args

    def prepare_agrs(
        self, inputs_tensor, attrs_tensor, outputs_tensor, dtype=None, static_args=False
    ):
        if self.dtype is None:
            self.dtype = np.float32
        tensor_dtype = self.dtype if dtype is None else dtype
        result = []
        for k, v in inputs_tensor.items():
            if static_args is True:
                v = paddle.static.data(name=k, shape=v.shape, dtype=v.dtype)
                inputs_tensor[k] = v
            elif isinstance(v, paddle.Tensor) is False:
                v = (
                    paddle.to_tensor(v, tensor_dtype)
                    if v.dtype == np.float16
                    else paddle.to_tensor(v)
                )
                inputs_tensor[k] = v
            result.append(v)

        for k, v in attrs_tensor.items():
            if isinstance(v, np.ndarray) is True:
                v = paddle.to_tensor(v)
                attrs_tensor[k] = v
            result.append(v)

        for k, v in outputs_tensor.items():
            if isinstance(v, paddle.Tensor) is False:
                v = paddle.to_tensor(v, tensor_dtype)
                outputs_tensor[k] = v

        return result

    def cal_python_api(self, args, use_inplace=False):

        if use_inplace is True:
            result = self.python_inplace_api(args)
        else:
            result = self.python_api(*args)
        real_result = {}
        if isinstance(result, tuple):
            if len(result) != len(self.outputs):
                raise Exception(
                    "please check the element nums in self.outputs and python_api outputs ,make sure they are equal"
                )
            residx = 0
            for k, v in self.outputs.items():
                real_result[k] = result[residx]
        else:
            if len(self.outputs) != 1:
                raise Exception(
                    "please check the element nums in self.outputs and python_api outputs ,make sure they are equal"
                )
            for k, v in self.outputs.items():
                real_result[k] = result
        return real_result

    def check_output_with_place_customized(self, checker, place, check_inplace=False):

        inputs_tensor = (
            copy.copy(self.inputs)
            if hasattr(self, "inputs")
            else {}
            if hasattr(self, "inputs")
            else {}
        )
        attrs_tensor = copy.copy(self.attrs) if hasattr(self, "attrs") else {}
        outputs_tensor = (
            copy.copy(self.outputs)
            if hasattr(self, "outputs")
            else {}
            if hasattr(self, "outputs")
            else {}
        )

        with base.dygraph.guard(place=place):
            op_args = self.prepare_agrs(inputs_tensor, attrs_tensor, outputs_tensor)
            outs = self.cal_python_api(op_args, check_inplace)

        outs = [np.array(out) for _, out in outs.items()]
        outs.sort(key=len)
        checker(outs)
        if check_inplace is True:
            self.check_inplace_output_with_place(self.place)

    @run_in_dy
    def check_output_with_place(
        self,
        place,
        atol=1e-5,
        max_relative_error=1e-5,
        no_check_set={},
        inplace_args=None,  # the inplace op maybe have multi_inplace_args
        equal_nan=False,
        check_dygraph=True,
        check_prim=False,
        inplace_atol=None,
        check_cinn=False,
        check_inplace=False,
    ):
        inputs_tensor = copy.copy(self.inputs) if hasattr(self, "inputs") else {}
        attrs_tensor = copy.copy(self.attrs) if hasattr(self, "attrs") else {}
        outputs_tensor = copy.copy(self.outputs) if hasattr(self, "outputs") else {}
        with base.dygraph.guard(place=place):
            op_args = self.prepare_agrs(inputs_tensor, attrs_tensor, outputs_tensor)
            actual_outs = self.cal_python_api(op_args, check_inplace)

        for k, v in self.outputs.items():
            if k in no_check_set:
                continue
            expect_out = np.array(v)
            actual_out = np.array(actual_outs[k])
            assert (
                actual_out.shape == expect_out.shape
            ), "Operator ({}) : Output ({}) shape mismatch, expect shape is {}, but actual shape is {}".format(
                self.op_type, self.op_type, expect_out.shape, actual_out.shape
            )

            np.testing.assert_allclose(
                expect_out,
                actual_out,
                rtol=max_relative_error,
                atol=atol,
                err_msg="Operator ("
                + self.op_type
                + ") Output ("
                + self.op_type
                + ") has diff at "
                + str(place)
                + " when using and not using inplace"
                + "\nExpect "
                + str(expect_out)
                + "\n"
                + "But Got"
                + str(actual_out)
                + " in class "
                + self.__class__.__name__,
            )

        if check_inplace:
            self.check_inplace_output_with_place(self.place)

    def is_float16_op(self):
        # self.dtype is the dtype of inputs
        return self.dtype == np.float16 or self.dtype == "float16"

    def check_grad_with_place(
        self,
        place,
        inputs_to_check,
        output_names,
        no_grad_set={},
        numeric_grad_delta=0.005,
        in_place=False,
        max_relative_error=0.005,
        user_defined_grads=None,
        user_defined_grad_outputs=None,
        check_dygraph=True,
        check_prim=False,
        only_check_prim=False,
        numeric_place=paddle.CPUPlace(),
        atol=1e-5,
        check_cinn=False,
        check_inplace=True,
        compare_static=False,
    ):
        warnings.warn("now inplace_grad_test only support Fp16 and Fp32")

        def convert_uint16_to_float(in_list):
            in_list = np.asarray(in_list)
            out = np.vectorize(
                lambda x: struct.unpack(
                    "<f", struct.pack("<I", np.uint32(x) << np.uint32(16))
                )[0],
                otypes=[np.float32],
            )(in_list.flat)
            return np.reshape(out, in_list.shape)

        def get_grad(
            place,
            use_inplace,
            dtype=self.dtype,
            user_defined_grad_outputs=None,
            run_static=False,
        ):

            grad_result = []
            inputs_tensor = copy.copy(self.inputs) if hasattr(self, "inputs") else {}
            attrs_tensor = copy.copy(self.attrs) if hasattr(self, "attrs") else {}
            outputs_tensor = copy.copy(self.outputs) if hasattr(self, "outputs") else {}
            # get no_inplace_grad
            if run_static is True:
                with static_guard():
                    prog = Program()
                    scope = core.Scope()
                    block = prog.global_block()
                    op_args = self.prepare_agrs(
                        inputs_tensor, attrs_tensor, outputs_tensor, dtype
                    )
                    for idx, value in enumerate(op_args):
                        if isinstance(value, paddle.static.Variable) is True:
                            value.stop_gradient = (
                                True if (idx == 0 and use_inplace is True) else False
                            )

                    paddle_outs = self.cal_python_api(op_args, use_inplace)

                    grad_outputs = []
                    if user_defined_grad_outputs is not None:
                        if not isinstance(user_defined_grad_outputs, list):
                            user_defined_grad_outputs = [user_defined_grad_outputs]

                        for grad_out_value in user_defined_grad_outputs:
                            # `persistable` is used to avoid executor create new var in local scope
                            var = block.create_var(
                                shape=grad_out_value.shape,
                                dtype=grad_out_value.dtype,
                                persistable=True,
                            )
                            true_var = scope.var(var.name)
                            tensor = true_var.get_tensor()
                            tensor.set(grad_out_value, place)
                            grad_outputs.append(var)

                    targets = [
                        paddle_outs[name]
                        for name in paddle_outs
                        if name in output_names
                    ]
                    inputs = [
                        inputs_tensor[name]
                        for name in inputs_to_check
                        if name in inputs_to_check
                    ]

                    if no_grad_set is None or no_grad_set == {}:
                        ngs = set()
                    else:
                        ngs = no_grad_set
                    if grad_outputs == []:
                        grad_outputs = None
                    grad_inputs = paddle.static.gradients(
                        targets, inputs, grad_outputs, ngs
                    )
                    fetch_list = grad_inputs
                    exe = paddle.static.Executor(place)
                    res = exe.run(
                        feed=self.inputs, fetch_list=fetch_list, return_numpy=True
                    )
                    return res
            # get inplace_grad
            with base.dygraph.guard(place=place):
                op_args = self.prepare_agrs(
                    inputs_tensor, attrs_tensor, outputs_tensor, dtype
                )
                for idx, value in enumerate(op_args):
                    if isinstance(value, paddle.Tensor) is True:
                        value.stop_gradient = (
                            True if idx == 0 and use_inplace is True else False
                        )
                # when use inplace the x.stop_grad must = True
                paddle_outs = self.cal_python_api(op_args, use_inplace)
                if user_defined_grad_outputs is None:
                    paddle_outs[output_names].backward()

                else:
                    # user_defined_grad_outputs here are numpy arrays
                    if not isinstance(user_defined_grad_outputs, list):
                        user_defined_grad_outputs = [user_defined_grad_outputs]
                    grad_outputs = []
                    for grad_out_value in user_defined_grad_outputs:
                        grad_outputs.append(paddle.to_tensor(grad_out_value))

                    # delete the inputs which no need to calculate grad
                    for no_grad_val in no_grad_set:
                        del inputs_tensor[no_grad_val]

                    grad_inputs = paddle.grad(
                        outputs=paddle.utils.flatten(paddle_outs),
                        inputs=paddle.utils.flatten(inputs_tensor),
                        grad_outputs=grad_outputs,
                    )
                    return [grad.numpy(False) for grad in grad_inputs]

            for k, v in inputs_tensor.items():
                if k in no_grad_set or (k not in inputs_to_check):
                    continue
                grad_result.append(v.grad)

            return grad_result

        # 和cpu/静态图对比，看inplace grad op计算结果是否出错
        analytic_grads = []
        analytic_grads = get_grad(place, False, self.dtype, user_defined_grad_outputs)

        if compare_static is False:
            # compare with cpu
            numeric_grads = (
                get_grad(numeric_place, False, np.float32, user_defined_grad_outputs)
                if user_defined_grads is None
                else user_defined_grads
            )
        else:
            # compare with no_inplace_op
            numeric_grads = (
                get_grad(
                    numeric_place,
                    False,
                    np.float32,
                    user_defined_grad_outputs,
                    run_static=True,
                )
                if user_defined_grads is None
                else user_defined_grads
            )

        if self.is_float16_op():
            max_relative_error = (
                0.001 if max_relative_error < 0.001 else max_relative_error
            )
        if compare_static is True:
            max_relative_error = 1e-6
            atol = 1e-6
        for idx, grad in enumerate(analytic_grads):
            actual_grad = np.array(grad).astype(np.float32)
            expect_grad = np.array(numeric_grads[idx]).astype(np.float32)
            np.testing.assert_allclose(
                expect_grad,
                actual_grad,
                rtol=max_relative_error,
                atol=atol,
                err_msg="Operator ("
                + self.op_type
                + ") Output ("
                + self.op_type
                + ") has diff at "
                + str(place)
                + " when using and not using inplace"
                + "\nExpect "
                + str(expect_grad)
                + "\n"
                + "But Got"
                + str(actual_grad)
                + " in class "
                + self.__class__.__name__,
            )
        # check the inplace api's backward if its correct
        if check_inplace is True:
            self.check_backward_error()
            self.check_backward_success_1()
            self.check_backward_success_2()

    def _get_need_run_ops(self, op_desc, fwd_op_desc=None):
        """Postorder traversal of the 'grad' tree to get all ops that need to run during inplace test.
        An op needs to run during inplace check if,
        (1) it has infer_inplace,
        (2) it has infer_inplace in its grad descendants. (since we need its outputs as to construct its grad's inputs)

        Args:
            op_desc (OpDesc): The op_desc of current op.
            fwd_op_desc (OpDesc): The op_desc of current op's forward op, None if current op has no forward op.
                E.g. relu's fwd_op is None, relu_grad's fwd_op is relu, relu_grad_grad's fwd_op is relu_grad, etc.

        Returns:
            need_run_ops (list[(op_desc, fwd_op_desc)]): The ops that need to run during inplace test.
        """
        need_run_ops = []
        visited_ops = []

        def _dfs_grad_op(op_desc, fwd_op_desc=None):
            visited_ops.append(op_desc.type())
            has_infer_inplace = base.core.has_infer_inplace(op_desc.type())
            has_grad_op_maker = base.core.has_grad_op_maker(op_desc.type())
            has_infer_inplace_in_grad_descendants = False
            if not has_grad_op_maker:
                has_infer_inplace_in_descendants = False
            else:
                # get grad_op_desc
                grad_op_desc_list, op_grad_to_var = core.get_grad_op_desc(
                    op_desc, set(), []
                )
                if not grad_op_desc_list:
                    has_infer_inplace_in_grad_descendants = False
                else:
                    for i, grad_op_desc in enumerate(grad_op_desc_list):
                        if grad_op_desc.type() not in visited_ops and _dfs_grad_op(
                            grad_op_desc, fwd_op_desc=op_desc
                        ):
                            has_infer_inplace_in_grad_descendants = True
            if has_infer_inplace or has_infer_inplace_in_grad_descendants:
                need_run_ops.append((op_desc, fwd_op_desc))
                return True
            else:
                return False

        _dfs_grad_op(op_desc, fwd_op_desc=fwd_op_desc)
        return need_run_ops

    # check the inplace forward api is correct
    def check_inplace_output_with_place(
        self, place, no_check_set=None, inplace_atol=None
    ):
        has_infer_inplace = (
            base.core.has_infer_inplace(self.op_type)
            if hasattr(self, "op_type")
            else False
        )
        has_grad_op_maker = base.core.has_grad_op_maker(self.op_type)
        if has_infer_inplace is False:
            return
        self.check_inplace_api()
        self.check_forward_version()
        self.check_leaf_inplace_var_error()

    # check is the input.address equal with output.address
    def check_inplace_api(self):

        inputs_tensor = copy.copy(self.inputs) if hasattr(self, "inputs") else {}
        attrs_tensor = copy.copy(self.attrs) if hasattr(self, "attrs") else {}
        outputs_tensor = copy.copy(self.outputs) if hasattr(self, "outputs") else {}
        args = self.prepare_agrs(inputs_tensor, attrs_tensor, outputs_tensor)

        inplace_output = self.cal_python_api(args, True)
        self.assertTrue(id(args[0]) == id(inplace_output["Out"]))

        inplace_output["Out"][0] = 2
        np.testing.assert_array_equal(args[0].numpy(), inplace_output["Out"].numpy())

    # check the inplace_input change times
    def check_forward_version(self):

        with paddle.base.dygraph.guard(place=self.place):
            inputs_tensor = copy.copy(self.inputs) if hasattr(self, "inputs") else {}
            attrs_tensor = copy.copy(self.attrs) if hasattr(self, "attrs") else {}
            outputs_tensor = copy.copy(self.outputs) if hasattr(self, "outputs") else {}

            args = self.prepare_agrs(inputs_tensor, attrs_tensor, outputs_tensor)

            self.assertEqual(args[0].inplace_version, 0)

            inplace_output = self.cal_python_api(args, True)
            self.assertEqual(args[0].inplace_version, 1)

            # TODO: figure out why call set_value here.
            # inplace_output['Out'][0] = 2
            # self.assertEqual(args[0].inplace_version, 2)

            if self.op_type not in {"squeeze2", "unsqueeze2"}:
                inplace_output = self.cal_python_api(args, True)
                self.assertEqual(args[0].inplace_version, 2)

    def check_leaf_inplace_var_error(self):
        with paddle.base.dygraph.guard(place=self.place):
            inputs_tensor = copy.copy(self.inputs) if hasattr(self, "inputs") else {}
            attrs_tensor = copy.copy(self.attrs) if hasattr(self, "attrs") else {}
            outputs_tensor = copy.copy(self.outputs) if hasattr(self, "outputs") else {}

            args = self.prepare_agrs(inputs_tensor, attrs_tensor, outputs_tensor)

            for k, v in inputs_tensor.items():
                v.stop_gradient = False

            def leaf_inplace_error():
                self.cal_python_api(args, True)

            self.assertRaises(ValueError, leaf_inplace_error)

    def check_backward_error(self):
        # It raises an error because the inplace operator will result
        # in incorrect gradient computation.
        with paddle.base.dygraph.guard(place=self.place):
            inputs_tensor = copy.copy(self.inputs) if hasattr(self, "inputs") else {}
            attrs_tensor = copy.copy(self.attrs) if hasattr(self, "attrs") else {}
            outputs_tensor = copy.copy(self.outputs) if hasattr(self, "outputs") else {}
            args = self.prepare_agrs(inputs_tensor, attrs_tensor, outputs_tensor)

            var_a = args[0]
            var_a.stop_gradient = False

            var_b = var_a**2

            # Here, the gradient computation will use the value of var_b
            var_c = var_b**2
            args[0] = var_b

            self.cal_python_api(args, True)

            loss = paddle.nn.functional.relu(var_c)
            with self.assertRaisesRegex(
                RuntimeError,
                "received tensor_version:{} != wrapper_version_snapshot:{}".format(
                    1, 0
                ),
            ):
                loss.backward()

    def check_backward_success_1(self):
        # var_b is modified inplace before using it, the inplace operator doesn't result
        # in incorrect gradient computation.
        grad_var_a, grad_var_a_inplace = 0, 1
        with base.dygraph.guard(place=self.place):
            inputs_tensor = copy.copy(self.inputs) if hasattr(self, "inputs") else {}
            attrs_tensor = copy.copy(self.attrs) if hasattr(self, "attrs") else {}
            outputs_tensor = copy.copy(self.outputs) if hasattr(self, "outputs") else {}

            args = self.prepare_agrs(inputs_tensor, attrs_tensor, outputs_tensor)
            var_a = args[0]
            var_a.stop_gradient = False
            var_b = var_a**2
            args[0] = var_b
            var_c = self.cal_python_api(
                args, True
            )  # var_b is modified inplace before using it

            # Here, the gradient computation will use the value of var_b
            var_d = var_c["Out"] ** 2
            loss = var_d.sum()
            loss.backward()
            grad_var_a_inplace = var_a.grad.numpy()

        with base.dygraph.guard(place=self.place):
            inputs_tensor = copy.copy(self.inputs) if hasattr(self, "inputs") else {}
            attrs_tensor = copy.copy(self.attrs) if hasattr(self, "attrs") else {}
            outputs_tensor = copy.copy(self.outputs) if hasattr(self, "outputs") else {}

            args = self.prepare_agrs(inputs_tensor, attrs_tensor, outputs_tensor)
            var_a = args[0]
            var_a.stop_gradient = False

            var_b = var_a**2
            args[0] = var_b
            var_c = self.cal_python_api(args, False)

            var_d = var_c["Out"] ** 2
            loss = var_d.sum()
            loss.backward()
            grad_var_a = var_a.grad.numpy()
        self.assertTrue(np.array_equal(grad_var_a_inplace, grad_var_a))

    def check_backward_success_2(self):
        # Although var_b is modified inplace after using it, it does not used in gradient computation.
        # The inplace operator doesn't result in incorrect gradient computation.
        grad_var_a, grad_var_a_inplace = 0, 1
        with paddle.base.dygraph.guard(place=self.place):
            inputs_tensor = copy.copy(self.inputs) if hasattr(self, "inputs") else {}
            attrs_tensor = copy.copy(self.attrs) if hasattr(self, "attrs") else {}
            outputs_tensor = copy.copy(self.outputs) if hasattr(self, "outputs") else {}

            args = self.prepare_agrs(inputs_tensor, attrs_tensor, outputs_tensor)

            var_a = args[0]

            var_a.stop_gradient = False

            var_b = var_a**2
            args[0] = var_b
            var_c = self.cal_python_api(args, True)[
                "Out"
            ]  # var_b is modified inplace before using it

            var_d = (
                var_c + var_c
            )  # Here, the grad op of sum doesn't use the value of var_b
            loss = var_d.sum()

            loss.backward()
            grad_var_a_inplace = var_a.grad.numpy()

        with paddle.base.dygraph.guard(place=self.place):

            inputs_tensor = copy.copy(self.inputs) if hasattr(self, "inputs") else {}
            attrs_tensor = copy.copy(self.attrs) if hasattr(self, "attrs") else {}
            outputs_tensor = copy.copy(self.outputs) if hasattr(self, "outputs") else {}

            args = self.prepare_agrs(inputs_tensor, attrs_tensor, outputs_tensor)

            var_a = args[0]

            var_a.stop_gradient = False

            var_b = var_a**2
            args[0] = var_b
            var_c = self.cal_python_api(args, False)["Out"]

            var_d = (
                var_c + var_c
            )  # Here, the grad op of sum doesn't use the value of var_b
            loss = var_d.sum()

            loss.backward()
            grad_var_a = var_a.grad.numpy()
        np.testing.assert_array_equal(grad_var_a_inplace, grad_var_a)
