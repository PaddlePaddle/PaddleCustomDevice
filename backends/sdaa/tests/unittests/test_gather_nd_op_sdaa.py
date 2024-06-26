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

import unittest

import numpy as np
from op_test import OpTest, skip_check_grad_ci

import paddle
from paddle import base


def test_class1(op_type, typename):
    @skip_check_grad_ci(reason="The backward test is not supported yet on sdaa.")
    class TestGatherNdOpWithIndex1(OpTest):
        def setUp(self):
            self.set_sdaa()
            self.place = paddle.CustomPlace("sdaa", 0)
            self.op_type = "gather_nd"
            self.python_api = paddle.gather_nd
            xnp = np.random.random((5, 20)).astype(typename)
            self.inputs = {"X": xnp, "Index": np.array([1]).astype("int32")}
            self.outputs = {"Out": self.inputs["X"][self.inputs["Index"][-1]]}

        def set_sdaa(self):
            self.__class__.use_custom_device = True

        def test_check_output(self):
            self.check_output_with_place(self.place)

    cls_name = "{0}_{1}_1".format(op_type, typename)
    TestGatherNdOpWithIndex1.__name__ = cls_name
    globals()[cls_name] = TestGatherNdOpWithIndex1


def test_class2(op_type, typename):
    @skip_check_grad_ci(reason="The backward test is not supported yet on sdaa.")
    class TestGatherNdOpWithLowIndex(OpTest):
        # Index has low rank, X has high rank

        def setUp(self):
            self.set_sdaa()
            self.place = paddle.CustomPlace("sdaa", 0)
            self.op_type = "gather_nd"
            self.python_api = paddle.gather_nd
            xnp = np.random.uniform(0, 100, (10, 10)).astype(typename)
            index = np.array([[1], [2]]).astype("int64")

            self.inputs = {"X": xnp, "Index": index}
            self.outputs = {"Out": xnp[tuple(index.T)]}

        def set_sdaa(self):
            self.__class__.use_custom_device = True

        def test_check_output(self):
            self.check_output_with_place(self.place)

    cls_name = "{0}_{1}_2".format(op_type, typename)
    TestGatherNdOpWithLowIndex.__name__ = cls_name
    globals()[cls_name] = TestGatherNdOpWithLowIndex


def test_class3(op_type, typename):
    @skip_check_grad_ci(reason="The backward test is not supported yet on sdaa.")
    class TestGatherNdOpWithEmptyIndex(OpTest):
        # Index has empty element, which means copy entire tensor

        def setUp(self):
            self.set_sdaa()
            self.place = paddle.CustomPlace("sdaa", 0)
            self.op_type = "gather_nd"
            self.python_api = paddle.gather_nd
            xnp = np.random.random((5, 20)).astype(typename)
            self.inputs = {"X": xnp, "Index": np.array([[], []]).astype("int32")}
            self.outputs = {"Out": np.vstack((xnp[np.newaxis, :], xnp[np.newaxis, :]))}

        def set_sdaa(self):
            self.__class__.use_custom_device = True

        def test_check_output(self):
            self.check_output_with_place(self.place)

    cls_name = "{0}_{1}_3".format(op_type, typename)
    TestGatherNdOpWithEmptyIndex.__name__ = cls_name
    globals()[cls_name] = TestGatherNdOpWithEmptyIndex


def test_class4(op_type, typename):
    @skip_check_grad_ci(reason="The backward test is not supported yet on sdaa.")
    class TestGatherNdOpWithSameIndexAsX(OpTest):
        # Index has same rank as X's rank

        def setUp(self):
            self.set_sdaa()
            self.place = paddle.CustomPlace("sdaa", 0)
            self.op_type = "gather_nd"
            self.python_api = paddle.gather_nd
            xnp = np.random.uniform(0, 100, (10, 10)).astype(typename)
            index = np.array([[1, 1], [2, 1]]).astype("int64")

            self.inputs = {"X": xnp, "Index": index}
            self.outputs = {"Out": xnp[tuple(index.T)]}

        def set_sdaa(self):
            self.__class__.use_custom_device = True

        def test_check_output(self):
            self.check_output_with_place(self.place)

    cls_name = "{0}_{1}_4".format(op_type, typename)
    TestGatherNdOpWithSameIndexAsX.__name__ = cls_name
    globals()[cls_name] = TestGatherNdOpWithSameIndexAsX


def test_class5(op_type, typename):
    @skip_check_grad_ci(reason="The backward test is not supported yet on sdaa.")
    class TestGatherNdOpWithHighRankSame(OpTest):
        # Both Index and X have high rank, and Rank(Index) = Rank(X)

        def setUp(self):
            self.set_sdaa()
            self.place = paddle.CustomPlace("sdaa", 0)
            self.op_type = "gather_nd"
            self.python_api = paddle.gather_nd
            shape = (5, 2, 3, 1, 10)
            xnp = np.random.rand(*shape).astype(typename)
            index = np.vstack([np.random.randint(0, s, size=2) for s in shape]).T

            self.inputs = {"X": xnp, "Index": index.astype("int32")}
            self.outputs = {"Out": xnp[tuple(index.T)]}

        def set_sdaa(self):
            self.__class__.use_custom_device = True

        def test_check_output(self):
            self.check_output_with_place(self.place)

    cls_name = "{0}_{1}_5".format(op_type, typename)
    TestGatherNdOpWithHighRankSame.__name__ = cls_name
    globals()[cls_name] = TestGatherNdOpWithHighRankSame


def test_class6(op_type, typename):
    @skip_check_grad_ci(reason="The backward test is not supported yet on sdaa.")
    class TestGatherNdOpWithHighRankDiff(OpTest):
        # Both Index and X have high rank, and Rank(Index) < Rank(X)

        def setUp(self):
            self.set_sdaa()
            self.place = paddle.CustomPlace("sdaa", 0)
            self.op_type = "gather_nd"
            self.python_api = paddle.gather_nd
            shape = (2, 3, 4, 1, 10)
            xnp = np.random.rand(*shape).astype(typename)
            index = np.vstack([np.random.randint(0, s, size=200) for s in shape]).T
            index_re = index.reshape([20, 5, 2, 5])

            self.inputs = {"X": xnp, "Index": index_re.astype("int32")}
            self.outputs = {"Out": xnp[tuple(index.T)].reshape([20, 5, 2])}

        def set_sdaa(self):
            self.__class__.use_custom_device = True

        def test_check_output(self):
            self.check_output_with_place(self.place)

    cls_name = "{0}_{1}_6".format(op_type, typename)
    TestGatherNdOpWithHighRankDiff.__name__ = cls_name
    globals()[cls_name] = TestGatherNdOpWithHighRankDiff


def test_class7(op_type, typename):
    @skip_check_grad_ci(
        reason="The backward test is not supported yet for float16 type on sdaa."
    )
    class TestGatherNdOpIndex1(OpTest):
        # Index has low rank, X has high rank

        def setUp(self):
            self.set_sdaa()
            self.place = paddle.CustomPlace("sdaa", 0)
            self.op_type = "gather_nd"
            self.python_api = paddle.gather_nd
            xnp = np.random.uniform(0, 100, (10, 10)).astype(typename)
            index = np.array([1, 2]).astype("int32")
            self.inputs = {"X": xnp, "Index": index}
            self.outputs = {"Out": xnp[tuple(index.T)]}

        def set_sdaa(self):
            self.__class__.use_custom_device = True

        def test_check_output(self):
            self.check_output_with_place(self.place)

    cls_name = "{0}_{1}_7".format(op_type, typename)
    TestGatherNdOpIndex1.__name__ = cls_name
    globals()[cls_name] = TestGatherNdOpIndex1


# Test Python API
class TestGatherNdAPI(unittest.TestCase):
    def test_imperative(self):
        paddle.disable_static()
        paddle.set_device("sdaa")
        input_1 = np.array([[1, 2], [3, 4], [5, 6]]).astype("float32")
        index_1 = np.array([[1]]).astype("int32")
        input = base.dygraph.to_variable(input_1)
        index = base.dygraph.to_variable(index_1)
        output = paddle.gather(input, index)
        output_np = output.numpy()
        expected_output = np.array([3, 4])
        np.testing.assert_allclose(output_np[0], expected_output, rtol=1e-6)
        paddle.enable_static()


for _typename in {"float32", "float16", "int32", "int64"}:
    test_class1("gather_nd", _typename)
    test_class2("gather_nd", _typename)
    test_class3("gather_nd", _typename)
    test_class4("gather_nd", _typename)
    test_class5("gather_nd", _typename)
    test_class6("gather_nd", _typename)
    test_class7("gather_nd", _typename)


@skip_check_grad_ci(reason="The backward test is not supported yet on sdaa.")
class TestGatherNdOpWithEmptyIndex(OpTest):
    # Index has empty element, which means copy entire tensor

    def setUp(self):
        self.set_sdaa()
        self.op_type = "gather_nd"
        self.prim_op_type = "prim"
        self.python_api = paddle.gather_nd
        self.public_python_api = paddle.gather_nd
        self.config_dtype()
        self.if_enable_cinn()
        if self.dtype == np.float64:
            target_dtype = "float64"
        elif self.dtype == np.float16:
            target_dtype = "float16"
        else:
            target_dtype = "float32"
        xnp = np.random.random((5, 20)).astype(target_dtype)
        output = np.vstack((xnp[np.newaxis, :], xnp[np.newaxis, :]))
        self.inputs = {"X": xnp, "Index": np.array([[], []]).astype("int32")}
        self.outputs = {"Out": output}

    def if_enable_cinn(self):
        pass

    def config_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("sdaa", 0)


@skip_check_grad_ci(reason="The backward test is not supported yet on sdaa.")
class TestGatherNdOpWithIndex1(OpTest):
    def setUp(self):
        self.set_sdaa()
        self.op_type = "gather_nd"
        self.prim_op_type = "prim"
        self.python_api = paddle.gather_nd
        self.public_python_api = paddle.gather_nd
        self.config_dtype()
        self.if_enable_cinn()
        if self.dtype == np.float64:
            target_dtype = "float64"
        elif self.dtype == np.float16:
            target_dtype = "float16"
        else:
            target_dtype = "float32"
        xnp = np.random.random((5, 20)).astype(target_dtype)
        index = np.array([1]).astype("int32")
        output = xnp[index[-1]]

        self.inputs = {"X": xnp, "Index": index}
        self.outputs = {"Out": output}

    def if_enable_cinn(self):
        pass

    def config_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("sdaa", 0)


@skip_check_grad_ci(reason="The backward test is not supported yet on sdaa.")
class TestGatherNdOpWithIndex1_ZeroDim(TestGatherNdOpWithIndex1):
    def setUp(self):
        self.set_sdaa()
        self.op_type = "gather_nd"
        self.prim_op_type = "prim"
        self.python_api = paddle.gather_nd
        self.public_python_api = paddle.gather_nd
        self.config_dtype()
        self.if_enable_cinn()
        if self.dtype == np.float64:
            target_dtype = "float64"
        elif self.dtype == np.float16:
            target_dtype = "float16"
        else:
            target_dtype = "float32"
        xnp = np.random.random((100,)).astype(target_dtype)
        index = np.array([1]).astype("int32")
        output = xnp[index[-1]]

        self.inputs = {"X": xnp, "Index": index}
        self.outputs = {"Out": output}

    def if_enable_cinn(self):
        self.enable_cinn = False

    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("sdaa", 0)


@skip_check_grad_ci(reason="The backward test is not supported yet on sdaa.")
class TestGatherNdOpWithLowIndex(OpTest):
    # Index has low rank, X has high rank

    def setUp(self):
        self.set_sdaa()
        self.op_type = "gather_nd"
        self.prim_op_type = "prim"
        self.python_api = paddle.gather_nd
        self.public_python_api = paddle.gather_nd
        self.config_dtype()
        if self.dtype == np.float64:
            target_dtype = "float64"
        elif self.dtype == np.float16:
            target_dtype = "float16"
        else:
            target_dtype = "float32"
        xnp = np.random.uniform(0, 100, (10, 10)).astype(target_dtype)
        index = np.array([[1], [2]]).astype("int64")
        output = xnp[tuple(index.T)]  # shape is [2, 10]

        self.inputs = {"X": xnp, "Index": index}
        self.outputs = {"Out": output}

    def config_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("sdaa", 0)


@skip_check_grad_ci(reason="The backward test is not supported yet on sdaa.")
class TestGatherNdOpIndex1(OpTest):
    # Index has low rank, X has high rank

    def setUp(self):
        self.set_sdaa()
        self.op_type = "gather_nd"
        self.prim_op_type = "prim"
        self.python_api = paddle.gather_nd
        self.public_python_api = paddle.gather_nd
        self.config_dtype()
        if self.dtype == np.float64:
            target_dtype = "float64"
        elif self.dtype == np.float16:
            target_dtype = "float16"
        else:
            target_dtype = "float32"
        xnp = np.random.uniform(0, 100, (10, 10)).astype(target_dtype)
        index = np.array([1, 2]).astype("int32")
        output = xnp[tuple(index.T)]

        self.inputs = {"X": xnp, "Index": index}
        self.outputs = {"Out": output}
        self.if_enable_cinn()

    def if_enable_cinn(self):
        # the outputs are 0D-tensor, CINN not support
        self.enable_cinn = False

    def config_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("sdaa", 0)


@skip_check_grad_ci(reason="The backward test is not supported yet on sdaa.")
class TestGatherNdOpWithSameIndexAsX(OpTest):
    # Index has same rank as X's rank
    def setUp(self):
        self.set_sdaa()
        self.op_type = "gather_nd"
        self.prim_op_type = "prim"
        self.python_api = paddle.gather_nd
        self.public_python_api = paddle.gather_nd
        self.config_dtype()
        if self.dtype == np.float64:
            target_dtype = "float64"
        elif self.dtype == np.float16:
            target_dtype = "float16"
        else:
            target_dtype = "float32"
        xnp = np.random.uniform(0, 100, (10, 10)).astype(target_dtype)
        index = np.array([[1, 1], [2, 1]]).astype("int64")
        output = xnp[tuple(index.T)]  # [25, 22]

        self.inputs = {"X": xnp, "Index": index}
        self.outputs = {"Out": output}

    def config_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("sdaa", 0)


@skip_check_grad_ci(reason="The backward test is not supported yet on sdaa.")
class TestGatherNdOpWithHighRankSame(OpTest):
    # Both Index and X have high rank, and Rank(Index) = Rank(X)

    def setUp(self):
        self.set_sdaa()
        self.op_type = "gather_nd"
        self.prim_op_type = "prim"
        self.python_api = paddle.gather_nd
        self.public_python_api = paddle.gather_nd
        shape = (5, 2, 3, 1, 10)
        self.config_dtype()
        if self.dtype == np.float64:
            target_dtype = "float64"
        elif self.dtype == np.float16:
            target_dtype = "float16"
        else:
            target_dtype = "float32"
        xnp = np.random.rand(*shape).astype(target_dtype)
        index = np.vstack([np.random.randint(0, s, size=2) for s in shape]).T
        output = xnp[tuple(index.T)]

        self.inputs = {"X": xnp, "Index": index.astype("int32")}
        self.outputs = {"Out": output}

    def config_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("sdaa", 0)


@skip_check_grad_ci(reason="The backward test is not supported yet on sdaa.")
class TestGatherNdOpWithHighRankDiff(OpTest):
    # Both Index and X have high rank, and Rank(Index) < Rank(X)

    def setUp(self):
        self.set_sdaa()
        self.op_type = "gather_nd"
        self.prim_op_type = "prim"
        self.python_api = paddle.gather_nd
        self.public_python_api = paddle.gather_nd
        shape = (2, 3, 4, 1, 10)
        self.config_dtype()
        if self.dtype == np.float64:
            target_dtype = "float64"
        elif self.dtype == np.float16:
            target_dtype = "float16"
        else:
            target_dtype = "float32"
        xnp = np.random.rand(*shape).astype(target_dtype)
        index = np.vstack([np.random.randint(0, s, size=200) for s in shape]).T
        index_re = index.reshape([20, 5, 2, 5])
        output = xnp[tuple(index.T)].reshape([20, 5, 2])

        self.inputs = {"X": xnp, "Index": index_re.astype("int32")}
        self.outputs = {"Out": output}

    def config_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("sdaa", 0)


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
