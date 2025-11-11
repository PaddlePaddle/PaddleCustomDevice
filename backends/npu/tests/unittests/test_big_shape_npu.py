# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import json
import ssl
import unittest
import subprocess

import numpy as np
import paddle
from paddle import _C_ops
import urllib.request
import zipfile as zipper

from tests.op_test import OpTest, convert_uint16_to_float, convert_float_to_uint16
from npu_utils import check_run_big_shape_test

ssl._create_default_https_context = ssl._create_unverified_context

device = "npu"
paddle.set_device(device)

ZIP_NAME = "test_tensor"

dtype_map = {
    "DT_BF16": np.uint16,
    "FLOAT": np.float32,
    "FLOAT16": np.float16,
    "BOOL": np.bool_,
    "INT64": np.int64,
}


def download_data(zipname="test_tensor"):
    urlPrefix = "https://sys-p0.bj.bcebos.com/"
    url = urlPrefix + zipname + ".zip"
    urllib.request.urlretrieve(url, zipname + ".zip")
    files = zipper.ZipFile(zipname + ".zip")
    for file in files.namelist():
        files.extract(file, ".")
    files.close()


def delete_data(zipname="test_tensor"):
    subprocess.run("rm -rf " + zipname + "*", shell=True)


def get_tensor(case_name, filenames=None):
    dir_ = ZIP_NAME + "/" + case_name + "/"
    res = []
    for file in filenames:
        res.append(np.load(dir_ + file))
    return res


def do_compare(case_name, npu_res, gpu_res, rtol=1e-4, atol=1e-4):
    if npu_res.dtype == np.bool_:
        result = np.equal(npu_res, gpu_res)
        if result.all() is False:
            np.testing.assert_(False, msg=f"[big shape case]:{case_name} failed!")
    elif npu_res.dtype == np.uint16:
        np.testing.assert_allclose(
            convert_uint16_to_float(npu_res),
            convert_uint16_to_float(gpu_res),
            rtol=rtol,
            atol=atol,
        )
    else:
        np.testing.assert_allclose(npu_res, gpu_res, rtol=rtol, atol=atol)


@check_run_big_shape_test()
class TestBigShape(OpTest):
    @classmethod
    def load_json(cls):
        with open("big_shape_cases.json", "r") as f:
            cls.case_json = json.load(f)

    @classmethod
    def setUpClass(cls):
        super(TestBigShape, cls).setUpClass()
        download_data(ZIP_NAME)
        cls.load_json()

    def set_npu(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("npu", 0)

    def setUp(self):
        self.op_func_map = {
            "add": self._add_func,
            "clip": self._clip_func,
            "concat": self._concat_func,
            "divide": self._divide_func,
            "equal": self._equal_func,
            "exp": self._exp_func,
            "matmul": self._matmul_func,
            "multiply": self._multiply_func,
            "silu": self._silu_func,
            "squared_l2_norm": self._squared_l2_norm_func,
            "sum": self._sum_func,
            "tril": self._tril_func,
            "triu": self._triu_func,
            "where": self._where_func,
        }

    @classmethod
    def tearDownClass(cls):
        delete_data(ZIP_NAME)

    def _add_func(self, data0, data1):
        input_0 = paddle.to_tensor(data0)
        input_1 = paddle.to_tensor(data1)
        res = paddle.add(input_0, input_1)
        return res.numpy()

    def _clip_func(self, data0, data1, data2):
        input_0 = paddle.to_tensor(data0)
        input_1 = paddle.to_tensor(data1)
        input_2 = paddle.to_tensor(data2)
        if input_2 < input_1:
            input_1, input_2 = input_2, input_1
        res = paddle.clip(input_0, input_1, input_2)
        return res.numpy()

    def _squared_l2_norm_func(self, data0):
        input_0 = paddle.to_tensor(data0)
        res = _C_ops.squared_l2_norm(input_0)
        return res.numpy()

    def _concat_func(self, inputs_tensor):
        inputs = [paddle.to_tensor(tensor) for tensor in inputs_tensor]
        res = paddle.concat(inputs, axis=-1)
        return res.numpy()

    def _divide_func(self, data0, data1):
        input_0 = paddle.to_tensor(data0)
        input_1 = paddle.to_tensor(data1)
        res = paddle.divide(input_0, input_1)
        return res.numpy()

    def _equal_func(self, data0, data1):
        input_0 = paddle.to_tensor(data0)
        input_1 = paddle.to_tensor(data1)
        res = paddle.equal(input_0, input_1)
        return res.numpy()

    def _exp_func(self, data0):
        input_0 = paddle.to_tensor(data0)
        res = paddle.exp(input_0)
        return res.numpy()

    def _matmul_func(self, case_id, data0, data1, transpose_x, transpose_y):
        input_0 = paddle.to_tensor(data0)
        input_1 = paddle.to_tensor(data1)
        res = paddle.matmul(input_0, input_1, transpose_x, transpose_y)
        return res.numpy()

    def _multiply_func(self, data0, data1):
        input_0 = paddle.to_tensor(data0)
        input_1 = paddle.to_tensor(data1)
        res = paddle.multiply(input_0, input_1)
        return res.numpy()

    def _silu_func(self, data0):
        input_0 = paddle.to_tensor(data0, stop_gradient=False)
        res = paddle.nn.Silu()(input_0)
        res.backward()
        grad_res = input_0.grad
        return res.numpy(), grad_res.numpy()

    def _sum_func(self, data0, data1):
        input_0 = paddle.to_tensor(data0)
        input_1 = paddle.to_tensor(data1)
        res = paddle.sum(input_0, axis=input_1)
        return res.numpy()

    def _tril_func(self, data0):
        input_0 = paddle.to_tensor(data0)
        res = paddle.tril(input_0)
        return res.numpy()

    def _triu_func(self, data0):
        input_0 = paddle.to_tensor(data0)
        res = paddle.triu(input_0)
        return res.numpy()

    def _where_func(self, data0, data1, data2):
        input_0 = paddle.to_tensor(data0)
        input_1 = paddle.to_tensor(data1)
        input_2 = paddle.to_tensor(data2)
        res = paddle.where(input_0, x=input_1, y=input_2)
        return res.numpy()

    def get_input_and_gpu_output(self, op_name, case, case_name, index):
        input_count = len(case["input_shapes"])
        if op_name == "silu":
            gpu_output_names = [
                f"result_tensor_{index}_0.npy",
                f"result_tensor_{index}_1.npy",
            ]
        else:
            gpu_output_names = [f"result_tensor_{index}.npy"]
        filenames = [f"tensor{i}.npy" for i in range(input_count)] + gpu_output_names
        files = get_tensor(case_name, filenames)
        inputs = files[:input_count]
        gpu_res = files[input_count:]
        return gpu_res, inputs

    def run_and_compare(self, op_name, case):
        case_id = case["case_id"]
        case_name = f"case_{op_name}_{case_id}"
        dtypes = [[dtype_map[j] for j in i] for i in case["dtypes"]]

        for index, dtype_list in enumerate(dtypes):
            print(f"Run Case: {op_name}({case_id}) {dtype_list}")
            gpu_res, input_tensors = self.get_input_and_gpu_output(
                op_name, case, case_name, index
            )
            cast_to_bf16 = np.uint16 in dtype_list
            if cast_to_bf16:
                for i, input_tensor in enumerate(input_tensors):
                    if dtype_list[i] != np.uint16:
                        continue
                    input_tensors[i] = convert_float_to_uint16(input_tensor)
                rtol = 4e-3
                atol = 4e-3
            else:
                rtol = 1e-4
                atol = 1e-4
            func = self.op_func_map.get(op_name)
            if op_name == "concat":
                npu_res = func(input_tensors)
            elif op_name == "matmul":
                npu_res = func(
                    case_id, *input_tensors, case["transpose_x"], case["transpose_y"]
                )
                rtol = 8e-3 if cast_to_bf16 else 3e-4
                atol = 8e-3 if cast_to_bf16 else 3e-4
            else:
                npu_res = func(*input_tensors)

            if len(gpu_res) > 1:
                for i in range(len(gpu_res)):
                    do_compare(case_name, npu_res[i], gpu_res[i], rtol, atol)
            else:
                do_compare(case_name, npu_res, gpu_res[0], rtol, atol)

    def _run_test(self, op_name):
        cases = self.case_json[op_name]
        for case in cases:
            self.run_and_compare(op_name, case)

    def test_add(self):
        self._run_test("add")

    def test_clip(self):
        self._run_test("clip")

    def test_squared_l2_norm(self):
        self._run_test("squared_l2_norm")

    def test_concat(self):
        self._run_test("concat")

    def test_divide(self):
        self._run_test("divide")

    def test_equal(self):
        self._run_test("equal")

    def test_exp(self):
        self._run_test("exp")

    def test_matmul(self):
        self._run_test("matmul")

    def test_multiply(self):
        self._run_test("multiply")

    def test_silu(self):
        self._run_test("silu")

    def test_sum(self):
        self._run_test("sum")

    def test_tril(self):
        self._run_test("tril")

    def test_triu(self):
        self._run_test("triu")

    def test_where(self):
        self._run_test("where")


if __name__ == "__main__":
    unittest.main()
