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

import itertools
import os
import subprocess
import sys
import unittest
import numpy as np
from op_test import convert_float_to_uint16


def get_device_count():
    import paddle

    dev_cnt = [
        dev.split(":")[0] == "sdaa" for dev in paddle.device.get_available_device()
    ].count(True)
    if dev_cnt < 2:
        return
    dev_cnt = 4 if dev_cnt > 4 else dev_cnt
    return dev_cnt


DEVICE_COUNT = 2


class CommunicationTestDistBase(unittest.TestCase):
    def setUp(self, save_log_dir=None, num_of_devices=2, timeout=120):
        self._python_interp = sys.executable
        self._save_log_dir = save_log_dir
        self._num_of_devices = num_of_devices
        self._device_list = [str(i) for i in range(num_of_devices)]
        self._timeout = timeout
        self._seeds = [i + 10 for i in range(num_of_devices)]
        self._devices = ",".join(self._device_list)

    def run_test_case(self, script_file, user_defined_envs=None):
        runtime_envs = os.environ
        runtime_envs.update(user_defined_envs)
        runtime_envs["SDAA_VISIBLE_DEVICES"] = self._devices
        runtime_envs["SDAA_VISIBLE_DEVICES"] = self._devices
        runtime_envs["PADDLE_DISTRI_BACKEND"] = "xccl"
        runtime_envs["PADDLE_XCCL_BACKEND"] = "sdaa"
        start_command = f"{self._python_interp} -u -m paddle.distributed.launch --devices {self._devices} {script_file}"
        start_command_list = start_command.strip().split()

        try:
            self._launcher = subprocess.run(
                start_command_list,
                env=runtime_envs,
                timeout=self._timeout,
                check=True,
            )
        except subprocess.TimeoutExpired as err:
            raise TimeoutError(
                "Timeout while running command {}, try to set a longer period, {} is not enough.".format(
                    err.cmd, err.timeout
                )
            )
        except subprocess.CalledProcessError as err:
            raise RuntimeError(
                "Error occurs when running this test case. The return code of command {} is {}".format(
                    err.cmd, err.returncode
                )
            )


def gen_product_envs_list(default_envs, changeable_envs):
    envs_list = []
    for values in itertools.product(*changeable_envs.values()):
        envs = dict(zip(changeable_envs.keys(), values))
        envs.update(default_envs)
        envs_list.append(envs)
    return envs_list


def create_bool_test_data(shape=None, seed=None):
    if seed:
        np.random.seed(seed)
    data = np.random.choice([True, False], size=shape)
    return data


def create_float_test_data(shape=None, dtype=None, seed=None):
    if seed:
        np.random.seed(seed)
    data = np.random.random(shape).astype(dtype)
    return data


def create_bfloat16_test_data(shape=None, seed=None):
    if seed:
        np.random.seed(seed)
    data = np.random.uniform(-100.0, 100.0, shape).astype("float32")
    data = convert_float_to_uint16(data)
    return data


def create_int_test_data(shape=None, dtype=None, seed=None):
    if seed:
        np.random.seed(seed)
    data = np.random.randint(0, high=12, size=shape).astype(dtype)
    return data


def create_complex_test_data(shape=None, dtype=None, seed=None):
    if seed:
        np.random.seed(seed)
    data = np.random.random(shape).astype(dtype)
    data.imag = np.random.random(shape)
    return data


def create_pyobject_test_data(shape=None, seed=None):
    if seed:
        np.random.seed(seed)
    list_shape = np.random.randint(0, high=100, size=(2)).tolist()
    list_data = np.random.random(shape).tolist()
    dict_key = [i for i in range(0, shape[0])]
    dict_val = np.random.random(shape).tolist()
    dict_data = dict(zip(dict_key, dict_val))
    return [list_data, dict_data]


def create_test_data(shape=None, dtype=None, seed=None):
    assert shape, "Shape should be specified"
    if dtype == "float32" or dtype == "float16" or dtype == "float64":
        return create_float_test_data(shape=shape, dtype=dtype, seed=seed)
    elif dtype == "bfloat16":
        # since numpy does not support bfloat16 yet, use `paddle_bfloat` to replace
        return create_bfloat16_test_data(shape=shape, seed=seed)
        # return create_float_test_data(shape=shape, dtype=bfloat16, seed=seed)
    elif dtype == "bool":
        return create_bool_test_data(shape=shape, seed=seed)
    elif dtype == "int32" or dtype == "int64" or dtype == "int8" or dtype == "uint8":
        return create_int_test_data(shape=shape, dtype=dtype, seed=seed)
    elif dtype == "complex64" or dtype == "complex128":
        return create_complex_test_data(shape=shape, dtype=dtype, seed=seed)
    elif dtype == "pyobject":
        return create_pyobject_test_data(shape=shape, seed=seed)
    else:
        raise NotImplementedError("Unsupported dtype for creating test data.")
