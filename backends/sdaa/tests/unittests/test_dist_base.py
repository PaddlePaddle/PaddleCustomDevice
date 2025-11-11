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

import argparse
import os
import subprocess
import sys
import tempfile
import unittest
import re
import numpy as np

import paddle
from paddle import base

RUN_STEP = 10
DEFAULT_BATCH_SIZE = 1
DEVICE_NUMS = 2


def get_loss_value(out_string):
    out_string = out_string[out_string.find(b"out losses:") :].decode()

    pattern = r"array\(([\d\.]+), dtype=float32\)"
    matches = re.findall(pattern, out_string)
    loss_values = [float(match.split(",")[0].strip()) for match in matches]

    return loss_values


def get_avg_loss(loss):
    loss_copy = loss.clone().detach()
    if paddle.distributed.is_initialized():
        paddle.distributed.all_reduce(loss_copy)
        loss_copy /= DEVICE_NUMS

    return loss_copy


class TestParallelDyGraphRunnerBase:
    def get_model(self, use_syncbn):
        raise NotImplementedError("get_model should be implemented by child classes.")

    def run_one_loop(self, model, opt, data):
        raise NotImplementedError(
            "train_one_loop should be implemented by the child classes."
        )

    def _get_data(self, batch, args, rank_id=0):
        if args.update_method != "local":
            new_batch = []

            for offset, item in enumerate(batch):
                if offset % DEVICE_NUMS == rank_id:
                    new_batch.append(item)
            return new_batch
        else:
            return batch

    def run_trainer(self, args):
        seed = 90
        device_id = int(os.getenv("FLAGS_selected_sdaas", "0"))
        place = paddle.CustomPlace("sdaa", device_id)

        with base.dygraph.guard(place):
            base.default_startup_program().random_seed = seed
            base.default_main_program().random_seed = seed
            np.random.seed(seed)
            import random

            random.seed(seed)
            model, train_reader, opt = self.get_model(
                use_syncbn=args.use_syncbn, data_format=args.data_format
            )

            if args.ignore_syncbn_bias_grad:
                for name, param in model.named_parameters():
                    if "_sync_batch_norm.weight" in name:
                        param.stop_gradient = True
                    if "_sync_batch_norm.bias" in name:
                        param.stop_gradient = True

            paddle.distributed.init_parallel_env()
            local_rank = paddle.distributed.get_rank()

            out_losses = []

            model = paddle.DataParallel(model)

            for step_id, data in enumerate(train_reader()):
                data = self._get_data(data, args, local_rank)
                if step_id == RUN_STEP:
                    break
                loss = self.run_one_loop(model, opt, data, data_format=args.data_format)

                loss.backward()

                opt.minimize(loss)
                model.clear_gradients()

                loss_ = get_avg_loss(loss=loss)
                out_losses.append(loss_.numpy())

            print("out losses: ", out_losses)


def runtime_main(test_class):
    parser = argparse.ArgumentParser(description="Run dist test.")
    parser.add_argument(
        "--update_method",
        type=str,
        default="local",
        choices=[
            "tccl",
            "local",
        ],
    )
    parser.add_argument("--use_cpu", action="store_true")
    parser.add_argument("--use_sdaa", action="store_true")
    parser.add_argument("--use_syncbn", action="store_true")
    parser.add_argument("--ignore_syncbn_bias_grad", action="store_true")
    parser.add_argument(
        "--data_format", default="NCHW", type=str, choices=["NCHW", "NHWC"]
    )
    parser.add_argument("--batch_size", required=False, type=int, default=4)
    parser.add_argument("--lr", required=False, type=float, default=0.001)

    args = parser.parse_args()

    model = test_class()
    model.run_trainer(args)


class TestDistBase(unittest.TestCase):
    def _setup_config(self):
        raise NotImplementedError("tests should have _setup_config implemented")

    def setUp(self):
        self._trainers = 2
        self._pservers = 2
        self._port_set = set()
        self._python_interp = sys.executable
        self._lr = 0.001
        self._dygraph = False
        self._use_sdaa = True
        self._find_unused_parameters = False
        self._setup_config()
        self._device_list = [str(i) for i in range(DEVICE_NUMS)]
        self._devices = ",".join(self._device_list)

        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _run_local(
        self,
        model_file,
        envs,
        batch_size=DEFAULT_BATCH_SIZE,
        devices="1",
    ):
        cmd = self._python_interp

        if os.getenv("WITH_COVERAGE", "OFF") == "ON":
            envs["COVERAGE_FILE"] = os.getenv("COVERAGE_FILE", "")
            cmd += " -m coverage run --branch -p"

        cmd += " {} --update_method local --lr {:f}".format(
            model_file,
            self._lr,
        )

        if batch_size != DEFAULT_BATCH_SIZE:
            cmd += " --batch_size %d" % batch_size

        if self._use_sdaa:
            cmd += " --use_sdaa"
            # TODO(zhanggq): If debugging is required, add the appropriate environment variables here.
            env_local = {
                "SDAA_VISIBLE_DEVICES": devices,
            }
        else:
            env_local = {"CPU_NUM": "1"}

        if self._use_NHWC:
            cmd += " --data_format NHWC"

        if self._ignore_syncbn_bias_grad:
            cmd += " --ignore_syncbn_bias_grad"

        env_local.update(envs)
        print(f"local_cmd: {cmd}, env: {env_local}")

        local_proc = subprocess.Popen(
            cmd.split(" "),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env_local,
        )
        local_out, local_err = local_proc.communicate()
        local_out = get_loss_value(local_out)

        return local_out

    def _run_cluster_sdaa(self, model_file, envs, update_method):
        runtime_envs = os.environ
        runtime_envs["SDAA_VISIBLE_DEVICES"] = self._devices
        runtime_envs["PADDLE_DISTRI_BACKEND"] = "xccl"
        runtime_envs["PADDLE_XCCL_BACKEND"] = "sdaa"
        start_command = f"{self._python_interp} -u -m paddle.distributed.launch --devices {self._devices} {model_file} --update_method tccl --use_syncbn"
        if self._use_NHWC:
            start_command += " --data_format NHWC"
        if self._ignore_syncbn_bias_grad:
            start_command += " --ignore_syncbn_bias_grad"
        start_command_list = start_command.strip().split(" ")

        print("dist_cmd:{}, env: {}".format(start_command, runtime_envs))

        global_proc = subprocess.Popen(
            start_command_list,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=runtime_envs,
        )

        global_out, global_err = global_proc.communicate()
        global_out = get_loss_value(global_out)

        return global_out

    def _get_required_envs(self, need_envs={}):
        # TODO(typhoonzero): should auto adapt GPU count on the machine.
        required_envs = {
            "PATH": os.getenv("PATH", ""),
            "PYTHONPATH": os.getenv("PYTHONPATH", ""),
            "LD_LIBRARY_PATH": os.getenv("LD_LIBRARY_PATH", ""),
        }

        required_envs.update(need_envs)
        return required_envs

    def check_with_place(
        self,
        model_file,
        delta=1e-3,
        need_envs={},
    ):
        self.check_with_place_func(
            model_file=model_file,
            delta=delta,
            need_envs=need_envs,
        )

    def check_with_place_func(
        self,
        model_file,
        delta=1e-3,
        need_envs={},
    ):
        required_envs = self._get_required_envs(need_envs)

        local_losses = self._run_local(model_file, required_envs)

        global_losses = self._run_cluster_sdaa(
            model_file, required_envs, update_method="tccl"
        )

        for step_id in range(RUN_STEP):
            local_loss = local_losses[step_id]
            global_loss = global_losses[step_id]
            print(
                "======= local loss: ",
                local_loss,
                "; dis loss: ",
                global_loss,
                "=======",
            )
            np.testing.assert_allclose(
                local_loss,
                global_loss,
                rtol=1e-04,
                atol=5e-03,
            )
