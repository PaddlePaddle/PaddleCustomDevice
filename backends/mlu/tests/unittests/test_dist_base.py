#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import os
import pickle
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import closing

import numpy as np

import paddle
from paddle import fluid

RUN_STEP = 5
DEFAULT_BATCH_SIZE = 2
DIST_UT_PORT = 0


def print_to_out(out_losses):
    sys.stdout.buffer.write(pickle.dumps(out_losses))


def print_to_err(class_name, log_str):
    localtime = time.asctime(time.localtime(time.time()))
    print_str = localtime + "\t" + class_name + "\t" + log_str
    sys.stderr.buffer.write(pickle.dumps(print_str))


class TestParallelDyGraphRunnerBase:
    def get_model(self):
        raise NotImplementedError("get_model should be implemented by child classes.")

    def run_one_loop(self, model, opt, data):
        raise NotImplementedError(
            "train_one_loop should be implemented by the child classes."
        )

    def _get_data(self, batch, args):
        if args.update_method != "local":
            new_batch = []

            for offset, item in enumerate(batch):
                if offset % 2 == args.trainer_id:
                    new_batch.append(item)
            return new_batch
        else:
            return batch

    def run_trainer(self, args):
        seed = 90
        device_id = int(os.getenv("FLAGS_selected_mlus", "0"))
        place = paddle.CustomPlace("mlu", device_id)

        with fluid.dygraph.guard(place):
            fluid.default_startup_program().random_seed = seed
            fluid.default_main_program().random_seed = seed
            np.random.seed(seed)
            import random

            random.seed(seed)
            model, train_reader, opt = self.get_model()
            nranks = len(args.endpoints.split(",")) if args.endpoints else 1

            strategy = paddle.distributed.parallel.ParallelStrategy()
            strategy.nranks = nranks
            strategy.local_rank = args.trainer_id
            strategy.trainer_endpoints = args.endpoints.split(",")
            strategy.current_endpoint = args.current_endpoint
            paddle.distributed.init_parallel_env()
            print_to_err(
                type(self).__name__,
                "begin to prepare context in dygraph",
            )
            if not args.find_unused_parameters:
                model = paddle.DataParallel(
                    model, strategy, find_unused_parameters=False
                )
            else:
                model = paddle.DataParallel(
                    model, strategy, find_unused_parameters=True
                )
            print_to_err(type(self).__name__, "model built in dygraph")

            out_losses = []
            print_to_err(type(self).__name__, "begin to run dygraph training")
            for step_id, data in enumerate(train_reader()):
                data = self._get_data(data, args)
                if step_id == RUN_STEP:
                    break
                loss = self.run_one_loop(model, opt, data)
                if step_id % 10 == 0:
                    print_to_err(
                        type(self).__name__,
                        "loss at step %d: %f" % (step_id, loss.numpy()),
                    )
                out_losses.append(loss.numpy())

                loss.backward()

                opt.minimize(loss)
                model.clear_gradients()
        print_to_out(out_losses)


def runtime_main(test_class):
    parser = argparse.ArgumentParser(description="Run dist test.")
    parser.add_argument(
        "--role", type=str, required=True, choices=["pserver", "trainer"]
    )
    parser.add_argument("--endpoints", type=str, required=False, default="")
    parser.add_argument(
        "--update_method",
        type=str,
        default="local",
        choices=[
            "cncl",
            "local",
        ],
    )
    parser.add_argument("--trainer_id", type=int, required=False, default=0)
    parser.add_argument("--trainers", type=int, required=False, default=1)

    parser.add_argument("--diff_batch", action="store_true")
    parser.add_argument(
        "--hallreduce_inter_nranks", type=int, required=False, default=2
    )
    parser.add_argument("--current_endpoint", type=str, required=False, default="")
    parser.add_argument("--use_cpu", action="store_true")
    parser.add_argument("--use_mlu", action="store_true")
    parser.add_argument("--find_unused_parameters", action="store_true")
    parser.add_argument("--use_reader_alloc", action="store_true", required=False)
    parser.add_argument("--batch_size", required=False, type=int, default=2)
    parser.add_argument("--lr", required=False, type=float, default=0.001)
    parser.add_argument("--batch_merge_repeat", required=False, type=int, default=1)

    parser.add_argument("--sync_batch_norm", action="store_true")

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
        self._use_mlu = False
        self._find_unused_parameters = False
        self._setup_config()

        global DIST_UT_PORT
        if DIST_UT_PORT == 0 and os.getenv("PADDLE_DIST_UT_PORT"):
            DIST_UT_PORT = int(os.getenv("PADDLE_DIST_UT_PORT"))

        if DIST_UT_PORT == 0:
            self._ps_endpoints = "127.0.0.1:{},127.0.0.1:{}".format(
                self._find_free_port(),
                self._find_free_port(),
            )
        else:
            self._ps_endpoints = "127.0.0.1:{},127.0.0.1:{}".format(
                DIST_UT_PORT,
                DIST_UT_PORT + 1,
            )
            DIST_UT_PORT += 2
            self._dist_port = DIST_UT_PORT

        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _find_free_port(self):
        def __free_port():
            with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
                s.bind(("", 0))
                print_to_err(
                    type(self).__name__, "socket name: %s" % s.getsockname()[1]
                )
                return s.getsockname()[1]

        while True:
            port = __free_port()
            if port not in self._port_set:
                self._port_set.add(port)
                return port

    def _run_local(
        self,
        model,
        envs,
        check_error_log=False,
        batch_size=DEFAULT_BATCH_SIZE,
        batch_merge_repeat=1,
        log_name="",
        devices="1",
    ):

        cmd = self._python_interp

        if os.getenv("WITH_COVERAGE", "OFF") == "ON":
            envs["COVERAGE_FILE"] = os.getenv("COVERAGE_FILE", "")
            cmd += " -m coverage run --branch -p"

        cmd += " {} --role trainer --update_method local --lr {:f}".format(
            model,
            self._lr,
        )

        if batch_size != DEFAULT_BATCH_SIZE:
            cmd += " --batch_size %d" % batch_size
        if batch_merge_repeat > 1:
            cmd += " --batch_merge_repeat %d" % batch_merge_repeat

        if self._use_mlu:
            cmd += " --use_mlu"
            env_local = {
                "MLU_VISIBLE_DEVICES": devices,
                "PADDLE_TRAINERS_NUM": "1",
                "PADDLE_TRAINER_ID": "0",
            }
        else:
            env_local = {"CPU_NUM": "1"}

        if self._find_unused_parameters:
            cmd += " --find_unused_parameters"

        if self._find_unused_parameters:
            cmd += " --find_unused_parameters"

        env_local.update(envs)
        print(f"local_cmd: {cmd}, env: {env_local}")

        if check_error_log:
            path = "/tmp/local_err_%d.log" % os.getpid()
            err_log = open(path, "w")
            local_proc = subprocess.Popen(
                cmd.split(" "),
                stdout=subprocess.PIPE,
                stderr=err_log,
                env=env_local,
            )
        else:
            local_proc = subprocess.Popen(
                cmd.split(" "),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env_local,
            )

        local_out, local_err = local_proc.communicate()

        if check_error_log:
            err_log.close()
            sys.stderr.write(
                "\n--run_local-- trainer 0 stderr file saved in: %s\n" % (path)
            )

        sys.stderr.write("local_stderr: %s\n" % local_err)
        sys.stderr.write("local_stdout: %s\n" % pickle.loads(local_out))

        return pickle.loads(local_out)

    def _get_mlu_trainer_cmd(
        self,
        model,
        ep,
        update_method,
        trainer_id,
        trainer_num,
    ):
        env = {}
        tr_cmd = "%s -u"

        if os.getenv("WITH_COVERAGE", "OFF") == "ON":
            tr_cmd += " -m coverage run --branch -p"

        tr_cmd += " %s --role trainer --endpoints %s --trainer_id %d --current_endpoint %s --update_method %s --lr %f"

        tr_cmd = tr_cmd % (
            self._python_interp,
            model,
            self._ps_endpoints,
            trainer_id,
            ep,
            update_method,
            self._lr,
        )

        if self._use_mlu:
            tr_cmd += " --use_mlu"
            env.update(
                {
                    "FLAGS_selected_mlus": f"{0}",
                    "MLU_VISIBLE_DEVICES": f"{trainer_id}",
                    "PADDLE_TRAINERS_NUM": f"{trainer_num}",
                    "PADDLE_XCCL_BACKEND": "mlu",
                    "PADDLE_TRAINER_ID": f"{trainer_id}",
                    "PADDLE_TRAINER_ENDPOINTS": self._ps_endpoints,
                    "PADDLE_CURRENT_ENDPOINT": ep,
                    "PADDLE_GLOBAL_SIZE": f"{trainer_num}",
                    "PADDLE_LOCAL_SIZE": f"{trainer_num}",
                    "PADDLE_GLOBAL_RANK": f"{trainer_id}",
                    "PADDLE_LOCAL_RANK": f"{trainer_id}",
                    "PADDLE_NNODES": "1",
                    "PADDLE_RANK_IN_NODE": "1",
                    "PADDLE_DISTRI_BACKEND": "xccl",
                }
            )
        else:
            env.update({"CPU_NUM": "1"})

        if self._find_unused_parameters:
            tr_cmd += " --find_unused_parameters"

        if os.getenv("WITH_COVERAGE", "OFF") == "ON":
            env["COVERAGE_FILE"] = os.getenv("COVERAGE_FILE", "")

        return tr_cmd, env

    def _run_cluster_mlu(self, model, envs, update_method, check_error_log, log_name):
        worker_endpoints = self._ps_endpoints.split(",")

        trainer_num = len(worker_endpoints)

        procs = []
        pipes = []
        for i in range(0, trainer_num):
            tr_cmd, tr_env = self._get_mlu_trainer_cmd(
                model, worker_endpoints[i], update_method, i, trainer_num
            )
            tr_env.update(envs)
            print("tr_cmd:{}, env: {}".format(tr_cmd, tr_env))

            path = os.path.join(self.temp_dir.name, log_name + f"_tr{i}_err.log")
            tr_pipe = open(path, "wb")

            print_to_err(
                type(self).__name__,
                f"going to start process {i}",
            )
            tr_proc = subprocess.Popen(
                tr_cmd.strip().split(" "),
                stdout=subprocess.PIPE,
                stderr=tr_pipe,
                env=tr_env,
            )

            procs.append(tr_proc)
            pipes.append(tr_pipe)

        outs = []
        for i in range(0, trainer_num):
            tr_out, tr_err = procs[i].communicate()
            outs.append(tr_out)
            pipes[i].close()
            sys.stderr.write(f"trainer {i} stderr: {tr_err}\n")

        if check_error_log:
            print("outs[0]:", outs[0])
            print("outs[1]:", outs[1])

        return pickle.loads(outs[0]), pickle.loads(outs[1])

    def _get_required_envs(self, check_error_log=False, need_envs={}):
        # TODO(typhoonzero): should auto adapt GPU count on the machine.
        required_envs = {
            "PATH": os.getenv("PATH", ""),
            "PYTHONPATH": os.getenv("PYTHONPATH", ""),
            "LD_LIBRARY_PATH": os.getenv("LD_LIBRARY_PATH", ""),
            "FLAGS_rpc_deadline": "30000",  # 5sec to fail fast
            "FLAGS_rpc_retry_bind_port": "50",
            "FLAGS_rpc_disable_reuse_port": "1",
            "http_proxy": "",
            "FLAGS_new_executor_static_build": "1",
        }

        if check_error_log:
            required_envs["GLOG_vmodule"] = (
                "fused_all_reduce_op_handle=10,all_reduce_op_handle=10,alloc_continuous_space_op=10,fuse_all_reduce_op_pass=10,"
                "alloc_continuous_space_for_grad_pass=10,fast_threaded_ssa_graph_executor=10,executor=10,operator=10,"
                "sparse_all_reduce_op_handle=10,grpc_client=10,"
                "grpc_server=10,request_handler_impl=10,section_worker=10"
            )
            required_envs["GLOG_logtostderr"] = "1"

        required_envs.update(need_envs)
        return required_envs

    def check_with_place(
        self,
        model_file,
        delta=1e-3,
        check_error_log=False,
        need_envs={},
        log_name="",
    ):
        self.check_with_place_func(
            model_file=model_file,
            delta=delta,
            check_error_log=check_error_log,
            need_envs=need_envs,
            log_name=log_name,
        )

    def check_with_place_func(
        self,
        model_file,
        delta=1e-3,
        check_error_log=False,
        need_envs={},
        log_name="",
    ):
        required_envs = self._get_required_envs(check_error_log, need_envs)
        local_losses = self._run_local(
            model_file, required_envs, check_error_log, log_name=log_name
        )

        tr0_losses, tr1_losses = self._run_cluster_mlu(
            model_file,
            required_envs,
            update_method="cncl",
            check_error_log=check_error_log,
            log_name=log_name,
        )

        for step_id in range(RUN_STEP):
            local_loss = local_losses[step_id]
            tr0_loss = tr0_losses[step_id]
            tr1_loss = tr1_losses[step_id]
            dist_loss = (np.array([tr0_loss]) + np.array([tr1_loss])) / 2
            print("=======", local_loss, ":", dist_loss[0], "=======")
            self.assertAlmostEqual(local_loss, dist_loss[0], delta=delta)
