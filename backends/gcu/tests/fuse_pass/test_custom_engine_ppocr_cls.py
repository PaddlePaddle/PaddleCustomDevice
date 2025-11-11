# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import os

os.environ["FLAGS_custom_engine_min_group_size"] = "10"

import numpy as np
import unittest
import paddle

import time
import shutil
import tarfile
import zipfile
import requests

from paddle_custom_device.gcu import passes as gcu_passes

paddle.enable_static()


def download(url, path):
    """
    Download from url, save to path.
    url (str): download url
    path (str): download to given path
    """
    if os.path.exists(path) and os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path)

    fname = os.path.split(url)[-1]
    fullname = os.path.join(path, fname)
    retry_cnt = 0
    while not os.path.exists(fullname):
        if retry_cnt < 5:
            retry_cnt += 1
        else:
            print("{} download failed.".format(fname), flush=True)
            raise RuntimeError(
                "Download from {} failed. " "Retry limit reached".format(url)
            )

        print("Downloading {} from {}".format(fname, url), flush=True)
        req = requests.get(url, stream=True)
        if req.status_code != 200:
            raise RuntimeError(
                "Downloading from {} failed with code "
                "{}!".format(url, req.status_code)
            )

        # For protecting download interupted, download to
        # tmp_fullname firstly, move tmp_fullname to fullname
        # after download finished
        tmp_fullname = fullname + "_tmp"
        total_size = req.headers.get("content-length")
        with open(tmp_fullname, "wb") as f:
            for chunk in req.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        shutil.move(tmp_fullname, fullname)
        print("{} download completed.".format(fname), flush=True)

    return fullname


def decompress(fname):
    """
    Decompress for zip and tar file
    """
    print("Decompressing {}...".format(fname), flush=True)
    fpath = os.path.split(fname)[0]

    if fname.find(".tar") >= 0 or fname.find(".tgz") >= 0:
        with tarfile.open(fname) as tf:

            def is_within_directory(directory, target):
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
                prefix = os.path.commonprefix([abs_directory, abs_target])
                return prefix == abs_directory

            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
                tar.extractall(path, members, numeric_owner=numeric_owner)

            safe_extract(tf, path=fpath)
    elif fname.find(".zip") >= 0:
        with zipfile.ZipFile(fname) as zf:
            zf.extractall(path=fpath)
    else:
        raise TypeError("Unsupport compress file type {}".format(fname))
    print("{} decompressed.".format(fname), flush=True)
    return


def download_and_decompress(url, path="."):
    full_name = download(url, path)
    print("File is donwloaded, now extracting...", flush=True)
    if url.count(".tgz") > 0 or url.count(".tar") > 0 or url.count("zip") > 0:
        decompress(full_name)
    return


class TestPPOCRCls(unittest.TestCase):
    def setUp(self):
        print("TestPPOCRCls setUp:", os.getenv("CUSTOM_DEVICE_ROOT"))
        gcu_passes.setUp()
        url = "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar"
        download_path = "custom_engine_test_models"
        download_and_decompress(url, download_path)

        self.model_dir = download_path + "/ch_ppocr_mobile_v2.0_cls_infer/"
        self.prog_file = self.model_dir + "inference.pdmodel"
        self.params_file = self.model_dir + "inference.pdiparams"

        self.image_shape = [1, 3, 48, 192]
        self.expected_improvement_rate = 2

    def run_ppocr_cls(self, warmup=10, total_steps=100, enable_custom_engine=True):
        paddle.seed(2066)
        np.random.seed(2099)
        config = paddle.inference.Config()
        config.set_prog_file(self.prog_file)
        config.set_params_file(self.params_file)
        config.enable_memory_optim()
        config.enable_custom_device("gcu")

        config.enable_new_ir(True)
        config.enable_new_executor(True)

        if enable_custom_engine:
            kPirGcuPasses = [
                # Functional pass
                "add_shadow_output_after_dead_parameter_pass",
                "delete_quant_dequant_linear_op_pass",
                "delete_weight_dequant_linear_op_pass",
                "map_op_to_another_pass",
                "identity_op_clean_pass",
                # Operator fusion pass
                "conv2d_bn_fuse_pass",
                # custom engine pass
                "gcu_op_marker_pass",
                "gcu_sub_graph_extract_pass",
                "gcu_replace_with_engine_op_pass",
            ]
            config.enable_custom_passes(kPirGcuPasses, True)

        predictor = paddle.inference.create_predictor(config)

        # warm up
        for i in range(warmup):
            # print("Warmup run PPOCRCls {}".format(i), flush=True)
            np_inputs = [
                np.random.random(self.image_shape).astype("float32"),
            ]
            input_names = predictor.get_input_names()
            for i, name in enumerate(input_names):
                input_tensor = predictor.get_input_handle(name)
                input_tensor.copy_from_cpu(np_inputs[i])

            predictor.run()
            output_names = predictor.get_output_names()
            for i, name in enumerate(output_names):
                output_tensor = predictor.get_output_handle(name)
                output_data = output_tensor.copy_to_cpu()

        all_results = []
        total_time = 0
        for i in range(total_steps):
            # print("Run run PPOCRCls {}".format(i), flush=True)
            start = time.time()
            np_inputs = [
                np.random.random(self.image_shape).astype("float32"),
            ]
            input_names = predictor.get_input_names()
            for i, name in enumerate(input_names):
                input_tensor = predictor.get_input_handle(name)
                input_tensor.copy_from_cpu(np_inputs[i])

            predictor.run()
            results = []
            output_names = predictor.get_output_names()
            for i, name in enumerate(output_names):
                output_tensor = predictor.get_output_handle(name)
                output_data = output_tensor.copy_to_cpu()
                results.append(output_data)
            end = time.time()
            total_time += end - start
            all_results.append(results)
        return all_results, total_time

    def test_ppocr_cls(self):
        warmup = 10
        total_steps = 100
        # Skip this testcase temporarily
        return
        custom_engine_result, custom_engine_total_time = self.run_ppocr_cls(
            warmup, total_steps, enable_custom_engine=True
        )
        aot_result, aot_total_time = self.run_ppocr_cls(
            warmup, total_steps, enable_custom_engine=False
        )
        custom_engine_fps = total_steps / custom_engine_total_time
        aot_fps = total_steps / aot_total_time
        print("======= Run PPOCRCls successfully =======", flush=True)
        print(
            "TestPPOCRCls custom_engine, steps: {}, avg cost: {}ms, fps: {}.".format(
                total_steps,
                (custom_engine_total_time * 1000 / total_steps),
                custom_engine_fps,
            ),
            flush=True,
        )
        print(
            "TestPPOCRCls AOT, steps: {}, avg cost: {}ms, fps: {}.".format(
                total_steps, (aot_total_time * 1000 / total_steps), aot_fps
            ),
            flush=True,
        )
        print(
            "The FPS improvement rate of custom_engine compared to AOT is: {:.2f}%.".format(
                (custom_engine_fps / aot_fps * 100)
            ),
            flush=True,
        )
        print("=========================================", flush=True)

        np.testing.assert_allclose(
            custom_engine_result, aot_result, rtol=1e-05, atol=1e-05
        )

        assert custom_engine_fps > (
            aot_fps * self.expected_improvement_rate
        ), "The FPS of custom_engine is expected to be more than {} times that of AOT.".format(
            self.expected_improvement_rate
        )


if __name__ == "__main__":
    unittest.main()
