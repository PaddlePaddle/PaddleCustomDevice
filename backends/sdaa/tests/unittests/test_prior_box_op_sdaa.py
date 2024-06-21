# BSD 3- Clause License Copyright (c) 2024, Tecorigin Co., Ltd. All rights
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

import math
import numpy as np
import unittest

from op_test import OpTest
import paddle


def python_prior_box(
    input,
    image,
    min_sizes,
    max_sizes=None,
    aspect_ratios=[1.0],
    variances=[0.1, 0.1, 0.2, 0.2],
    flip=False,
    clip=False,
    step_w=0,
    step_h=0,
    offset=0.5,
    min_max_aspect_ratios_order=False,
    name=None,
):
    return paddle.vision.ops.prior_box(
        input,
        image,
        min_sizes=min_sizes,
        max_sizes=max_sizes,
        aspect_ratios=aspect_ratios,
        variance=variances,
        flip=flip,
        clip=clip,
        steps=[step_w, step_h],
        offset=offset,
        name=name,
        min_max_aspect_ratios_order=min_max_aspect_ratios_order,
    )


class TestPriorBoxOp(OpTest):
    def set_data(self):
        self.place = paddle.CustomPlace("sdaa", 0)
        self.init_test_params()
        self.init_test_input()
        self.init_test_output()
        self.inputs = {"Input": self.input, "Image": self.image}

        self.attrs = {
            "min_sizes": self.min_sizes,
            "aspect_ratios": self.aspect_ratios,
            "variances": self.variances,
            "flip": self.flip,
            "clip": self.clip,
            "step_w": self.step_w,
            "step_h": self.step_h,
            "offset": self.offset,
            "min_max_aspect_ratios_order": self.min_max_aspect_ratios_order,
        }
        if len(self.max_sizes) > 0:
            self.attrs["max_sizes"] = self.max_sizes

        self.outputs = {"Boxes": self.out_boxes, "Variances": self.out_var}

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def setUp(self):
        self.op_type = "prior_box"
        self.python_api = python_prior_box
        self.set_sdaa()
        self.set_data()

    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def set_max_sizes(self):
        max_sizes = [5, 10]
        self.max_sizes = np.array(max_sizes).astype("float32").tolist()

    def set_min_max_aspect_ratios_order(self):
        self.min_max_aspect_ratios_order = False

    def init_test_params(self):
        self.layer_w = 32
        self.layer_h = 32

        self.image_w = 40
        self.image_h = 40

        self.step_w = float(self.image_w) / float(self.layer_w)
        self.step_h = float(self.image_h) / float(self.layer_h)

        self.input_channels = 2
        self.image_channels = 3
        self.batch_size = 10

        self.min_sizes = [2, 4]
        self.min_sizes = np.array(self.min_sizes).astype("float32").tolist()
        self.set_max_sizes()
        self.aspect_ratios = [2.0, 3.0]
        self.flip = True
        self.set_min_max_aspect_ratios_order()
        self.real_aspect_ratios = [1, 2.0, 1.0 / 2.0, 3.0, 1.0 / 3.0]
        self.variances = [0.1, 0.1, 0.2, 0.2]
        self.variances = np.array(self.variances, dtype=np.float64).flatten()

        self.clip = True
        self.num_priors = len(self.real_aspect_ratios) * len(self.min_sizes)
        if len(self.max_sizes) > 0:
            self.num_priors += len(self.max_sizes)
        self.offset = 0.5

    def init_test_input(self):
        self.image = np.random.random(
            (self.batch_size, self.image_channels, self.image_w, self.image_h)
        ).astype("float32")

        self.input = np.random.random(
            (self.batch_size, self.input_channels, self.layer_w, self.layer_h)
        ).astype("float32")

    def init_test_output(self):
        out_dim = (self.layer_h, self.layer_w, self.num_priors, 4)
        out_boxes = np.zeros(out_dim).astype("float32")
        out_var = np.zeros(out_dim).astype("float32")

        idx = 0
        for h in range(self.layer_h):
            for w in range(self.layer_w):
                c_x = (w + self.offset) * self.step_w
                c_y = (h + self.offset) * self.step_h
                idx = 0
                for s in range(len(self.min_sizes)):
                    min_size = self.min_sizes[s]
                    if not self.min_max_aspect_ratios_order:
                        # rest of priors
                        for r in range(len(self.real_aspect_ratios)):
                            ar = self.real_aspect_ratios[r]
                            c_w = min_size * math.sqrt(ar) / 2
                            c_h = (min_size / math.sqrt(ar)) / 2
                            out_boxes[h, w, idx, :] = [
                                (c_x - c_w) / self.image_w,
                                (c_y - c_h) / self.image_h,
                                (c_x + c_w) / self.image_w,
                                (c_y + c_h) / self.image_h,
                            ]
                            idx += 1

                        if len(self.max_sizes) > 0:
                            max_size = self.max_sizes[s]
                            # second prior: aspect_ratio = 1,
                            c_w = c_h = math.sqrt(min_size * max_size) / 2
                            out_boxes[h, w, idx, :] = [
                                (c_x - c_w) / self.image_w,
                                (c_y - c_h) / self.image_h,
                                (c_x + c_w) / self.image_w,
                                (c_y + c_h) / self.image_h,
                            ]
                            idx += 1
                    else:
                        c_w = c_h = min_size / 2.0
                        out_boxes[h, w, idx, :] = [
                            (c_x - c_w) / self.image_w,
                            (c_y - c_h) / self.image_h,
                            (c_x + c_w) / self.image_w,
                            (c_y + c_h) / self.image_h,
                        ]
                        idx += 1
                        if len(self.max_sizes) > 0:
                            max_size = self.max_sizes[s]
                            # second prior: aspect_ratio = 1,
                            c_w = c_h = math.sqrt(min_size * max_size) / 2
                            out_boxes[h, w, idx, :] = [
                                (c_x - c_w) / self.image_w,
                                (c_y - c_h) / self.image_h,
                                (c_x + c_w) / self.image_w,
                                (c_y + c_h) / self.image_h,
                            ]
                            idx += 1

                        # rest of priors
                        for r in range(len(self.real_aspect_ratios)):
                            ar = self.real_aspect_ratios[r]
                            if abs(ar - 1.0) < 1e-6:
                                continue
                            c_w = min_size * math.sqrt(ar) / 2
                            c_h = (min_size / math.sqrt(ar)) / 2
                            out_boxes[h, w, idx, :] = [
                                (c_x - c_w) / self.image_w,
                                (c_y - c_h) / self.image_h,
                                (c_x + c_w) / self.image_w,
                                (c_y + c_h) / self.image_h,
                            ]
                            idx += 1

        # clip the prior's coordidate such that it is within[0, 1]
        if self.clip:
            out_boxes = np.clip(out_boxes, 0.0, 1.0)
        # set the variance.
        out_var = np.tile(
            self.variances, (self.layer_h, self.layer_w, self.num_priors, 1)
        )
        self.out_boxes = out_boxes.astype("float32")
        self.out_var = out_var.astype("float32")


class TestPriorBoxOpWithoutMaxSize(TestPriorBoxOp):
    def set_max_sizes(self):
        self.max_sizes = []


class TestPriorBoxOpWithSpecifiedOutOrder(TestPriorBoxOp):
    def set_min_max_aspect_ratios_order(self):
        self.min_max_aspect_ratios_order = True


class TestPriorBoxAPI(unittest.TestCase):
    def setUp(self):
        self.place = paddle.CustomPlace("sdaa", 0)
        np.random.seed(678)
        self.input_np = np.random.rand(2, 10, 32, 32).astype("float32")
        self.image_np = np.random.rand(2, 10, 40, 40).astype("float32")
        self.min_sizes = [2.0, 4.0]

    def test_dygraph_with_static(self):
        paddle.enable_static()
        input = paddle.static.data(name="input", shape=[2, 10, 32, 32], dtype="float32")
        image = paddle.static.data(name="image", shape=[2, 10, 40, 40], dtype="float32")

        box, var = paddle.vision.ops.prior_box(
            input=input,
            image=image,
            min_sizes=self.min_sizes,
            clip=True,
            flip=True,
        )

        exe = paddle.static.Executor()
        box_np, var_np = exe.run(
            paddle.static.default_main_program(),
            feed={
                "input": self.input_np,
                "image": self.image_np,
            },
            fetch_list=[box, var],
        )

        paddle.disable_static()
        inputs_dy = paddle.to_tensor(self.input_np)
        image_dy = paddle.to_tensor(self.image_np)

        box_dy, var_dy = paddle.vision.ops.prior_box(
            input=inputs_dy,
            image=image_dy,
            min_sizes=self.min_sizes,
            clip=True,
            flip=True,
        )
        box_dy_np = box_dy.numpy()
        var_dy_np = var_dy.numpy()

        np.testing.assert_allclose(box_np, box_dy_np)
        np.testing.assert_allclose(var_np, var_dy_np)
        paddle.enable_static()


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
