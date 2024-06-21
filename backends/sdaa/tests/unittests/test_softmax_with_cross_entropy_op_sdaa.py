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

import unittest
import numpy as np

from op_test import OpTest
import paddle

import paddle.nn.functional as F
from paddle import nn
from paddle.io import DataLoader, Dataset

paddle.enable_static()
SEED = 2021


def stable_softmax(x):
    # Compute the softmax of vector x in a numerically stable way.
    # clip to shiftx, otherwise, when calc loss with
    # log(exp(shiftx)), may get log(0)=INF
    shiftx = (x - np.max(x)).clip(-64.0)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)


def cross_entropy(softmax, label, soft_label, axis, ignore_index=-1):
    if soft_label:
        return (-label * np.log(softmax)).sum(axis=axis, keepdims=True)
    shape = softmax.shape
    axis %= len(shape)
    n = int(np.prod(shape[:axis]))
    axis_dim = shape[axis]
    remain = int(np.prod(shape[axis + 1 :]))
    softmax_reshape = softmax.reshape((n, axis_dim, remain))
    label_reshape = label.reshape((n, 1, remain))
    result = np.zeros_like(label_reshape, dtype=softmax.dtype)
    for i in range(n):
        for j in range(remain):
            lbl = label_reshape[i, 0, j]
            if lbl != ignore_index:
                result[i, 0, j] -= np.log(softmax_reshape[i, lbl, j])
    return result.reshape(label.shape)


def python_core_api(
    logits,
    label,
    soft_label=False,
    use_softmax=False,
    numeric_stable_mode=True,
    ignore_index=-100,
    axis=-1,
):
    # the API paddle.nn.functional.softmax_with_cross_entropy cannot
    # set use_softmax=False, so add a core api manually
    # assert use_softmax is False
    softmax, loss = paddle._C_ops.cross_entropy_with_softmax(
        logits, label, soft_label, use_softmax, numeric_stable_mode, ignore_index, axis
    )
    return loss


class TestSoftmaxWithCrossEntropyOp(OpTest):
    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def init_dtype(self):
        self.dtype = np.float32

    def init_params(self):
        self.numeric_stable_mode = True
        self.python_api = python_core_api
        self.python_out_sig = ["Loss"]
        self.soft_label = False
        self.init_dtype()
        self.axis = -1
        self.ignore_index = -1
        self.shape = [41, 37]
        self.use_softmax = True
        np.random.seed(SEED)

    def init_logits(self):
        self.logits = getattr(
            self, "logits", np.random.uniform(0.1, 1.0, self.shape).astype(self.dtype)
        )
        self.softmax = np.apply_along_axis(stable_softmax, self.axis, self.logits)

    def init_label(self):
        if self.soft_label:
            self.labels = np.random.uniform(0.1, 1.0, self.shape).astype(self.dtype)
            self.labels /= np.sum(self.labels, axis=self.axis, keepdims=True)
        else:
            axis_dim = self.shape[self.axis]
            self.shape[self.axis] = 1
            self.labels = np.random.randint(0, axis_dim, self.shape, dtype="int64")

    def setUp(self):
        self.set_sdaa()
        self.op_type = "softmax_with_cross_entropy"
        self.place = paddle.CustomPlace("sdaa", 0)
        self.init_params()
        self.init_logits()
        self.init_label()

        loss = cross_entropy(
            self.softmax, self.labels, self.soft_label, self.axis, self.ignore_index
        )

        if self.use_softmax == False:  # noqa
            self.inputs = {"Logits": self.softmax, "Label": self.labels}
        else:
            self.inputs = {"Logits": self.logits, "Label": self.labels}

        self.outputs = {
            "Softmax": self.softmax.astype(self.dtype),
            "Loss": loss.astype(self.dtype),
        }
        self.attrs = {
            "numeric_stable_mode": self.numeric_stable_mode,
            "soft_label": self.soft_label,
            "ignore_index": self.ignore_index,
            "use_softmax": self.use_softmax,
        }

        if self.axis != -1:
            self.attrs["axis"] = self.axis

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        # fp32 has low precision, cpu and sdaa both need to relax the max_relative_error if using fp32
        self.check_grad_with_place(
            self.place,
            ["Logits"],
            "Loss",
            numeric_grad_delta=0.001,
            max_relative_error=0.5,
        )


class TestSoftmaxWithCrossEntropyOpNoSoftmax(TestSoftmaxWithCrossEntropyOp):
    def init_params(self):
        self.numeric_stable_mode = True
        self.python_api = python_core_api
        self.python_out_sig = ["Loss"]
        self.soft_label = False
        self.init_dtype()
        self.axis = -1
        self.ignore_index = -1
        self.shape = [41, 37]
        self.use_softmax = False
        np.random.seed(SEED)


class TestSoftmaxWithCrossEntropyOpOneHot(TestSoftmaxWithCrossEntropyOp):
    def init_params(self):
        self.numeric_stable_mode = True
        self.python_api = python_core_api
        self.python_out_sig = ["Loss"]
        self.soft_label = True
        self.init_dtype()
        self.axis = -1
        self.ignore_index = -1
        self.shape = [41, 37]
        self.use_softmax = True
        np.random.seed(SEED)

    # initialize one hot label
    def init_label(self):
        batch_size, class_num = self.shape
        self.label_index = np.random.randint(0, class_num, (batch_size))
        self.labels = np.zeros(self.logits.shape).astype(self.dtype)
        self.labels[np.arange(batch_size), self.label_index] = 1


class TestSoftmaxWithCrossEntropyOpSoftLabel(TestSoftmaxWithCrossEntropyOp):
    def init_params(self):
        self.numeric_stable_mode = True
        self.python_api = python_core_api
        self.python_out_sig = ["Loss"]
        self.soft_label = True
        self.init_dtype()
        self.axis = -1
        self.ignore_index = -1
        self.shape = [41, 37]
        self.use_softmax = True
        np.random.seed(SEED)


class TestPowNet(unittest.TestCase):
    def _test(self, run_sdaa=True):
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        main_prog.random_seed = SEED
        startup_prog.random_seed = SEED
        np.random.seed(SEED)

        a_np = np.random.random(size=(32, 32)).astype("float32")
        b_np = np.random.random(size=(32, 32)).astype("float32")
        label_np = np.random.randint(2, size=(32, 1)).astype("int64")

        with paddle.static.program_guard(main_prog, startup_prog):
            a = paddle.static.data(name="a", shape=[32, 32], dtype="float32")
            b = paddle.static.data(name="b", shape=[32, 32], dtype="float32")
            label = paddle.static.data(name="label", shape=[32, 1], dtype="int64")

            sum = paddle.add(a, b)
            z = paddle.pow(sum, 2.0)

            fc_1 = paddle.static.nn.fc(x=z, size=128)
            prediction = paddle.static.nn.fc(x=fc_1, size=2)

            cost = paddle.nn.functional.cross_entropy(input=prediction, label=label)
            loss = paddle.mean(cost)
            # TODO(zhanggq): Use of the momentum optimizer results in a loss of precision
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

        self.assertTrue(np.allclose(sdaa_pred, cpu_pred, rtol=1e-3))
        self.assertTrue(np.allclose(sdaa_loss, cpu_loss, rtol=1e-3))


class RandomDataset(Dataset):
    def __init__(self, input_size, num_samples):
        self.num_samples = num_samples
        self.input_size = input_size

    def __getitem__(self, idx):
        image = np.random.random([self.input_size]).astype("float32")
        label = np.random.randint(0, self.input_size - 1, (1,)).astype("int64")
        return image, label

    def __len__(self):
        return self.num_samples


class SimpleLayer(nn.Layer):
    def __init__(self, input_size):
        super().__init__()
        shape1 = [input_size, 512]
        shape2 = [512, input_size]
        self.linear1 = nn.Linear(*shape1)
        self.silu1 = nn.Silu()
        self.linear2 = nn.Linear(*shape2)
        self.silu2 = nn.Silu()

    def forward(self, image, lable=None):
        t1 = self.silu1(self.linear1(image))
        return self.silu2(self.linear2(t1))


class SimpleNet(nn.Layer):
    def __init__(self, input_size):
        super().__init__()
        self.layer_list = [SimpleLayer(input_size) for i in range(4)]
        self.layers = nn.LayerList(self.layer_list)

    def forward(self, image, label=None):
        for idx, (layer) in enumerate(self.layers):
            image = layer(image)

        return image


# cross_entropy stability check
class TestSimpleNet(unittest.TestCase):
    def run_simple_net_training(self):
        paddle.set_device("sdaa")
        paddle.disable_static()

        paddle.seed(SEED)
        np.random.seed(SEED)

        input_size = 10
        bs = 1
        dataset = RandomDataset(input_size, 20 * bs)
        simple_net = SimpleNet(input_size)
        loader = DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=True)
        opt = paddle.optimizer.Adam(
            learning_rate=1e-3,
            parameters=simple_net.parameters(),
            # use_multi_tensor=True,
            grad_clip=nn.ClipGradByGlobalNorm(1.0),
        )

        losses = []
        for i, (image, label) in enumerate(loader()):
            out = simple_net(image)

            loss = F.cross_entropy(out, label)
            avg_loss = paddle.mean(loss)
            avg_loss.backward()

            opt.step()

            simple_net.clear_gradients()

            loss_cpu = avg_loss.numpy()
            losses.append(loss_cpu)

        paddle.enable_static()

        return losses

    def test(self):
        # same input run twice, loss must totally equal.
        losses1 = self.run_simple_net_training()
        losses2 = self.run_simple_net_training()

        for i in range(0, len(losses1)):
            loss1 = losses1[i]
            loss2 = losses2[i]

            print(f"loss1: {loss1}, loss2: {loss2}")

            np.testing.assert_equal(loss1, loss2)


if __name__ == "__main__":
    unittest.main()
