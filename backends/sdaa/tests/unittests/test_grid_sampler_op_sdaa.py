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

try:
    from paddle.fluid import dygraph
except ImportError:
    from paddle.base import dygraph


import paddle

SEED = 2022


def AffineGrid(theta, grid_shape):
    n = grid_shape[0]
    h = grid_shape[1]
    w = grid_shape[2]
    h_idx = np.repeat(np.linspace(-1, 1, h)[np.newaxis, :], w, axis=0).T[
        :, :, np.newaxis
    ]
    w_idx = np.repeat(np.linspace(-1, 1, w)[np.newaxis, :], h, axis=0)[:, :, np.newaxis]
    grid = np.concatenate([w_idx, h_idx, np.ones([h, w, 1])], axis=2)  # h * w * 3
    grid = np.repeat(grid[np.newaxis, :], n, axis=0)  # n * h * w *3

    ret = np.zeros([n, h * w, 2])
    theta = theta.transpose([0, 2, 1])
    for i in range(len(theta)):
        ret[i] = np.dot(grid[i].reshape([h * w, 3]), theta[i])

    return ret.reshape([n, h, w, 2]).astype("float64")


def getGridPointValue(data, x, y):
    data_shape = data.shape
    N = data_shape[0]
    C = data_shape[1]
    in_H = data_shape[2]
    in_W = data_shape[3]
    out_H = x.shape[1]
    out_W = x.shape[2]

    # out = np.zeros(data_shape, dtype='float64')
    out = np.zeros([N, C, out_H, out_W], dtype="float64")
    for i in range(N):
        for j in range(out_H):
            for k in range(out_W):
                if (
                    y[i, j, k] < 0
                    or y[i, j, k] > in_H - 1
                    or x[i, j, k] < 0
                    or x[i, j, k] > in_W - 1
                ):
                    out[i, :, j, k] = 0
                else:
                    out[i, :, j, k] = data[i, :, y[i, j, k], x[i, j, k]]

    return out


def AffineGrid3D(theta, grid_shape):
    n = grid_shape[0]
    d = grid_shape[1]
    h = grid_shape[2]
    w = grid_shape[3]
    d_idx = np.repeat(
        np.repeat(np.linspace(-1, 1, d)[:, np.newaxis, np.newaxis], h, axis=1),
        w,
        axis=2,
    )[:, :, :, np.newaxis]
    h_idx = np.repeat(
        np.repeat(np.linspace(-1, 1, h)[np.newaxis, :, np.newaxis], w, axis=2),
        d,
        axis=0,
    )[:, :, :, np.newaxis]
    w_idx = np.repeat(
        np.repeat(np.linspace(-1, 1, w)[np.newaxis, np.newaxis, :], h, axis=1),
        d,
        axis=0,
    )[:, :, :, np.newaxis]
    grid = np.concatenate(
        [w_idx, h_idx, d_idx, np.ones([d, h, w, 1])], axis=3
    )  # d * h * w * 4
    grid = np.repeat(grid[np.newaxis, :], n, axis=0)  # n * d * h * w *4
    ret = np.zeros([n, d * h * w, 3])
    theta = theta.transpose([0, 2, 1])
    for i in range(len(theta)):
        ret[i] = np.dot(grid[i].reshape([d * h * w, 4]), theta[i])

    return ret.reshape([n, d, h, w, 3]).astype("float64")


def getGridPointValue3D(data, x, y, z):
    data_shape = data.shape
    N = data_shape[0]
    C = data_shape[1]
    in_D = data_shape[2]
    in_H = data_shape[3]
    in_W = data_shape[4]
    out_D = x.shape[1]
    out_H = x.shape[2]
    out_W = x.shape[3]

    out = np.zeros([N, C, out_D, out_H, out_W], dtype="float64")
    for i in range(N):
        for j in range(out_D):
            for k in range(out_H):
                for l in range(out_W):
                    if (
                        y[i, j, k, l] < 0
                        or y[i, j, k, l] > in_H - 1
                        or x[i, j, k, l] < 0
                        or x[i, j, k, l] > in_W - 1
                        or z[i, j, k, l] < 0
                        or z[i, j, k, l] > in_D - 1
                    ):
                        out[i, :, j, k, l] = 0
                    else:
                        out[i, :, j, k, l] = data[
                            i, :, z[i, j, k, l], y[i, j, k, l], x[i, j, k, l]
                        ]

    return out


def clip(x, min_n, max_n):
    return np.maximum(np.minimum(x, max_n), min_n)


def unnormalizeAndClip(grid_slice, max_val, align_corners, padding_mode):
    if align_corners:
        grid_slice = 0.5 * ((grid_slice.astype("float64") + 1.0) * max_val)
    else:
        grid_slice = 0.5 * ((grid_slice.astype("float64") + 1.0) * (max_val + 1)) - 0.5

    if padding_mode == "border":
        grid_slice = clip(grid_slice, 0, max_val)
    elif padding_mode == "reflection":
        double_range = 2 * max_val if align_corners else (max_val + 1) * 2
        grid_abs = np.abs(grid_slice) if align_corners else np.abs(grid_slice + 0.5)
        extra = grid_abs - np.floor(grid_abs / double_range) * double_range
        grid_slice = np.minimum(extra, double_range - extra)
        grid_slice = grid_slice if align_corners else clip(grid_slice - 0.5, 0, max_val)
    return grid_slice


def GridSampler(data, grid, align_corners=True, mode="bilinear", padding_mode="zeros"):
    dims = data.shape
    N = dims[0]
    in_C = dims[1]
    in_H = dims[2]
    in_W = dims[3]

    out_H = grid.shape[1]
    out_W = grid.shape[2]

    x = grid[:, :, :, 0]
    y = grid[:, :, :, 1]
    y_max = in_H - 1
    x_max = in_W - 1

    x = unnormalizeAndClip(x, x_max, align_corners, padding_mode)
    y = unnormalizeAndClip(y, y_max, align_corners, padding_mode)

    if mode == "bilinear":
        x0 = np.floor(x).astype("int32")
        x1 = x0 + 1
        y0 = np.floor(y).astype("int32")
        y1 = y0 + 1

        wa = np.tile(
            ((x1 - x) * (y1 - y)).reshape((N, 1, out_H, out_W)), (1, in_C, 1, 1)
        )
        wb = np.tile(
            ((x1 - x) * (y - y0)).reshape((N, 1, out_H, out_W)), (1, in_C, 1, 1)
        )
        wc = np.tile(
            ((x - x0) * (y1 - y)).reshape((N, 1, out_H, out_W)), (1, in_C, 1, 1)
        )
        wd = np.tile(
            ((x - x0) * (y - y0)).reshape((N, 1, out_H, out_W)), (1, in_C, 1, 1)
        )

        va = getGridPointValue(data, x0, y0)
        vb = getGridPointValue(data, x0, y1)
        vc = getGridPointValue(data, x1, y0)
        vd = getGridPointValue(data, x1, y1)

        out = (wa * va + wb * vb + wc * vc + wd * vd).astype("float64")
    elif mode == "nearest":
        x = np.round(x).astype("int32")
        y = np.round(y).astype("int32")
        out = getGridPointValue(data, x, y)
    return out


def GridSampler3D(
    data, grid, align_corners=True, mode="bilinear", padding_mode="zeros"
):
    dims = data.shape
    N = dims[0]
    in_C = dims[1]
    in_D = dims[2]
    in_H = dims[3]
    in_W = dims[4]

    out_D = grid.shape[1]
    out_H = grid.shape[2]
    out_W = grid.shape[3]

    x = grid[:, :, :, :, 0]
    y = grid[:, :, :, :, 1]
    z = grid[:, :, :, :, 2]

    z_max = in_D - 1
    y_max = in_H - 1
    x_max = in_W - 1

    x = unnormalizeAndClip(x, x_max, align_corners, padding_mode)
    y = unnormalizeAndClip(y, y_max, align_corners, padding_mode)
    z = unnormalizeAndClip(z, z_max, align_corners, padding_mode)

    if mode == "bilinear":
        x0 = np.floor(x).astype("int32")
        x1 = x0 + 1
        y0 = np.floor(y).astype("int32")
        y1 = y0 + 1
        z0 = np.floor(z).astype("int32")
        z1 = z0 + 1

        w_tnw = np.tile(
            ((x1 - x) * (y1 - y) * (z1 - z)).reshape((N, 1, out_D, out_H, out_W)),
            (1, in_C, 1, 1, 1),
        )
        w_tne = np.tile(
            ((x - x0) * (y1 - y) * (z1 - z)).reshape((N, 1, out_D, out_H, out_W)),
            (1, in_C, 1, 1, 1),
        )
        w_tsw = np.tile(
            ((x1 - x) * (y - y0) * (z1 - z)).reshape((N, 1, out_D, out_H, out_W)),
            (1, in_C, 1, 1, 1),
        )
        w_tse = np.tile(
            ((x - x0) * (y - y0) * (z1 - z)).reshape((N, 1, out_D, out_H, out_W)),
            (1, in_C, 1, 1, 1),
        )
        w_bnw = np.tile(
            ((x1 - x) * (y1 - y) * (z - z0)).reshape((N, 1, out_D, out_H, out_W)),
            (1, in_C, 1, 1, 1),
        )
        w_bne = np.tile(
            ((x - x0) * (y1 - y) * (z - z0)).reshape((N, 1, out_D, out_H, out_W)),
            (1, in_C, 1, 1, 1),
        )
        w_bsw = np.tile(
            ((x1 - x) * (y - y0) * (z - z0)).reshape((N, 1, out_D, out_H, out_W)),
            (1, in_C, 1, 1, 1),
        )
        w_bse = np.tile(
            ((x - x0) * (y - y0) * (z - z0)).reshape((N, 1, out_D, out_H, out_W)),
            (1, in_C, 1, 1, 1),
        )

        v_tnw = getGridPointValue3D(data, x0, y0, z0)
        v_tne = getGridPointValue3D(data, x1, y0, z0)
        v_tsw = getGridPointValue3D(data, x0, y1, z0)
        v_tse = getGridPointValue3D(data, x1, y1, z0)
        v_bnw = getGridPointValue3D(data, x0, y0, z1)
        v_bne = getGridPointValue3D(data, x1, y0, z1)
        v_bsw = getGridPointValue3D(data, x0, y1, z1)
        v_bse = getGridPointValue3D(data, x1, y1, z1)

        out = (
            w_tnw * v_tnw
            + w_tne * v_tne
            + w_tsw * v_tsw
            + w_tse * v_tse
            + w_bnw * v_bnw
            + w_bne * v_bne
            + w_bsw * v_bsw
            + w_bse * v_bse
        ).astype("float64")

    elif mode == "nearest":
        x = np.round(x).astype("int32")
        y = np.round(y).astype("int32")
        z = np.round(z).astype("int32")
        out = getGridPointValue3D(data, x, y, z)
    return out


class TestGridSamplerDygraph(unittest.TestCase):
    def init_place(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("sdaa", 0)

    def setUp(self):
        np.random.seed(SEED)
        self.init_place()
        self.init_dtype()

        self.align_corners = False
        self.padding_mode = "zeros"
        self.mode = "bilinear"

        self.initTestCase()

        x = np.random.uniform(-10, 10, self.x_shape).astype(self.dtype)

        theta = np.zeros(self.theta_shape).astype(self.dtype)

        if len(self.grid_shape) == 4:
            for i in range(self.theta_shape[0]):
                for j in range(2):
                    for k in range(3):
                        theta[i, j, k] = np.random.rand(1)[0]
            grid = AffineGrid(theta, self.grid_shape).astype(self.dtype)

            self.inputs = {"X": x, "Grid": grid}
            self.attrs = {
                "align_corners": self.align_corners,
                "padding_mode": self.padding_mode,
                "mode": self.mode,
            }
        else:
            for i in range(self.theta_shape[0]):
                for j in range(3):
                    for k in range(4):
                        theta[i, j, k] = np.random.rand(1)[0]
            grid = AffineGrid3D(theta, self.grid_shape)
            self.inputs = {"X": x, "Grid": grid}
            self.attrs = {
                "align_corners": self.align_corners,
                "padding_mode": self.padding_mode,
                "mode": self.mode,
            }

    def initTestCase(self):
        self.x_shape = (32, 32, 96, 96)
        self.grid_shape = (32, 500, 4, 2)
        self.theta_shape = (2, 2, 3)
        self.align_corners = False
        self.padding_mode = "zeros"
        self.mode = "bilinear"

    def init_dtype(self):
        self.dtype = "float32"

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-4, rtol=1e-5)

    def check_output_with_place(self, place, atol=1e-5, rtol=1e-7):

        with dygraph.guard(place=paddle.CPUPlace()):
            self.cpu_x = paddle.to_tensor(self.inputs["X"], stop_gradient=False)
            self.cpu_grid = paddle.to_tensor(self.inputs["Grid"], stop_gradient=False)

            self.cpu_out = paddle.nn.functional.grid_sample(
                self.cpu_x,
                self.cpu_grid,
                mode=self.attrs["mode"],
                padding_mode=self.attrs["padding_mode"],
                align_corners=self.attrs["align_corners"],
            )

        with dygraph.guard(place=place):
            self.device_x = paddle.to_tensor(self.inputs["X"], stop_gradient=False)
            self.device_grid = paddle.to_tensor(
                self.inputs["Grid"], stop_gradient=False
            )

            self.device_out = paddle.nn.functional.grid_sample(
                self.device_x,
                self.device_grid,
                mode=self.attrs["mode"],
                padding_mode=self.attrs["padding_mode"],
                align_corners=self.attrs["align_corners"],
            )

        np.testing.assert_allclose(
            self.cpu_out.numpy(), self.device_out.numpy(), atol=atol, rtol=rtol
        )

        dout = np.random.random(size=self.cpu_out.shape).astype(self.dtype)
        with dygraph.guard(place=paddle.CPUPlace()):
            self.cpu_out.backward(paddle.to_tensor(dout))
            self.cpu_x_grad = self.cpu_x.grad
            self.cpu_grid_grad = self.cpu_grid.grad

        with dygraph.guard(place=place):
            self.device_out.backward(paddle.to_tensor(dout, place=place))
            self.device_x_grad = self.device_x.grad
            self.device_grid_grad = self.device_grid.grad

        np.testing.assert_allclose(
            self.cpu_out.numpy(),
            self.device_out.numpy(),
            atol=atol,
            rtol=rtol,
            err_msg="forward output not satisfy",
        )

        np.testing.assert_allclose(
            self.cpu_x_grad.numpy(),
            self.device_x_grad.numpy(),
            atol=atol,
            rtol=rtol,
            err_msg="backward x_grad not satisfy",
        )

        # NOTE(huangzhen): grid gradient of grid_sample backward has max absolute diff 1e-2 since tecodnn 1.15.0
        np.testing.assert_allclose(
            self.cpu_grid_grad.numpy(),
            self.device_grid_grad.numpy(),
            atol=1e-2,
            rtol=1e-5,
            err_msg="backward grid_grad not satisfy",
        )


class TestGridSampleDygraphCase1(TestGridSamplerDygraph):
    def initTestCase(self):
        self.x_shape = (32, 32, 80, 80)
        self.grid_shape = (32, 498, 4, 2)
        self.theta_shape = (2, 2, 3)
        self.align_corners = False
        self.padding_mode = "zeros"
        self.mode = "bilinear"


class TestGridSampleDygraphCase2(TestGridSamplerDygraph):
    def initTestCase(self):
        self.x_shape = (32, 32, 40, 40)
        self.grid_shape = (32, 498, 4, 2)
        self.theta_shape = (2, 2, 3)
        self.align_corners = False
        self.padding_mode = "zeros"
        self.mode = "bilinear"


class TestGridSampleDygraphCase3(TestGridSamplerDygraph):
    def initTestCase(self):
        self.x_shape = (32, 32, 20, 20)
        self.grid_shape = (32, 498, 4, 2)
        self.theta_shape = (2, 2, 3)
        self.align_corners = False
        self.padding_mode = "zeros"
        self.mode = "bilinear"


class TestGridSampleDygraphCase4(TestGridSamplerDygraph):
    def initTestCase(self):
        self.x_shape = (32, 32, 80, 80)
        self.grid_shape = (32, 300, 4, 2)
        self.theta_shape = (2, 2, 3)
        self.align_corners = False
        self.padding_mode = "zeros"
        self.mode = "bilinear"


class TestGridSampleDygraphCase5(TestGridSamplerDygraph):
    def initTestCase(self):
        self.x_shape = (32, 32, 40, 40)
        self.grid_shape = (32, 300, 4, 2)
        self.theta_shape = (2, 2, 3)
        self.align_corners = False
        self.padding_mode = "zeros"
        self.mode = "bilinear"


class TestGridSampleDygraphCase6(TestGridSamplerDygraph):
    def initTestCase(self):
        self.x_shape = (32, 32, 20, 20)
        self.grid_shape = (32, 300, 4, 2)
        self.theta_shape = (2, 2, 3)
        self.align_corners = False
        self.padding_mode = "zeros"
        self.mode = "bilinear"


if __name__ == "__main__":
    unittest.main()
