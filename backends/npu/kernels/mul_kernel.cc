// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"

namespace custom_kernel {

template <typename T, typename Context>
void MulKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               const phi::DenseTensor& y,
               int x_num_col_dims,
               int y_num_col_dims,
               phi::DenseTensor* out) {
  auto stream = dev_ctx.stream();

  if (x_num_col_dims == 1 && y_num_col_dims == 1) {
    if (x.dims().size() == 2 && y.dims().size() == 2) {
      dev_ctx.template Alloc<T>(out);
      const auto& runner =
          NpuOpRunner("MatMul",
                      {x, y},
                      {*out},
                      {{"transpose_x1", false}, {"transpose_x2", false}});

      runner.Run(stream);
    } else if (x.dims().size() >= 3 && y.dims().size() == 2) {
      // reshape
      phi::DenseTensor tmp_x(x);
      int64_t sec_dim = x.dims()[1];
      for (auto i = 2; i < x.dims().size(); i++) {
        sec_dim *= x.dims()[i];
      }
      int64_t first_dim = x.dims()[0];
      tmp_x.Resize(phi::make_ddim({first_dim, sec_dim}));
      dev_ctx.template Alloc<T>(out);
      // matmul
      const auto& runner =
          NpuOpRunner("MatMul",
                      {tmp_x, y},
                      {*out},
                      {{"transpose_x1", false}, {"transpose_x2", false}});
      runner.Run(stream);
    } else {
      PADDLE_THROW(phi::errors::InvalidArgument("npu error: not support dims"));
    }
    // to do other
  } else if (x.dims().size() == 3 && y.dims().size() == 2) {
    // for example: x.shape=[2, 3, 4] y.shape=[4, 5], expect [2, 3, 5]
    PADDLE_ENFORCE_EQ(x_num_col_dims,
                      2,
                      phi::errors::InvalidArgument(
                          "now only support x_num_col_dims == 2: but got %d",
                          x_num_col_dims));
    if (x.dtype() == phi::DenseTensorMeta::DataType::FLOAT16 &&
        y.dtype() == phi::DenseTensorMeta::DataType::FLOAT16) {
      // NOTE: When the dim of the input and output shapes is inconsistent,
      // (Boradcast) BatchMatMul NPU OP only support FP16.
      dev_ctx.template Alloc<T>(out);
      const auto& runner = NpuOpRunner("BatchMatMul",
                                       {x, y},
                                       {*out},
                                       {{"adj_x1", false}, {"adj_x2", false}});

      auto stream = dev_ctx.stream();
      runner.Run(stream);
    } else {
      // flatten => x.shape=[6, 4]
      phi::DenseTensor tmp_x(x);
      int64_t first_dim = x.dims()[0] * x.dims()[1];
      int64_t sec_dim = x.dims()[2];
      tmp_x.Resize(phi::make_ddim({first_dim, sec_dim}));

      // matmul [6,4] , [4, 5] => [6, 5]
      dev_ctx.template Alloc<T>(out);

      phi::DenseTensor tmp_out(*out);
      phi::DenseTensorMeta tmp_out_meta = {
          x.dtype(), phi::make_ddim({first_dim, y.dims()[1]})};
      tmp_out.set_meta(tmp_out_meta);

      const auto& runner_matmul =
          NpuOpRunner("MatMul",
                      {tmp_x, y},
                      {tmp_out},
                      {{"transpose_x1", false}, {"transpose_x2", false}});
      runner_matmul.Run(stream);
    }
  }
}

template <typename T, typename Context>
void MulGradKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& y,
                   const phi::DenseTensor& dout,
                   int x_num_col_dims,
                   int y_num_col_dims,
                   phi::DenseTensor* dx,
                   phi::DenseTensor* dy) {
  auto stream = dev_ctx.stream();

  if (x_num_col_dims == 1 && y_num_col_dims == 1) {
    if (x.dims().size() == 2 && y.dims().size() == 2) {
      if (dx) {
        dev_ctx.template Alloc<T>(dx);
        const auto& runner_dx =
            NpuOpRunner("MatMul",
                        {dout, y},
                        {*dx},
                        {{"transpose_x1", false}, {"transpose_x2", true}});

        runner_dx.Run(stream);
      }

      if (dy) {
        dev_ctx.template Alloc<T>(dy);
        const auto& runner_dy =
            NpuOpRunner("MatMul",
                        {x, dout},
                        {*dy},
                        {{"transpose_x1", true}, {"transpose_x2", false}});

        runner_dy.Run(stream);
      }
    } else if (x.dims().size() >= 3 && y.dims().size() == 2) {
      // flatten => x.shape=[6, 4]
      // matmul
      if (dx) {
        // matmul [2, 5] * [12, 5] => [2, 12]
        dev_ctx.template Alloc<T>(dx);
        phi::DenseTensor tmp_dx(*dx);
        phi::DenseTensorMeta tmp_dx_meta = {
            x.dtype(), phi::make_ddim({dout.dims()[0], y.dims()[0]})};
        tmp_dx.set_meta(tmp_dx_meta);

        const auto& runner_matmul =
            NpuOpRunner("MatMul",
                        {dout, y},
                        {tmp_dx},
                        {{"transpose_x1", false}, {"transpose_x2", true}});
        runner_matmul.Run(stream);
      }

      if (dy) {
        // flatten
        phi::DenseTensor tmp_x(x);
        int64_t sec_dim = x.dims()[1];
        for (auto i = 2; i < x.dims().size(); i++) {
          sec_dim *= x.dims()[i];
        }
        int64_t first_dim = x.dims()[0];
        tmp_x.Resize(phi::make_ddim({first_dim, sec_dim}));
        dev_ctx.template Alloc<T>(dy);
        const auto& runner_dy =
            NpuOpRunner("MatMul",
                        {tmp_x, dout},
                        {*dy},
                        {{"transpose_x1", true}, {"transpose_x2", false}});

        runner_dy.Run(stream);
      }
    }
  } else if (x.dims().size() == 3 && y.dims().size() == 2) {
    // for example: x.shape=[2, 3, 4] y.shape=[4, 5], expect [2, 3, 5]
    PADDLE_ENFORCE_EQ(x_num_col_dims,
                      2,
                      phi::errors::InvalidArgument(
                          "now only support x_num_col_dims == 2: but got %d",
                          x_num_col_dims));
    // tmp_dout both used by dx and dy
    phi::DenseTensor tmp_dout(dout);
    int64_t dout_first_dim = dout.dims()[0] * dout.dims()[1];
    int64_t dout_sec_dim = dout.dims()[2];
    phi::DenseTensorMeta tmp_dout_meta = {
        x.dtype(), phi::make_ddim({dout_first_dim, dout_sec_dim})};
    tmp_dout.set_meta(tmp_dout_meta);

    if (dx) {
      // tmp_dout * y [2, 3, 5] * [4,5] => [2, 3, 4]
      if (dout.dtype() == phi::DenseTensorMeta::DataType::FLOAT16 &&
          y.dtype() == phi::DenseTensorMeta::DataType::FLOAT16) {
        // NOTE: When the dim of the input and output shapes is inconsistent,
        // (Boradcast) BatchMatMul NPU OP only support FP16.
        dev_ctx.template Alloc<T>(dx);
        const auto& runner = NpuOpRunner("BatchMatMul",
                                         {dout, y},
                                         {*dx},
                                         {{"adj_x1", false}, {"adj_x2", true}});

        auto stream = dev_ctx.stream();
        runner.Run(stream);
      } else {
        dev_ctx.template Alloc<T>(dx);
        phi::DenseTensor tmp_dx(*dx);
        phi::DenseTensorMeta tmp_dx_meta = {
            x.dtype(), phi::make_ddim({dout_first_dim, y.dims()[0]})};
        tmp_dx.set_meta(tmp_dx_meta);

        const auto& runner_matmul =
            NpuOpRunner("MatMul",
                        {tmp_dout, y},
                        {tmp_dx},
                        {{"transpose_x1", false}, {"transpose_x2", true}});
        runner_matmul.Run(stream);
      }
    }
    if (dy) {
      // flatten x.shape [2,3,4] => [6, 4]
      phi::DenseTensor tmp_x(x);
      int64_t first_dim = x.dims()[0] * x.dims()[1];
      int64_t sec_dim = x.dims()[2];
      tmp_x.Resize(phi::make_ddim({first_dim, sec_dim}));
      // mamtul [6,4] [6,5] =>[4,5]
      dev_ctx.template Alloc<T>(dy);
      const auto& runner_dy =
          NpuOpRunner("MatMul",
                      {tmp_x, tmp_dout},
                      {*dy},
                      {{"transpose_x1", true}, {"transpose_x2", false}});
      runner_dy.Run(stream);
    }
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(matmul_with_flatten,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::MulKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(matmul_with_flatten_grad,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::MulGradKernel,
                          float,
                          phi::dtype::float16) {}
