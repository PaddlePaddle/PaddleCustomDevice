// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
void GridSampleKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      const phi::DenseTensor& grid,
                      const std::string& mode,
                      const std::string& padding_mode,
                      bool align_corners,
                      phi::DenseTensor* out) {
  // GridSample2D
  if (x.dims().size() == 4) {
    const int n = grid.dims()[0];
    const int out_h = grid.dims()[1];
    const int out_w = grid.dims()[2];
    const int c = x.dims()[1];
    const int in_h = x.dims()[2];
    const int in_w = x.dims()[3];
    out->Resize(phi::make_ddim({n, c, out_h, out_w}));
    dev_ctx.template Alloc<T>(out);
    NpuOpRunner grid_sample_runner;
    grid_sample_runner.SetType("GridSampler2D")
        .AddInput(x)
        .AddInput(grid)
        .AddAttr("interpolation_mode", mode)
        .AddAttr("padding_mode", padding_mode)
        .AddAttr("align_corners", align_corners)
        .AddOutput(*out)
        .Run(dev_ctx.stream());
  } else if (x.dims().size() == 5) {
    // GridSample3D
    const int n = grid.dims()[0];
    const int out_d = grid.dims()[1];
    const int out_h = grid.dims()[2];
    const int out_w = grid.dims()[3];
    const int c = x.dims()[1];
    const int in_d = x.dims()[2];
    const int in_h = x.dims()[3];
    const int in_w = x.dims()[4];

    out->Resize(phi::make_ddim({n, c, out_d, out_h, out_w}));
    dev_ctx.template Alloc<T>(out);
    NpuOpRunner grid_sample_runner;
    grid_sample_runner.SetType("GridSampler3D")
        .AddInput(x)
        .AddInput(grid)
        .AddAttr("interpolation_mode", mode)
        .AddAttr("padding_mode", padding_mode)
        .AddAttr("align_corners", align_corners)
        .AddOutput(*out)
        .Run(dev_ctx.stream());
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "the input x must be 4D/5D tensor of grid_sample, but got [%d]D tensor",
        x.dims().size()));
  }
}

template <typename T, typename Context>
void GridSampleGradKernel(const Context& dev_ctx,
                          const phi::DenseTensor& x,
                          const phi::DenseTensor& grid,
                          const phi::DenseTensor& out_grad,
                          const std::string& mode,
                          const std::string& padding_mode,
                          bool align_corners,
                          phi::DenseTensor* x_grad,
                          phi::DenseTensor* grid_grad) {
  if (x.dims().size() == 4) {
    const int n = grid.dims()[0];
    const int out_h = grid.dims()[1];
    const int out_w = grid.dims()[2];
    const int c = x.dims()[1];
    const int in_h = x.dims()[2];
    const int in_w = x.dims()[3];

    x_grad->Resize({n, c, in_h, in_w});
    dev_ctx.template Alloc<T>(x_grad);

    if (grid_grad != nullptr) {
      grid_grad->Resize({n, out_h, out_w, 2});
      dev_ctx.template Alloc<T>(grid_grad);

      NpuOpRunner grid_sample_runner;
      grid_sample_runner.SetType("GridSampler2DGrad")
          .AddInput(out_grad)
          .AddInput(x)
          .AddInput(grid)
          .AddAttr("interpolation_mode", mode)
          .AddAttr("padding_mode", padding_mode)
          .AddAttr("align_corners", align_corners)
          .AddOutput(*x_grad)
          .AddOutput(*grid_grad)
          .Run(dev_ctx.stream());
    } else {
      phi::DenseTensor tmp_grid_grad;
      tmp_grid_grad.Resize({n, out_h, out_w, 2});
      dev_ctx.template Alloc<T>(&tmp_grid_grad);
      NpuOpRunner grid_sample_runner;
      grid_sample_runner.SetType("GridSampler2DGrad")
          .AddInput(out_grad)
          .AddInput(x)
          .AddInput(grid)
          .AddAttr("interpolation_mode", mode)
          .AddAttr("padding_mode", padding_mode)
          .AddAttr("align_corners", align_corners)
          .AddOutput(*x_grad)
          .AddOutput(tmp_grid_grad)
          .Run(dev_ctx.stream());
    }
  } else if (x.dims().size() == 5) {
    const int n = grid.dims()[0];
    const int out_d = grid.dims()[1];
    const int out_h = grid.dims()[2];
    const int out_w = grid.dims()[3];
    const int c = x.dims()[1];
    const int in_d = x.dims()[2];
    const int in_h = x.dims()[3];
    const int in_w = x.dims()[4];

    x_grad->Resize({n, c, in_d, in_h, in_w});
    dev_ctx.template Alloc<T>(x_grad);

    if (grid_grad != nullptr) {
      grid_grad->Resize({n, out_d, out_h, out_w, 3});
      dev_ctx.template Alloc<T>(grid_grad);

      NpuOpRunner grid_sample_runner;
      grid_sample_runner.SetType("GridSampler3DGrad")
          .AddInput(out_grad)
          .AddInput(x)
          .AddInput(grid)
          .AddAttr("interpolation_mode", mode)
          .AddAttr("padding_mode", padding_mode)
          .AddAttr("align_corners", align_corners)
          .AddOutput(*x_grad)
          .AddOutput(*grid_grad)
          .Run(dev_ctx.stream());
    } else {
      phi::DenseTensor tmp_grid_grad;
      tmp_grid_grad.Resize({n, out_h, out_w, 3});
      dev_ctx.template Alloc<T>(&tmp_grid_grad);
      NpuOpRunner grid_sample_runner;
      grid_sample_runner.SetType("GridSampler3DGrad")
          .AddInput(out_grad)
          .AddInput(x)
          .AddInput(grid)
          .AddAttr("interpolation_mode", mode)
          .AddAttr("padding_mode", padding_mode)
          .AddAttr("align_corners", align_corners)
          .AddOutput(*x_grad)
          .AddOutput(tmp_grid_grad)
          .Run(dev_ctx.stream());
    }
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "the input x must be 4D/5D tensor of grid_sample, but got [%d]D tensor",
        x.dims().size()));
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(grid_sample,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::GridSampleKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(grid_sample_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::GridSampleGradKernel,
                          float,
                          phi::dtype::float16) {}
