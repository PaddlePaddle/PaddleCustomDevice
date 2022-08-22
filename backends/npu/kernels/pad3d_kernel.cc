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
void Pad3dKernel(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 const phi::IntArray& paddings_array,
                 const std::string& mode,
                 float pad_value,
                 const std::string& data_format,
                 phi::DenseTensor* out) {
  auto in_dims = x.dims();
  auto pads = paddings_array.GetData();

  PADDLE_ENFORCE_LT(abs(pad_value),
                    1e-5,
                    phi::errors::Unimplemented(
                        "Ascend npu only support constant_values=0 right now,"
                        "but received constant_value is %f .",
                        pad_value));

  PADDLE_ENFORCE_EQ(mode,
                    "constant",
                    phi::errors::Unimplemented(
                        "Ascend npu only support mode=constant right now,"
                        "but received mode is %s .",
                        mode));

  std::vector<int> paddings(
      {0, 0, 0, 0, pads[4], pads[5], pads[2], pads[3], pads[0], pads[1]});
  if (data_format == "NCDHW") {
    out->Resize({in_dims[0],
                 in_dims[1],
                 in_dims[2] + pads[4] + pads[5],
                 in_dims[3] + pads[2] + pads[3],
                 in_dims[4] + pads[0] + pads[1]});
  } else {
    out->Resize({in_dims[0],
                 in_dims[1] + pads[4] + pads[5],
                 in_dims[2] + pads[2] + pads[3],
                 in_dims[3] + pads[0] + pads[1],
                 in_dims[4]});
    paddings = {
        0, 0, pads[4], pads[5], pads[2], pads[3], pads[0], pads[1], 0, 0};
  }
  dev_ctx.template Alloc<T>(out);

  NpuOpRunner runner;
  runner.SetType("PadV3")
      .AddInput(x)
      .AddInput(dev_ctx, std::move(paddings))
      .AddInput(dev_ctx,
                std::vector<int>({0}))  // npu only support constant_value=0 now
      .AddOutput(*out)
      .AddAttr("mode", mode);

  auto stream = dev_ctx.stream();
  runner.Run(stream);
}

template <typename T, typename Context>
void Pad3dGradKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const phi::DenseTensor& out_grad,
                     const phi::IntArray& paddings_array,
                     const std::string& mode,
                     float pad_value,
                     const std::string& data_format,
                     phi::DenseTensor* x_grad) {
  auto x_grad_dims = x_grad->dims();
  dev_ctx.template Alloc<T>(x_grad);

  auto pads = paddings_array.GetData();

  const int pad_left = pads[0];
  const int pad_top = pads[2];
  const int pad_front = pads[4];

  auto stream = dev_ctx.stream();

  std::vector<int64_t> size({x_grad_dims[0],
                             x_grad_dims[1],
                             x_grad_dims[2],
                             x_grad_dims[3],
                             x_grad_dims[4]});
  if (mode == "constant") {  // this method can be only used for constant mode
    std::vector<int> offsets({0, 0, pad_front, pad_top, pad_left});
    if (data_format == "NDHWC") {
      offsets = {0, pad_front, pad_top, pad_left, 0};
    }
    const auto& runner = NpuOpRunner("SliceD",
                                     {out_grad},
                                     {*x_grad},
                                     {{"offsets", offsets}, {"size", size}});
    runner.Run(stream);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(pad3d,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::Pad3dKernel,
                          phi::dtype::float16,
                          float,
                          int) {}

PD_REGISTER_PLUGIN_KERNEL(pad3d_grad,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::Pad3dGradKernel,
                          phi::dtype::float16,
                          float) {}
