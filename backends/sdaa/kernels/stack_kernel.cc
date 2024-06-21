// BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
// reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
// Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software
// without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
// WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
// OF SUCH DAMAGE.

#include "kernels/funcs/sdaa_baseop.h"
#include "paddle/phi/extension.h"

namespace custom_kernel {

template <typename T, typename Context>
void StackKernel(const Context& dev_ctx,
                 const std::vector<const phi::DenseTensor*>& x,
                 int axis,
                 phi::DenseTensor* out) {
  VLOG(4) << "CALL SDAA StackKernel";
  dev_ctx.template Alloc<T>(out);

  if (axis < 0) axis += (x[0]->dims().size() + 1);
  int num = static_cast<int>(x.size());

  PADDLE_ENFORCE_GT(
      num, 0, phi::errors::InvalidArgument("number of input Tensor <= 0"));

  std::vector<phi::DenseTensor*> x_;
  std::vector<int> input_dims = phi::vectorize<int>(x[0]->dims());
  input_dims.insert(input_dims.begin() + axis, 1);

  for (int i = 0; i < num; i++) {
    x_.push_back(const_cast<phi::DenseTensor*>(x[i]));
    x_[i]->Resize(phi::make_ddim(input_dims));
  }

  std::vector<const phi::DenseTensor*> x_temp;
  for (int i = 0; i < num; i++) {
    x_temp.push_back(x_[i]);
  }

  sdaa_ops::doConcatTensor(dev_ctx, x_temp, axis, out);
}

template <typename T, typename Context>
void StackGradKernel(const Context& dev_ctx,
                     const phi::DenseTensor& dy,
                     int axis,
                     std::vector<phi::DenseTensor*> dx) {
  VLOG(4) << "CALL SDAA StackGradKernel";

  // get dx dims
  std::vector<int> input_dims;
  for (int i = 0; i < dx.size(); ++i) {
    if (dx[i]) {
      input_dims = phi::vectorize<int>(dx[i]->dims());
      break;
    }
  }
  PADDLE_ENFORCE_EQ(input_dims.empty(),
                    false,
                    phi::errors::InvalidArgument(
                        "input tensors do not have gradients to compute"));

  PADDLE_ENFORCE_EQ(input_dims.size() + 1,
                    dy.dims().size(),
                    phi::errors::InvalidArgument(
                        "input_grad's dims shoule be one less than dy's dims",
                        "but got input_grad's dims %d ",
                        "dy's dims %d",
                        input_dims.size(),
                        dy.dims().size()));
  if (axis < 0) axis += (input_dims.size() + 1);
  int num = static_cast<int>(dx.size());
  PADDLE_ENFORCE_GT(
      num, 0, phi::errors::InvalidArgument("number of input Tensor <= 0"));

  std::vector<int> input_dims_origin(input_dims);
  input_dims.insert(input_dims.begin() + axis, 1);

  std::vector<phi::DenseTensor> tmp_outputs_vec;
  tmp_outputs_vec.resize(dx.size());
  std::vector<phi::DenseTensor*> dx_;

  const phi::DenseTensorMeta meta_data(dy.dtype(), phi::make_ddim(input_dims));
  for (int i = 0; i < dx.size(); ++i) {
    if (dx[i]) {
      dev_ctx.template Alloc<T>(dx[i]);
      dx_.push_back(dx[i]);
      dx_[i]->Resize(phi::make_ddim(input_dims));
    } else {
      phi::DenseTensor tmp_tensor;
      tmp_tensor.set_meta(meta_data);
      dev_ctx.template Alloc<T>(&tmp_tensor);
      tmp_outputs_vec[i] = std::move(tmp_tensor);
      dx_.push_back(&(tmp_outputs_vec[i]));
    }
  }

  sdaa_ops::doSplitTensor(dev_ctx, dy, axis, dx_);

  for (int i = 0; i < num; i++) {
    if (dx[i]) {
      dx_[i]->Resize(phi::make_ddim(input_dims_origin));
    }
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(stack,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::StackKernel,
                          float,
                          phi::dtype::float16,
                          bool,
                          int,
                          uint8_t,
                          int64_t) {}

PD_REGISTER_PLUGIN_KERNEL(stack_grad,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::StackGradKernel,
                          float,
                          phi::dtype::float16,
                          bool,
                          int,
                          uint8_t,
                          int64_t) {}
