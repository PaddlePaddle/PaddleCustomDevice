// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <iostream>

#include "kernels/funcs/sdaa_baseop.h"
#include "paddle/phi/extension.h"
namespace custom_kernel {

template <typename T, typename Context>
void NPUIdentityKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const int format,
                       phi::DenseTensor* out) {
  PADDLE_ENFORCE_EQ(
      format,
      -1,
      phi::errors::InvalidArgument("tecodnn only support format=-1"));
  VLOG(4) << "NPUIdentityKernel -1";
  PADDLE_ENFORCE_EQ(x.storage_properties_initialized(),
                    true,
                    phi::errors::InvalidArgument(
                        "sdaa identity kernel only support tensor with "
                        "storage_properties when format == -1"));
  auto storages = x.storage_properties<SDAAStorageProperties>();
  phi::DDim x_dims = storages.storage_dims;  // CHWN
  phi::DenseTensorMeta out_meta;
  out_meta = {x.dtype(), {x_dims[3], x_dims[0], x_dims[1], x_dims[2]}};
  out->set_meta(out_meta);
  dev_ctx.template Alloc<T>(out, x.numel() * sizeof(T));
  sdaa_ops::doTransformTensor(dev_ctx, x, Convert_TF::CHWN2NCHW, out);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(npu_identity,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::NPUIdentityKernel,
                          float,
                          phi::dtype::float16) {}
