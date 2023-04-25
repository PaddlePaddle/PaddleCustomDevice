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

#include "kernels/funcs/mlu_funcs.h"

namespace custom_kernel {

template <typename T, typename Context>
void CastKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                phi::DataType dtype,
                phi::DenseTensor* out) {
  if (x.dtype() == dtype) {
    dev_ctx.template Alloc<T>(out);
    TensorCopy(dev_ctx, x, false, out);
    return;
  }

  if (!MLUSupportsCast(x.dtype(), dtype)) {
    // fallback to cpu
    VLOG(3) << "MLU not support cast " << x.dtype() << " to " << dtype
            << ". Fallback to cpu.";
    Tensor x_cpu;
    Tensor out_cpu;
    TensorCopy(dev_ctx, x, false, &x_cpu, phi::CPUPlace());
    dev_ctx.Wait();
    phi::CPUContext dev_ctx_cpu;
    dev_ctx_cpu.SetAllocator(&(dev_ctx.GetHostAllocator()));
    dev_ctx_cpu.SetHostAllocator(&(dev_ctx.GetHostAllocator()));
    out_cpu.Resize(out->dims());
    phi::CastKernel<T, phi::CPUContext>(dev_ctx_cpu, x_cpu, dtype, &out_cpu);
    TensorCopy(dev_ctx, out_cpu, true, out);

    return;
  }

  dev_ctx.Alloc(out, dtype);

  MLUCnnlTensorDesc x_desc(x);
  MLUCnnlTensorDesc out_desc(*out);
  cnnlCastDataType_t cast_type = GetCastDataType(x.dtype(), dtype);
  MLUCnnl::Cast(dev_ctx,
                cast_type,
                x_desc.get(),
                GetBasePtr(&x),
                out_desc.get(),
                GetBasePtr(out));
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(cast,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::CastKernel,
                          phi::dtype::float16,
                          float,
                          double,
                          uint8_t,
                          int8_t,
                          int16_t,
                          int32_t,
                          int64_t,
                          bool) {
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
}
