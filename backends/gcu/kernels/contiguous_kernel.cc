// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "common/gcu_op_runner.h"
#include "kernels/funcs/gcu_kernel_funcs.h"

namespace custom_kernel {

template <typename T, typename Context>
void ContiguousKernel(const Context& dev_ctx,
                      const phi::DenseTensor& input,
                      phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("contiguous");
  phi::DenseTensorMeta meta = input.meta();
  meta.strides = meta.calc_strides(meta.dims);
  meta.offset = 0;
  out->set_meta(meta);
  dev_ctx.template Alloc<T>(out);

  if (LaunchAOTKernel()) {
    auto x_tensor = CreateTopsatenTensor(input);
    auto out_tensor = CreateTopsatenTensor(*out);
    topsatenMemoryFormat_t topsaten_format = TOPSATEN_MEMORY_CONTIGUOUS;
    auto stream = static_cast<topsStream_t>(dev_ctx.stream());
    VLOG(3) << "ContiguousKernel, use topsatenContiguous, stream: " << stream;
    ATEN_OP_CALL_MAYBE_SYNC(topsaten::topsatenContiguous(
                                out_tensor, x_tensor, topsaten_format, stream),
                            dev_ctx);

    // const T* input_data = input.data<T>();
    // auto* output_data = dev_ctx.template Alloc<T>(out);
    // int rank = input.dims().size();
    // auto dims = input.dims();
    // auto input_stride = input.strides();
    // auto numel = input.numel();

    // for (int64_t i = 0; i < numel; i++) {
    //   int64_t input_offset = 0;
    //   int64_t index_tmp = i;
    //   for (int dim = rank - 1; dim >= 0; --dim) {
    //     int64_t mod = index_tmp % dims[dim];
    //     index_tmp = index_tmp / dims[dim];
    //     input_offset += mod * input_stride[dim];
    //   }
    //   C_Device_st device{input.place().GetDeviceId()};
    //   C_Stream stream = static_cast<C_Stream>(dev_ctx.stream());
    //   auto* dst_ptr = &output_data[i];
    //   auto* src_ptr = &input_data[input_offset];
    //   AsyncMemCpyD2D(&device, stream, dst_ptr, src_ptr,
    //                  phi::SizeOf(input.dtype()));
    // }
  } else {  // kernel impl base on JIT
    THROW_JIT_UNIMPLEMENTED();
  }
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(contiguous,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::ContiguousKernel,
                          bool,
                          uint8_t,
                          int8_t,
                          int16_t,
                          int32_t,
                          int64_t,
                          float,
                          double,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          phi::dtype::complex<float>,
                          phi::dtype::complex<double>) {}
