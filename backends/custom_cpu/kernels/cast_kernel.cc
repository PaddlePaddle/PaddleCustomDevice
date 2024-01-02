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

#include "paddle/phi/capi/all.h"
#include "phi_funcs.h"  //NOLINT

namespace custom_kernel {

template <typename T>
void CastKernel(const phi::Context& dev_ctx,
                const phi::DenseTensor& x,
                phi::DataType out_dtype,
                phi::DenseTensor* out) {
  auto x_data = x.data<T>();
  out->Resize(x.dims());
  auto numel = x.numel();
  switch (out_dtype) {
    case phi::DataType::BFLOAT16: {
      auto out_data = dev_ctx.template Alloc<phi::dtype::bfloat16>(out);
      for (auto i = 0; i < numel; ++i) {
        out_data[i] =
            static_cast<phi::dtype::bfloat16>(static_cast<float>(x_data[i]));
      }
      break;
    }
    case phi::DataType::FLOAT16: {
      auto out_data = dev_ctx.template Alloc<phi::dtype::float16>(out);
      for (auto i = 0; i < numel; ++i) {
        out_data[i] =
            static_cast<phi::dtype::float16>(static_cast<float>(x_data[i]));
      }
      break;
    }
    case phi::DataType::FLOAT32: {
      auto out_data = dev_ctx.template Alloc<float>(out);
      for (auto i = 0; i < numel; ++i) {
        out_data[i] = static_cast<float>(static_cast<float>(x_data[i]));
      }
      break;
    }
    case phi::DataType::FLOAT64: {
      auto out_data = dev_ctx.template Alloc<double>(out);
      for (auto i = 0; i < numel; ++i) {
        out_data[i] = static_cast<double>(static_cast<float>(x_data[i]));
      }
      break;
    }
    case phi::DataType::INT8: {
      auto out_data = dev_ctx.template Alloc<int8_t>(out);
      for (auto i = 0; i < numel; ++i) {
        out_data[i] = static_cast<int8_t>(static_cast<float>(x_data[i]));
      }
      break;
    }
    case phi::DataType::INT16: {
      auto out_data = dev_ctx.template Alloc<int16_t>(out);
      for (auto i = 0; i < numel; ++i) {
        out_data[i] = static_cast<int16_t>(static_cast<float>(x_data[i]));
      }
      break;
    }
    case phi::DataType::INT32: {
      auto out_data = dev_ctx.template Alloc<int32_t>(out);
      for (auto i = 0; i < numel; ++i) {
        out_data[i] = static_cast<int32_t>(static_cast<float>(x_data[i]));
      }
      break;
    }
    case phi::DataType::INT64: {
      auto out_data = dev_ctx.template Alloc<int64_t>(out);
      for (auto i = 0; i < numel; ++i) {
        out_data[i] = static_cast<int64_t>(static_cast<float>(x_data[i]));
      }
      break;
    }
    case phi::DataType::UINT8: {
      auto out_data = dev_ctx.template Alloc<uint8_t>(out);
      for (auto i = 0; i < numel; ++i) {
        out_data[i] = static_cast<uint8_t>(static_cast<float>(x_data[i]));
      }
      break;
    }
    case phi::DataType::BOOL: {
      auto out_data = dev_ctx.template Alloc<bool>(out);
      for (auto i = 0; i < numel; ++i) {
        out_data[i] = static_cast<bool>(static_cast<float>(x_data[i]));
      }
      break;
    }
    default:
      break;
  }
}

}  // namespace custom_kernel

PD_BUILD_PHI_KERNEL(cast,
                    custom_cpu,
                    ALL_LAYOUT,
                    custom_kernel::CastKernel,
                    float,
                    double,
                    int,
                    int64_t,
                    int16_t,
                    bool,
                    int8_t,
                    uint8_t,
                    phi::dtype::float16,
                    phi::dtype::bfloat16) {}
