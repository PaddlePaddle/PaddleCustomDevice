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

#include "paddle/phi/common/complex.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/gpu/reduce.h"
#include "paddle/phi/kernels/legacy/reduce_max_kernel.h"
#include "paddle/phi/kernels/prod_kernel.h"
#include "paddle/phi/kernels/reduce_all_kernel.h"
#include "paddle/phi/kernels/reduce_amax_kernel.h"
#include "paddle/phi/kernels/reduce_amin_kernel.h"
#include "paddle/phi/kernels/reduce_any_kernel.h"
#include "paddle/phi/kernels/reduce_max_kernel.h"
#include "paddle/phi/kernels/reduce_mean_kernel.h"
#include "paddle/phi/kernels/reduce_min_kernel.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"

using float16 = phi::dtype::float16;
using bfloat16 = phi::dtype::bfloat16;
using complex64 = ::phi::dtype::complex<float>;
using complex128 = ::phi::dtype::complex<double>;

PD_CUSTOM_KERNEL_REGISTER(all_raw,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::AllRawKernel,
                          float,
                          int,
                          int64_t,
                          bool,
                          complex64) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}

PD_CUSTOM_KERNEL_REGISTER(amax_raw,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::AMaxRawKernel,
                          float,
                          int,
                          int64_t) {}

PD_CUSTOM_KERNEL_REGISTER(amin_raw,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::AMinRawKernel,
                          float,
                          int,
                          int64_t) {}

PD_CUSTOM_KERNEL_REGISTER(any_raw,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::AnyRawKernel,
                          float,
                          int,
                          int64_t,
                          bool,
                          complex64) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}

PD_CUSTOM_KERNEL_REGISTER(max,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::MaxKernel,
                          float,
                          int,
                          int64_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(mean_raw,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::MeanRawKernel,
                          float,
                          bool,
                          phi::dtype::bfloat16,
                          float16,
                          int,
                          int64_t,
                          phi::dtype::complex<float>) {}

PD_CUSTOM_KERNEL_REGISTER(min_raw,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::MinRawKernel,
                          float,
                          int,
                          int64_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(sum_raw,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::SumRawKernel,
                          bool,
                          float,
                          float16,
                          bfloat16,
                          int8_t,
                          uint8_t,
                          int16_t,
                          int,
                          int64_t,
                          complex64) {
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
}

PD_CUSTOM_KERNEL_REGISTER(prod,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::ProdKernel,
                          float,
                          int,
                          int64_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          phi::dtype::complex<float>) {}
