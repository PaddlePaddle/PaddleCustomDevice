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

#include "glog/logging.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cudnn_lstm_kernel.h"
#include "paddle/phi/kernels/gpu/cudnn_lstm_utils.h"

#ifdef PADDLE_WITH_HIP
PD_CUSTOM_KERNEL_REGISTER(
    cudnn_lstm, metax_gpu, ALL_LAYOUT, phi::CudnnLSTMKernel, float) {
  kernel->InputAt(5).SetDataType(phi::DataType::INT32);
  kernel->OutputAt(3).SetDataType(phi::DataType::UINT8);
  kernel->OutputAt(4).SetDataType(phi::DataType::UINT8);
}
#else
PD_CUSTOM_KERNEL_REGISTER(
    cudnn_lstm, metax_gpu, ALL_LAYOUT, phi::CudnnLSTMKernel, float, double) {
  kernel->InputAt(5).SetDataType(phi::DataType::INT32);
  kernel->OutputAt(3).SetDataType(phi::DataType::UINT8);
  kernel->OutputAt(4).SetDataType(phi::DataType::UINT8);
}
#endif
