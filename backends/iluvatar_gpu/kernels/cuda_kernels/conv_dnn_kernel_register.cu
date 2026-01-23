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

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/gpudnn/conv_kernel.cu"  // NOLINT

PD_CUSTOM_KERNEL_REGISTER(conv2d,
                          GPUDNN,
                          ALL_LAYOUT,
                          phi::ConvCudnnKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(conv3d,
                          GPUDNN,
                          ALL_LAYOUT,
                          phi::Conv3DCudnnKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}
