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

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/interpolate_kernel.h"

PD_CUSTOM_KERNEL_REGISTER(bilinear_interp,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::BilinearInterpKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          int) {
  kernel->InputAt(1).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(2).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(3).SetBackend(phi::Backend::ALL_BACKEND);
}
PD_CUSTOM_KERNEL_REGISTER(legacy_bilinear_interp,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::LegacyBilinearInterpKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          int) {
  kernel->InputAt(1).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(2).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(3).SetBackend(phi::Backend::ALL_BACKEND);
}
PD_CUSTOM_KERNEL_REGISTER(nearest_interp,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::NearestInterpKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          int,
                          int64_t) {
  kernel->InputAt(1).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(2).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(3).SetBackend(phi::Backend::ALL_BACKEND);
}
PD_CUSTOM_KERNEL_REGISTER(legacy_nearest_interp,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::LegacyNearestInterpKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          int,
                          int64_t) {
  kernel->InputAt(1).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(2).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(3).SetBackend(phi::Backend::ALL_BACKEND);
}
PD_CUSTOM_KERNEL_REGISTER(trilinear_interp,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::TrilinearInterpKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          int) {
  kernel->InputAt(1).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(2).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(3).SetBackend(phi::Backend::ALL_BACKEND);
}
PD_CUSTOM_KERNEL_REGISTER(linear_interp,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::LinearInterpKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          int) {
  kernel->InputAt(1).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(2).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(3).SetBackend(phi::Backend::ALL_BACKEND);
}
PD_CUSTOM_KERNEL_REGISTER(bicubic_interp,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::BicubicInterpKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          int) {
  kernel->InputAt(1).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(2).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(3).SetBackend(phi::Backend::ALL_BACKEND);
}
