/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/softmax_kernel.h"
#include "paddle/phi/kernels/funcs/softmax_impl.h"

PD_CUSTOM_KERNEL_REGISTER(softmax,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::SoftmaxKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

template class phi::funcs::SoftmaxFunctor<phi::CustomContext, phi::dtype::float16>;
template class phi::funcs::SoftmaxFunctor<phi::CustomContext, phi::dtype::bfloat16>;
template class phi::funcs::SoftmaxFunctor<phi::CustomContext, float>;
template class phi::funcs::SoftmaxGradFunctor<phi::CustomContext, float>;
template class phi::funcs::SoftmaxGradFunctor<phi::CustomContext, phi::dtype::float16>;
template class phi::funcs::SoftmaxGradFunctor<phi::CustomContext, phi::dtype::bfloat16>;
