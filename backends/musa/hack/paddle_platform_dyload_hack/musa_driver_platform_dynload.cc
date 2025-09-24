/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Modifications:
Copyright (c) 2025 Moore Threads Technology Co., Ltd("Moore Threads"). All
rights reserved.

- [modify the code to adapt to musa backend] */
#include "hack/paddle_platform_dyload_hack/musa_driver_platform_dynload.h"

#include "hack/paddle_backend_dyload_hack/musa_driver_dynload.h"

namespace paddle {
namespace platform {
namespace dynload {

#define DEFINE_WRAP(__name) DynLoad__##__name __name

MUSA_ROUTINE_EACH(DEFINE_WRAP);

bool HasCUDADriver() { return phi::dynload::HasCUDADriver(); }

}  // namespace dynload
}  // namespace platform
}  // namespace paddle
