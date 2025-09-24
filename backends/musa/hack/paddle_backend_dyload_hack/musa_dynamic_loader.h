/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

- [Modify the relevant code to adapt to musa backend]*/

#pragma once
#include <string>

#include "paddle/phi/backends/dynload/dynamic_loader.h"

namespace phi {
namespace dynload {

void* GetMublasDsoHandle();
void* GetMUDNNDsoHandle();
void* GetMUPTIDsoHandle();
void* GetMurandDsoHandle();
void* GetMusolverDsoHandle();
void* GetMusparseDsoHandle();
void* GetMURTCDsoHandle();
void* GetMUSADsoHandle();
void* GetMCCLDsoHandle();
void* GetMUFFTDsoHandle();

}  // namespace dynload
}  // namespace phi
