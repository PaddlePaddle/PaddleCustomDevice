// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <murand.h>
#include <thrust/system/musa/error.h>
#include <thrust/system_error.h>

using curandStatus_t = murandStatus_t;

using curandCreateGenerator = murandCreateGenerator;
using curandSetStream = murandSetStream;
using curandSetPseudoRandomGeneratorSeed = murandSetPseudoRandomGeneratorSeed;
using curandGenerateUniform = murandGenerateUniform;
using curandGenerateUniformDouble = murandGenerateUniformDouble;
using curandGenerateNormal = murandGenerateNormal;
using curandDestroyGenerator = murandDestroyGenerator;

#define CURAND_STATUS_SUCCESS MURAND_STATUS_SUCCESS
