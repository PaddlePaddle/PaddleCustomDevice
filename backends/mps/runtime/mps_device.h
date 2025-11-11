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

#pragma once

#include <Foundation/Foundation.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
typedef id<MTLDevice> MTLDevice_t;
typedef id<MTLFunction> MTLFunction_t;

namespace mps {

class MPSDevice {
 public:
  MPSDevice(const MPSDevice& other) = delete;
  void operator=(const MPSDevice&) = delete;
  static MPSDevice* getInstance();
  MTLDevice_t device() { return _mtl_device; }
  ~MPSDevice();

 private:
  MTLDevice_t _mtl_device;
  MPSDevice();
};

}  // namespace mps
