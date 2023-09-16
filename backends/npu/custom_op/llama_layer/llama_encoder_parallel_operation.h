/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <atb/atb_infer.h>
#include <atb/svector.h>

struct LlamaLayerEncoderParallelParam {
  float rmsNormEps = 0;
  int headNum = 0;
  int dk = 0;
  int rank = 0;
  int rankSize = 1;
  std::string model = "llama7b";
  void *hcclComm = nullptr; // only effect when hcclComm is not null
};

atb::Status CreateLlamaLayerEncoderParallelOperation(const LlamaLayerEncoderParallelParam &param,
                                                     atb::Operation **operation);
