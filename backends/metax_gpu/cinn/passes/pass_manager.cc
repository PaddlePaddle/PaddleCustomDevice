// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include <cstring>
#include <iostream>

#include "paddle/phi/backends/device_ext.h"

namespace paddle {
namespace custom_device {
namespace metax {

// ============================================================
// MetaxApplyCustomPass
// ============================================================
// Called by the CINN framework when it encounters a pass name in the pipeline
// that is NOT a built-in pass.  `ir_func` is a `cinn::ir::LoweredFunc*` cast
// to void*.
C_Status MetaxApplyCustomPass(void* dev_ptr,
                              const char* pass_name,
                              void* ir_func) {
  std::string name(pass_name);

  if (name == "MetaxDebugLogPass") {
    // A trivial pass that simply logs the function pointer address.
    // Demonstrates the custom-pass mechanism without modifying IR.
    std::cout << "[MetaX] MetaxDebugLogPass: ir_func=" << ir_func << std::endl;
    return C_Status::C_SUCCESS;
  }

  std::cerr << "[MetaX] Unknown custom pass: " << name << std::endl;
  return C_Status::C_FAILED;
}

// ============================================================
// MetaxQueryPassPipeline
// ============================================================
// Defines the ordered pass pipeline for MetaX GPU hardware.
//
// Rules:
//   - Built-in pass names (understood by CINN) are executed by the framework.
//   - Unknown names are forwarded to MetaxApplyCustomPass().
C_Status MetaxQueryPassPipeline(void* dev_ptr,
                                char pass_names[][128],
                                int* count) {
  // Full NVGPU-equivalent pipeline with one custom pass inserted.
  static const char* kPipeline[] = {
      "Simplify",
      "EliminateInvariantLoop",
      "RealizeCompositeReduce",
      "ReindexTransposeBuffer",
      "ReplaceCrossThreadReduction",
      "ReplaceCrossBlockReduction",
      "SetCudaAxisInfo",
      "RemoveGpuForLoops",
      "CudaSyncThreadsDropIfThenElse",
      "TransBufferWithDynamicShape",
      "SimplifyUnitBlock",
      "MapExternCall",
      "ExternCallMultiOutputShallowStore",
      "Simplify",
      "IfFusion",
      "EntailLoopCondition",
      // Vendor-defined custom pass: forwarded to MetaxApplyCustomPass().
      "MetaxDebugLogPass",
      "RearrangeLoadInstruction",
      "VectorizeForTrans",
      "Simplify",
      "RemoveScheduleBlock",
      "IfFold",
      "LowerIntrin",
      "PrepareBufferCastExprs",
  };
  static const int kPipelineSize =
      static_cast<int>(sizeof(kPipeline) / sizeof(kPipeline[0]));

  // If pass_names is null the caller only wants the count.
  if (pass_names == nullptr) {
    *count = kPipelineSize;
    return C_Status::C_SUCCESS;
  }

  // Check buffer capacity.
  if (*count < kPipelineSize) {
    *count = kPipelineSize;
    return C_Status::C_FAILED;
  }

  for (int i = 0; i < kPipelineSize; ++i) {
    std::strncpy(pass_names[i], kPipeline[i], 127);
    pass_names[i][127] = '\0';
  }
  *count = kPipelineSize;
  return C_Status::C_SUCCESS;
}

}  // namespace metax
}  // namespace custom_device
}  // namespace paddle
