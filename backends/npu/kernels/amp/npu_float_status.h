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

#pragma once

// Step0: NPUAllocFloatStatus => equal to malloc and memset float_status_ptr
//        - Input: NULL
//        - Output: float_status_ptr
// ONLY need to init once, and replaced by NPUFloatStatus constructor

// Loop the following steps for each NPU OP to check
// Step1: NPUClearFloatStatus
//        - Input: TEMP
//        - Output: float_status_ptr => memset values to ZERO
// Step2: aclopCompileAndExecute the NPU OP need to check
// Step3: NPUGetFloatStatus
//        - Input: float_status_ptr => get overflow status of NPU OP
//        - Output: TEMP
// COPY float_status_ptr to CPU, and its value should be ZERO if no overflow

class NPUFloatStatus {
 public:
  static NPUFloatStatus& Instance() {
    static NPUFloatStatus g_npu_float_status;
    return g_npu_float_status;
  }

  void RunClearFloatStatusOp(aclrtStream stream);
  bool RunGetFloatStatusOp(aclrtStream stream);

 private:
  NPUFloatStatus();
  NPUFloatStatus(const NPUFloatStatus& runner) = delete;
  NPUFloatStatus& operator=(const NPUFloatStatus& runner) = delete;
  ~NPUFloatStatus();

  aclTensorDesc* CreateOpDesc();
  void DestroyOpDesc(const aclTensorDesc* desc);

  aclDataBuffer* CreateOpBuff(const aclTensorDesc* desc);
  void DestroyOpBuff(const aclDataBuffer* buff);

  void RunOp(const std::string op_type,
             aclTensorDesc* input_desc,
             aclDataBuffer* input_buf,
             aclTensorDesc* output_desc,
             aclDataBuffer* output_buf,
             aclrtStream stream);

 private:
  aclopAttr* attr_{nullptr};
  aclTensorDesc* float_status_desc_{nullptr};
  aclDataBuffer* float_status_buff_{nullptr};
  void* float_status_addr_{nullptr};
};
