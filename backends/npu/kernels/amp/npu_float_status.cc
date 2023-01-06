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

#include "kernels/amp/npu_float_status.h"

#include "acl/acl_op_compiler.h"
#include "kernels/funcs/npu_enforce.h"
#include "kernels/funcs/string_helper.h"
#include "pybind11/pybind11.h"

const char OP_TYPE_NPU_GET_FLOAT_STATUS[] = "NPUGetFloatStatus";
const char OP_TYPE_NPU_CLEAR_FLOAT_STATUS[] = "NPUClearFloatStatus";

const int FLOAT_STATUS_OP_TENSOR_DIMS_SIZE = 8;
const int FLOAT_STATUS_OP_TENSOR_DESC_SIZE = 1;
const int FLOAT_STATUS_OVERFLOW = 1;

NPUFloatStatus::NPUFloatStatus() {
  std::vector<int64_t> shape{FLOAT_STATUS_OP_TENSOR_DIMS_SIZE};
  // float_status_desc_ init
  float_status_desc_ =
      aclCreateTensorDesc(ACL_FLOAT, shape.size(), shape.data(), ACL_FORMAT_ND);
  PADDLE_ENFORCE_NOT_NULL(float_status_desc_,
                          phi::errors::NotFound("Op desc is not initalized."));
  // float_status_addr_ init
  auto size = aclGetTensorDescSize(float_status_desc_);
  PADDLE_ENFORCE_NPU_SUCCESS(
      aclrtMalloc(&float_status_addr_, size, ACL_MEM_MALLOC_NORMAL_ONLY));
  PADDLE_ENFORCE_NOT_NULL(
      float_status_addr_,
      phi::errors::NotFound("The device memory is not initalized."));
  PADDLE_ENFORCE_NPU_SUCCESS(aclrtMemset(float_status_addr_, size, 0, size));
  // float_status_buff_ init
  float_status_buff_ = aclCreateDataBuffer(float_status_addr_, size);
  PADDLE_ENFORCE_NOT_NULL(
      float_status_buff_,
      phi::errors::NotFound("The input buffer is not initalized."));
  // attr_ init
  attr_ = aclopCreateAttr();
}

NPUFloatStatus::~NPUFloatStatus() {
  if (attr_) {
    aclopDestroyAttr(attr_);
  }
  if (float_status_desc_) {
    aclDestroyTensorDesc(float_status_desc_);
  }
  if (float_status_buff_) {
    PADDLE_ENFORCE_NPU_SUCCESS(aclDestroyDataBuffer(float_status_buff_));
  }
  if (float_status_addr_) {
    PADDLE_ENFORCE_NPU_SUCCESS(aclrtFree(float_status_addr_));
  }
}

aclTensorDesc* NPUFloatStatus::CreateOpDesc() {
  std::vector<int64_t> shape{FLOAT_STATUS_OP_TENSOR_DIMS_SIZE};
  auto desc =
      aclCreateTensorDesc(ACL_FLOAT, shape.size(), shape.data(), ACL_FORMAT_ND);
  PADDLE_ENFORCE_NOT_NULL(desc,
                          phi::errors::NotFound("Op desc is not initalized."));
  return desc;
}

void NPUFloatStatus::DestroyOpDesc(const aclTensorDesc* desc) {
  aclDestroyTensorDesc(desc);
}

aclDataBuffer* NPUFloatStatus::CreateOpBuff(const aclTensorDesc* desc) {
  auto size = aclGetTensorDescSize(desc);
  // device memory and buffer
  void* device_mem = nullptr;
  PADDLE_ENFORCE_NPU_SUCCESS(
      aclrtMalloc(&device_mem, size, ACL_MEM_MALLOC_NORMAL_ONLY));
  PADDLE_ENFORCE_NOT_NULL(
      device_mem,
      phi::errors::NotFound("The device memory is not initalized."));
  PADDLE_ENFORCE_NPU_SUCCESS(aclrtMemset(device_mem, size, 0, size));
  auto device_buf = aclCreateDataBuffer(device_mem, size);
  PADDLE_ENFORCE_NOT_NULL(
      device_buf, phi::errors::NotFound("The input buffer is not initalized."));
  return device_buf;
}

void NPUFloatStatus::DestroyOpBuff(const aclDataBuffer* buff) {
  PADDLE_ENFORCE_NPU_SUCCESS(aclrtFree(aclGetDataBufferAddr(buff)));
  PADDLE_ENFORCE_NPU_SUCCESS(aclDestroyDataBuffer(buff));
}

void NPUFloatStatus::RunOp(const std::string op_type,
                           aclTensorDesc* input_desc,
                           aclDataBuffer* input_buf,
                           aclTensorDesc* output_desc,
                           aclDataBuffer* output_buf,
                           aclrtStream stream) {
  // inputs
  std::vector<aclTensorDesc*> input_descs;
  std::vector<aclDataBuffer*> input_buffers;
  if (input_desc && input_buf) {
    input_descs.emplace_back(input_desc);
    input_buffers.emplace_back(input_buf);
  }
  // outputs
  std::vector<aclTensorDesc*> output_descs;
  std::vector<aclDataBuffer*> output_buffers;
  output_descs.emplace_back(output_desc);
  output_buffers.emplace_back(output_buf);

  VLOG(1) << "aclopCompileAndExecute start: " << op_type << "\n"
          << GetOpInfoString(input_descs, input_buffers, "Input")
          << GetOpInfoString(output_descs, output_buffers, "Output");

  if (PyGILState_Check()) {
    Py_BEGIN_ALLOW_THREADS PADDLE_ENFORCE_NPU_SUCCESS(
        aclopCompileAndExecute(op_type.c_str(),
                               input_descs.size(),
                               input_descs.data(),
                               input_buffers.data(),
                               output_descs.size(),
                               output_descs.data(),
                               output_buffers.data(),
                               attr_,
                               ACL_ENGINE_SYS,
                               ACL_COMPILE_SYS,
                               nullptr,
                               stream));
    Py_END_ALLOW_THREADS
  } else {
    PADDLE_ENFORCE_NPU_SUCCESS(aclopCompileAndExecute(op_type.c_str(),
                                                      input_descs.size(),
                                                      input_descs.data(),
                                                      input_buffers.data(),
                                                      output_descs.size(),
                                                      output_descs.data(),
                                                      output_buffers.data(),
                                                      attr_,
                                                      ACL_ENGINE_SYS,
                                                      ACL_COMPILE_SYS,
                                                      nullptr,
                                                      stream));
  }

  VLOG(1) << "aclopCompileAndExecute finish: " << op_type << "\n"
          << GetOpInfoString(input_descs, input_buffers, "Input")
          << GetOpInfoString(output_descs, output_buffers, "Output");

  PADDLE_ENFORCE_NPU_SUCCESS(aclrtSynchronizeStream(stream));
}

void NPUFloatStatus::RunClearFloatStatusOp(aclrtStream stream) {
  auto input_desc = CreateOpDesc();
  auto input_buff = CreateOpBuff(input_desc);

  RunOp(OP_TYPE_NPU_CLEAR_FLOAT_STATUS,
        input_desc,
        input_buff,
        float_status_desc_,
        float_status_buff_,
        stream);

  // destroy temp inputs
  DestroyOpBuff(input_buff);
  DestroyOpDesc(input_desc);
}

bool NPUFloatStatus::RunGetFloatStatusOp(aclrtStream stream) {
  auto output_desc = CreateOpDesc();
  auto output_buff = CreateOpBuff(output_desc);

  RunOp(OP_TYPE_NPU_GET_FLOAT_STATUS,
        float_status_desc_,
        float_status_buff_,
        output_desc,
        output_buff,
        stream);

  // copy value to host
  auto size = aclGetTensorDescSize(float_status_desc_);
  void* host_mem = nullptr;
  PADDLE_ENFORCE_NPU_SUCCESS(aclrtMallocHost(&host_mem, size));
  PADDLE_ENFORCE_NOT_NULL(
      host_mem, phi::errors::NotFound("The host memory is not initalized."));
  PADDLE_ENFORCE_NPU_SUCCESS(aclrtMemcpy(
      host_mem, size, float_status_addr_, size, ACL_MEMCPY_DEVICE_TO_HOST));

  // check value for overflow
  bool overflowFlag = false;
  auto host_ptr = reinterpret_cast<const float*>(host_mem);
  if (FLOAT_STATUS_OVERFLOW == host_ptr[0]) {
    overflowFlag = true;
    LOG(INFO) << "Ascend NPU float status is overflow";
  }

  PADDLE_ENFORCE_NPU_SUCCESS(aclrtFreeHost(host_mem));
  DestroyOpBuff(output_buff);
  DestroyOpDesc(output_desc);

  return overflowFlag;
}
