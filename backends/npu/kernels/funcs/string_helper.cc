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

#include "kernels/funcs/string_helper.h"

#include "kernels/funcs/npu_enforce.h"

template <typename T>
std::string GetVectorString(const std::vector<T>& vec) {
  std::stringstream ss;
  int i = 0;
  ss << "[";
  for (auto e : vec) {
    if (i++ > 0) ss << ", ";
    ss << e;
  }
  ss << "]";
  return ss.str();
}

template std::string GetVectorString(const std::vector<int64_t>& vec);
template std::string GetVectorString(const std::vector<float>& vec);

std::string GetDataBufferString(const aclDataBuffer* buf) {
  auto size = aclGetDataBufferSizeV2(buf);
  auto addr = aclGetDataBufferAddr(buf);
  auto numel = size / sizeof(float);
  std::vector<float> cpu_data(numel, 0);
  PADDLE_ENFORCE_NPU_SUCCESS(aclrtMemcpy(
      cpu_data.data(), size, addr, size, ACL_MEMCPY_DEVICE_TO_HOST));

  std::stringstream ss;
  ss << "TensorData: " << GetVectorString(cpu_data);
  // for (auto value : cpu_data) {
  //   ss << value << ",";
  // }
  // ss << "]";
  return ss.str();
}

std::string GetTensorDescString(const aclTensorDesc* desc) {
  auto data_type = aclGetTensorDescType(desc);
  auto origin_format = aclGetTensorDescFormat(desc);  // origin format

  std::stringstream ss;
  ss << "TensorDesc: data_type = " << data_type
     << ", origin_format = " << origin_format << ", origin_dims = [";

  size_t rank = aclGetTensorDescNumDims(desc);
  for (auto i = 0; i < rank; ++i) {
    int64_t dim_size = -1;
    PADDLE_ENFORCE_NPU_SUCCESS(aclGetTensorDescDimV2(desc, i, &dim_size));
    ss << dim_size;
    if (i < rank - 1) {
      ss << ", ";
    }
  }
  ss << "]";
  return ss.str();
}

std::string GetOpDescString(std::vector<aclTensorDesc*> descs,
                            const std::string msg) {
  std::stringstream ss;
  for (auto i = 0; i < descs.size(); ++i) {
    ss << " - " << msg << "[" << std::to_string(i)
       << "]: ";  // Input[i] or Output[i]
    ss << GetTensorDescString(descs[i]) << "\n";
  }
  return ss.str();
}

std::string GetOpInfoString(std::vector<aclTensorDesc*> descs,
                            std::vector<aclDataBuffer*> buffs,
                            const std::string msg) {
  PADDLE_ENFORCE_EQ(buffs.size(),
                    descs.size(),
                    phi::errors::InvalidArgument(
                        "Input size of buffers and descs should be same, but "
                        "got buff size [%d] and desc size [%d]",
                        buffs.size(),
                        descs.size()));

  std::stringstream ss;
  for (auto i = 0; i < descs.size(); ++i) {
    ss << msg << "[" << std::to_string(i) << "]: ";  // Input[i] or Output[i]
    ss << GetTensorDescString(descs[i]) << "\n";
    ss << GetDataBufferString(buffs[i]) << "\n";
  }
  return ss.str();
}
