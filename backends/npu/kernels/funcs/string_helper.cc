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

template <typename Context, typename T>
void FormatData(const Context& dev_ctx,
                const phi::DenseTensor& print_tensor,
                std::stringstream& log_stream) {
  int64_t print_size = print_tensor.numel();
  const T* data = nullptr;
  phi::DenseTensor cpu_tensor;
  if (print_tensor.storage_properties_initialized()) {
    paddle::Tensor x_tensor(std::make_shared<phi::DenseTensor>(print_tensor));
    auto temp = paddle::npu_identity(x_tensor);
    auto dense_tensor =
        std::dynamic_pointer_cast<phi::DenseTensor>(temp.impl());
    custom_kernel::TensorCopy(
        dev_ctx, *dense_tensor, false, &cpu_tensor, phi::CPUPlace());
  } else {
    custom_kernel::TensorCopy(
        dev_ctx, print_tensor, false, &cpu_tensor, phi::CPUPlace());
  }
  dev_ctx.Wait();
  data = cpu_tensor.data<T>();
  log_stream << "  - data: [";
  if (print_size > 0) {
    log_stream << data[0];
    for (int64_t i = 1; i < print_size; ++i) {
      log_stream << " " << data[i];
    }
  }
  log_stream << "]" << std::endl;
}

template <typename Context>
std::string GetPDTensorString(const Context& dev_ctx,
                              const phi::DenseTensor& print_tensor,
                              const std::string& tensor_name,
                              const std::string& message) {
  static std::mutex mutex;
  std::lock_guard<std::mutex> lock(mutex);
  std::stringstream log_stream;
  if (!tensor_name.empty()) {
    log_stream << "Variable: " << tensor_name << std::endl;
  }

  if (!message.empty()) {
    log_stream << "  - message: " << message << std::endl;
  }

  log_stream << "  - place: " << print_tensor.place() << std::endl;
  log_stream << "  - shape: " << print_tensor.dims().to_str() << std::endl;
  log_stream << "  - layout: " << phi::DataLayoutToString(print_tensor.layout())
             << std::endl;

  auto dtype = print_tensor.dtype();
  log_stream << "  - dtype: " << dtype << std::endl;

  if (dtype == phi::DataType::FLOAT32) {
    FormatData<Context, float>(dev_ctx, print_tensor, log_stream);
  } else if (dtype == phi::DataType::FLOAT64) {
    FormatData<Context, double>(dev_ctx, print_tensor, log_stream);
  } else if (dtype == phi::DataType::INT32) {
    FormatData<Context, int>(dev_ctx, print_tensor, log_stream);
  } else if (dtype == phi::DataType::INT64) {
    FormatData<Context, int64_t>(dev_ctx, print_tensor, log_stream);
  } else if (dtype == phi::DataType::BOOL) {
    FormatData<Context, bool>(dev_ctx, print_tensor, log_stream);
  } else if (dtype == phi::DataType::FLOAT16) {
    FormatData<Context, phi::dtype::float16>(dev_ctx, print_tensor, log_stream);
  } else {
    log_stream << "  - data: unprintable type: " << dtype << std::endl;
  }
  return log_stream.str();
}
