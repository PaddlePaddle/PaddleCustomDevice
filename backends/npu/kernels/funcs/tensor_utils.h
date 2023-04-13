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
#include <algorithm>
#include <codecvt>
#include <locale>
#include <string>
#include <unordered_map>
#include <vector>

#include "glog/logging.h"
#include "kernels/funcs/npu_funcs.h"
#include "paddle/extension.h"
#include "paddle/phi/extension.h"

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
std::string PrintTensor(const Context& dev_ctx,
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
