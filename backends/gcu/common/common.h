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
#include <string>
#include <vector>

#include "dtu/hlir/dispatch.h"
#include "dtu/hlir/metadata.h"
#include "dtu/hlir/types.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/dense_tensor.h"
#include "runtime/gcu_memory.h"

namespace custom_kernel {
using LoDTensor = phi::DenseTensor;

void* GetOpFuncPtr(const std::string& name,
                   const hlir::DispatchParam& params,
                   bool include_output = true);

void* GetOpFuncPtr(const std::string& name,
                   const hlir::DispatchParam& params,
                   const std::vector<hlir::Tensor*>& tensors);

GcuMemory* GetGcuMemory(const phi::DenseTensor& tensor,
                        bool check_place = true);

hlir::Tensor* GetHlirTensor(const phi::DenseTensor& tensor);

hlir::Tensor* GetHlirTensorV2(const phi::DenseTensor& tensor,
                              const phi::DDim src_dims);

int GetGCUDataType(const phi::DataType& dtype);

bool GcuOpStreamSync(const phi::DeviceContext& dev_ctx);

void BuildDispatchParam(const std::vector<LoDTensor*>& inputs,
                        const std::vector<LoDTensor*>& outputs,
                        hlir::DispatchParam& params);  // NOLINT

void FreeDispatchParam(hlir::DispatchParam& params);  // NOLINT

int64_t GetCurrentTimestap();

double GetTimeCostMs(int64_t start_time, int64_t end_time);

std::string GetTargetName();

std::vector<int32_t> LayoutToVector(phi::DataLayout layout);

phi::DataLayout VectorToLayout(const std::vector<int32_t>& layout);

void LayoutConvertDims(const std::vector<int64_t>& dims,
                       const std::vector<int32_t>& src_layout,
                       const std::vector<int32_t>& dst_layout,
                       std::vector<int64_t>& out_permute_dims,   // NOLINT
                       std::vector<int64_t>& out_convert_dims);  // NOLINT

std::vector<int64_t> LayoutAffine(const std::vector<int32_t>& src_layout,
                                  const std::vector<int32_t>& dst_layout);

phi::DenseTensor EmptyTensor(
    const phi::DeviceContext& dev_ctx,
    const phi::DataType dtype,
    const phi::DDim& dims,
    const phi::DataLayout layout = phi::DataLayout::NCHW);

phi::DenseTensor EmptyTensor(const phi::DeviceContext& dev_ctx,
                             const phi::DenseTensorMeta& meta);

template <class T>
inline bool vector_contains(const std::vector<T>& values, T value) {
  return std::find(values.begin(), values.end(), value) != values.end();
}

inline bool is_valid_permutation(const std::vector<int64_t>& permutation) {
  auto size = permutation.size();
  std::vector<bool> flags(size, false);
  for (size_t i = 0; i < size; ++i) {
    auto k = permutation[i];
    if (k >= 0 && k < size && !flags[k])
      flags[k] = true;
    else
      return false;
  }

  return true;
}

template <class T>
inline std::vector<T> reorder_vector(const std::vector<T>& src,
                                     const std::vector<int64_t>& permutation) {
  PADDLE_ENFORCE(
      permutation.size() == src.size() && is_valid_permutation(permutation),
      phi::errors::InvalidArgument("Invalid permutation."));

  std::vector<T> dst(src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    dst[i] = src[permutation[i]];
  }

  return dst;
}

}  // namespace custom_kernel
