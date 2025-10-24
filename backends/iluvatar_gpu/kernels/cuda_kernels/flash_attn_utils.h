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

#include <ixattnbkd.h>

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/reduce_function.h"
#include "runtime/iluvatar_context.h"

namespace phi {
static void RaiseNotSupportedError() {
  PADDLE_THROW(
      phi::errors::Unimplemented("FlashAttention is unsupported, please check "
                                 "the GPU compability and CUDA Version."));
}

inline int RoundSeq(int seq) {
  constexpr int SEQ_BASE = 128;
  return (seq + SEQ_BASE - 1) / SEQ_BASE * SEQ_BASE;
}

struct PhiloxCudaState {
  PhiloxCudaState() = default;
  // Called if graph capture is not underway
  PhiloxCudaState(uint64_t seed, uint64_t offset) {
    seed_.val = seed;
    offset_.val = offset;
  }
  // Called if graph capture is underway
  PhiloxCudaState(int64_t* seed,
                  int64_t* offset_extragraph,
                  uint32_t offset_intragraph) {
    seed_.ptr = seed;
    offset_.ptr = offset_extragraph;
    offset_intragraph_ = offset_intragraph;
    captured_ = true;
  }

  // Public members, directly accessible by at::cuda::philox::unpack.
  // If we made them private with getters/setters, the getters/setters
  // would have to be __device__, and we can't declare __device__ in ATen.
  union Payload {
    uint64_t val;
    int64_t* ptr;
  };

  Payload seed_;
  Payload offset_;
  uint32_t offset_intragraph_ = 0;
  bool captured_ = false;
};

inline void copyDimsAndStrides(const DenseTensor& tensor,
                               std::vector<int32_t>& shape,     // NOLINT
                               std::vector<int32_t>& stride) {  // NOLINT
  for (size_t i = 0; i < tensor.dims().size(); ++i) {
    shape.push_back(static_cast<int>(tensor.dims()[i]));
    stride.push_back(static_cast<int>(tensor.strides()[i]));
  }
}

inline void copyDimsAndStrides(DenseTensor* tensor,
                               std::vector<int32_t>& shape,     // NOLINT
                               std::vector<int32_t>& stride) {  // NOLINT
  for (size_t i = 0; i < tensor->dims().size(); ++i) {
    shape.push_back(static_cast<int>(tensor->dims()[i]));
    stride.push_back(static_cast<int>(tensor->strides()[i]));
  }
}

inline void SetIxAttnBkdTensor(ixAttnBkdTensorDesc* desc,
                               const phi::DenseTensor& t,
                               ixAttnBkdDataType_t dataType) {
  PADDLE_ENFORCE_NOT_NULL(desc,
                          phi::errors::InvalidArgument(
                              "Output tensor desc pointer cannot be null."));
  desc->n_dims = t.dims().size();
  std::vector<int64_t> dims = phi::vectorize(t.dims());
  desc->dim = std::vector<size_t>(dims.begin(), dims.end());
  desc->dataType = dataType;
  std::vector<int64_t> strides = phi::vectorize(t.strides());
  desc->stride = std::vector<size_t>(strides.begin(), strides.end());
}

inline void SetIxAttnBkdTensor(ixAttnBkdTensorDesc* desc,
                               const phi::DenseTensor* t,
                               ixAttnBkdDataType_t dataType) {
  PADDLE_ENFORCE_NOT_NULL(
      t, phi::errors::InvalidArgument("Input tensor pointer cannot be null."));
  PADDLE_ENFORCE_NOT_NULL(desc,
                          phi::errors::InvalidArgument(
                              "Output tensor desc pointer cannot be null."));
  desc->n_dims = t->dims().size();
  std::vector<int64_t> dims = phi::vectorize(t->dims());
  desc->dim = std::vector<size_t>(dims.begin(), dims.end());
  desc->dataType = dataType;
  std::vector<int64_t> strides = phi::vectorize(t->strides());
  desc->stride = std::vector<size_t>(strides.begin(), strides.end());
}

#define PADDLE_IXATTNBKD_CHECK(EXPR)                                     \
  do {                                                                   \
    ixAttnBkdStatus_t __err = EXPR;                                      \
    PADDLE_ENFORCE_EQ(__err,                                             \
                      IXATTNBKD_STATUS_SUCCESS,                          \
                      common::errors::External(                          \
                          "Ixattnbkd error, when calling `%s`", #EXPR)); \
  } while (0)
}  // namespace phi
