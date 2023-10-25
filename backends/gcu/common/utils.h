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
#include <string>
#include <utility>
#include <vector>

#include "dtu/hlir/dispatch.h"
#include "dtu/hlir/library.h"
#include "dtu/hlir/library_attributes.h"
#include "dtu/hlir/metadata.h"
#include "dtu/hlir/types.h"
#include "topstx/topstx.hpp"

namespace custom_kernel {
using HlirVector = hlir::VectorMetaData<int64_t>;
using HlirShape = hlir::ShapeMetaData<int64_t>;

std::string HLIRTensorToString(const std::vector<hlir::Tensor*>& tensors,
                               bool is_inputs);

template <typename T>
inline std::vector<int64_t> vector_s64(const std::vector<T>& dims) {
  std::vector<int64_t> s64_dims(dims.size());
  for (size_t i = 0; i < dims.size(); ++i)
    s64_dims[i] = static_cast<int64_t>(dims[i]);
  return s64_dims;
}

template <typename T>
inline HlirVector VectorToHlirVector(const std::vector<T>& dims) {
  return HlirVector(vector_s64(dims));
}

template <typename T>
inline HlirShape VectorToHlirShape(const std::vector<T>& dims) {
  return HlirShape(vector_s64(dims), {static_cast<int64_t>(dims.size())});
}

// for conv
// ref: https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2
static std::vector<int64_t> get_same_padding_value(int64_t dim,
                                                   int64_t ksize,
                                                   int64_t stride) {
  int64_t pad_along_dim = 0;
  if (dim % stride == 0) {
    pad_along_dim = std::max(ksize - stride, static_cast<int64_t>(0));
  } else {
    pad_along_dim = std::max(ksize - (dim % stride), static_cast<int64_t>(0));
  }
  int64_t pad_low = pad_along_dim / 2;
  int64_t pad_high = pad_along_dim - pad_low;
  std::vector<int64_t> padding{pad_low, pad_high};
  return std::move(padding);
}

static std::vector<int64_t> get_conv2d_transpose_padding(
    const std::vector<int64_t>& input_spatial_dims,
    const std::vector<int64_t>& output_spatial_dims,
    const std::vector<int64_t>& ksize,
    const std::vector<int64_t>& stride,
    const std::vector<int64_t>& dilation,
    const std::vector<int64_t>& padding,
    const std::vector<int64_t>& output_padding,
    const std::string& auto_pad) {
  std::vector<int64_t> padding_value;
  for (size_t i = 0; i < input_spatial_dims.size(); ++i) {
    int64_t expanded_input_size = (input_spatial_dims[i] - 1) * stride[i] + 1;
    int64_t effective_filter_size = (ksize[i] - 1) * dilation[i] + 1;
    int64_t pad_before = effective_filter_size - 1 - padding[i * 2];
    int64_t padded_out_size =
        output_spatial_dims[i] + effective_filter_size - 1;
    int64_t pad_after = padded_out_size - expanded_input_size - pad_before;
    padding_value.emplace_back(pad_before);
    padding_value.emplace_back(pad_after);
  }
  return padding_value;
}

static int64_t get_conv_transpose_dim(int64_t input_dim,
                                      int64_t ksize,
                                      int64_t stride,
                                      int64_t dilation,
                                      int64_t pad_low,
                                      int64_t pad_high,
                                      int64_t output_padding) {
  int64_t expanded_input_size = (input_dim - 1) * stride + 1;
  int64_t effective_filter_size = (ksize - 1) * dilation + 1;
  int64_t output_dim = expanded_input_size - 1 + output_padding +
                       effective_filter_size - pad_low - pad_high;
  return output_dim;
}
}  // namespace custom_kernel

#define AOTOPS_DEBUG(op_name, params)                                 \
  VLOG(6) << op_name << "\ninputs: \n"                                \
          << custom_kernel::HLIRTensorToString(params.inputs, true)   \
          << "outputs: \n"                                            \
          << custom_kernel::HLIRTensorToString(params.outputs, false) \
          << "metadata: " << params.metadata;

enum {
  GCUOPS,
  PADDLE_GCU_KERNELS,
  EXEC,
  DISPATCH,
};

class PaddleGcuTrace {
 public:
  static const topstx::Domain& domain() {
    static PaddleGcuTrace inst;
    return inst.domain_;
  }

  PaddleGcuTrace() : domain_("PADDLE_GCU") {
    topstx::domainNameCategory(domain_, GCUOPS, "GCUOPS");
    topstx::domainNameCategory(
        domain_, PADDLE_GCU_KERNELS, "PADDLE_GCU_KERNELS");
    topstx::domainNameCategory(domain_, EXEC, "EXEC");
    topstx::domainNameCategory(domain_, DISPATCH, "DISPATCH");
  }

  topstx::Domain domain_;
};

#define PADDLE_GCU_TRACE_START(category, name)                    \
  topstxRangeId_t rngid_##category##_##name = topstx::rangeStart( \
      PaddleGcuTrace::domain(), topstx::Message(category, #name))
#define PADDLE_GCU_TRACE_END(category, name) \
  topstx::rangeEnd(PaddleGcuTrace::domain(), rngid_##category##_##name)

#define GCUOPS_TRACE_START(name) PADDLE_GCU_TRACE_START(GCUOPS, name)
#define GCUOPS_TRACE_END(name) PADDLE_GCU_TRACE_END(GCUOPS, name)

#define PADDLE_GCU_KERNELS_TRACE_START(name) \
  PADDLE_GCU_TRACE_START(PADDLE_GCU_KERNELS, name)
#define PADDLE_GCU_KERNELS_TRACE_END(name) \
  PADDLE_GCU_TRACE_END(PADDLE_GCU_KERNELS, name)

#define PADDLE_GCU_KERNEL_START(dev_ctx, type_string, kernel) \
  PADDLE_GCU_KERNELS_TRACE_START(kernel);                     \
  VLOG(1) << "start run op_type: " << type_string;            \
  InitResource(dev_ctx.GetPlace().GetDeviceId());

#define PADDLE_GCU_KERNEL_END(type_string, kernel) \
  PADDLE_GCU_KERNELS_TRACE_END(kernel);            \
  VLOG(1) << "end run op_type: " << type_string;
