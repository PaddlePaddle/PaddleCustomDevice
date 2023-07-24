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

#include "paddle/phi/capi/all.h"
#include "phi_funcs.h"  //NOLINT

namespace custom_kernel {

template <typename T>
void SliceRawKernel(const phi::Context& ctx,
                    const phi::DenseTensor& input,
                    const std::vector<int64_t>& axes,
                    const phi::IntArray& starts_arr,
                    const phi::IntArray& ends_arr,
                    const std::vector<int64_t>& infer_flags,
                    const std::vector<int64_t>& decrease_axis,
                    phi::DenseTensor* out) {
  // Step 1: Get the accurate attribute value of starts and ends
  auto starts = starts_arr.GetData();
  auto ends = ends_arr.GetData();
  PD_CHECK(starts.size() == axes.size(),
           "The size of starts must be equal to the size of axes.");
  PD_CHECK(ends.size() == axes.size(),
           "The size of ends must be equal to the size of axes.");

  // Step 2: Compute output
  auto in = &input;
  auto in_data = input.data<T>();
  int rank = input.dims().size();

  auto in_dims = in->dims();
  auto out_dims = out->dims();
  auto slice_dims = out_dims;

  // 2.1 Infer output dims
  for (size_t i = 0; i < axes.size(); ++i) {
    // when start == -1 && end == start+1
    if (starts[i] == -1 && ends[i] == 0 && infer_flags[i] == -1) {
      auto ret = std::find(decrease_axis.begin(), decrease_axis.end(), axes[i]);
      if (ret != decrease_axis.end()) {
        ends[i] = in_dims[axes[i]];
      }
    }
  }

  phi::funcs::CheckAndUpdateSliceAttrs<int64_t>(in_dims, axes, &starts, &ends);
  slice_dims = phi::funcs::GetSliceDims<int64_t>(
      in_dims, axes, starts, ends, nullptr, nullptr);
  out_dims = phi::funcs::GetDecreasedDims<int64_t>(slice_dims, decrease_axis);

  // 2.2 Get output
  auto offsets = std::vector<size_t>(rank);
  auto extents = std::vector<size_t>(rank);

  for (size_t i = 0; i < rank; ++i) {
    offsets[i] = 0;
    extents[i] = slice_dims[i];
  }
  for (size_t i = 0; i < axes.size(); ++i) {
    offsets[axes[i]] = starts[i];
  }

  out->Resize(slice_dims);
  auto out_data = ctx.template Alloc<T>(out);

  std::vector<size_t> in_step(rank, 1);
  for (auto i = rank - 1; i > 0; --i) {
    in_step[i - 1] = in_step[i] * in_dims[i];
  }

  auto numel = phi::product(slice_dims);
  auto index = std::vector<size_t>(offsets.cbegin(), offsets.cend());
  for (auto i = 0; i < numel; i += slice_dims.back()) {
    memcpy(out_data + i,
           in_data + phi::vec_product(index, in_step),
           sizeof(T) * slice_dims.back());
    index[index.size() - 2]++;
    for (auto j = index.size() - 2; j > 0; --j) {
      if (index[j] >= offsets[j] + extents[j]) {
        index[j] = offsets[j];
        index[j - 1] += 1;
      } else {
        break;
      }
    }
  }
  out->Resize(out_dims);
}

}  // namespace custom_kernel

PD_BUILD_PHI_KERNEL(slice,
                    custom_cpu,
                    ALL_LAYOUT,
                    custom_kernel::SliceRawKernel,
                    bool,
                    int8_t,
                    int16_t,
                    int32_t,
                    int64_t,
                    uint8_t,
                    float,
                    double) {}
