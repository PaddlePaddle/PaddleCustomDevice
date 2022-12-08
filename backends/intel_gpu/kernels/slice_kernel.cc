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
#include "dnn_support.hpp"
#include "paddle/phi/capi/all.h"
#include "phi_funcs.h"

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

  show_kernel("SliceRawKernel, type="<< dnn_support::type2String<T>::name());

  // Step 1: Get the accurate attribute value of starts and ends
  auto starts = starts_arr.GetData();
  auto ends = ends_arr.GetData();
  PD_CHECK(starts.size() == axes.size(),
           "The size of starts must be equal to the size of axes.");
  PD_CHECK(ends.size() == axes.size(),
           "The size of ends must be equal to the size of axes.");

  void* stream = const_cast<void*>(ctx.stream());
  auto* q = static_cast<sycl::queue*>(stream);

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

  std::vector<size_t> temp(numel/slice_dims.back()*2, 0);
  {
  sycl::buffer temp_buf(temp);
  sycl::buffer<size_t> index_buf(index);
  sycl::buffer<size_t> offsets_buf(offsets);
  sycl::buffer<size_t> extents_buf(extents);
  sycl::buffer<size_t> in_step_buf(in_step);

  auto e1 = q->submit([&](sycl::handler& h) {
    sycl::accessor a_index(index_buf, h, sycl::read_write);
    sycl::accessor a_temp(temp_buf, h, sycl::write_only, sycl::no_init);
    sycl::accessor a_offsets(offsets_buf, h, sycl::read_only);
    sycl::accessor a_extents(extents_buf, h, sycl::read_only);
    sycl::accessor a_in_step(in_step_buf, h, sycl::read_only);
    h.single_task([numel, a_index, a_offsets, a_extents, a_in_step,
        slice_dims_back=slice_dims.back(), a_temp, out, index_size=index.size()]()  {
      for (auto i = 0; i < numel; i += slice_dims_back) {
        auto wyn = phi::vec_product(
          &a_index[0],
          &a_in_step[0],
          index_size);
        a_temp[i/slice_dims_back*2] = i;
        a_temp[i/slice_dims_back*2+1] = wyn;

        a_index[index_size - 2]++;
        for (auto j = index_size - 2; j > 0; --j) {
          if (a_index[j] >= a_offsets[j] + a_extents[j]) {
            a_index[j] = a_offsets[j];
            a_index[j - 1] += 1;
          } else {
            break;
          }
        }
      }
    });
  });
  }

  for (auto i = 0 ; i < numel/slice_dims.back()*2; i+=2)  {
    q->submit([&](sycl::handler& h) {
      h.memcpy(out_data +  temp[i],
            in_data +  temp[i+1],
            sizeof(T) * slice_dims.back());
    });
  }
 q->wait();

  out->Resize(out_dims);
}

}  // namespace custom_kernel

PD_BUILD_PHI_KERNEL(slice,
                    intel_gpu,
                    ALL_LAYOUT,
                    custom_kernel::SliceRawKernel,
                    int64_t,
                    float,
                    double
                    ) {}
