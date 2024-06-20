// BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
// reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
// Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software
// without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
// WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
// OF SUCH DAMAGE.

#include <iostream>

#include "kernels/funcs/sdaa_baseop.h"
#include "paddle/phi/extension.h"
#include "paddle/phi/kernels/funcs/slice_utils.h"

PHI_DECLARE_bool(set_to_1d);

namespace custom_kernel {

void UpdateAttr(const phi::DDim& in_dims,
                const std::vector<int> axes,
                const std::vector<int> starts,
                const std::vector<int> ends,
                std::vector<int>* offsets,
                std::vector<int>* size) {
  int cnt = 0;
  for (int i = 0; i < in_dims.size(); ++i) {
    int start = 0;
    int end = in_dims[i];
    // NOTE(zhiqiu): Becareful that cnt may > axes.size() and result in
    // overflow.
    int axis = cnt < static_cast<int>(axes.size()) ? axes[cnt] : -1;
    if (axis == i) {
      start = starts[cnt];
      if (start < 0) {
        start = (start + in_dims[i]);
      }
      start = std::max(start, static_cast<int>(0));
      end = ends[cnt];
      if (end < 0) {
        end = (end + in_dims[i]);
      }
      end = std::min(end, static_cast<int>(in_dims[i]));
      cnt++;
    }

    (*offsets)[i] = start;
    (*size)[i] = end - start;
  }
}

template <typename T = int64_t>
inline phi::DDim GetDecreasedDims(const phi::DDim slice_dims,
                                  const std::vector<T>& decrease_axes,
                                  std::vector<T>* infer_flags = nullptr) {
  phi::DDim decreased_dims(slice_dims);
  std::vector<uint8_t> decrease_flag(slice_dims.size(), 0);
  if (decrease_axes.size() > 0) {
    for (size_t i = 0; i < decrease_axes.size(); ++i) {
      T axis = decrease_axes[i];
      decrease_flag[axis] = 1;
      if (infer_flags && (*infer_flags)[i] != -1) {
        PADDLE_ENFORCE_EQ(decreased_dims[axis],
                          1,
                          phi::errors::InvalidArgument(
                              "Decrease dim should be 1, but now received %d",
                              decreased_dims[axis]));
      }
    }

    std::vector<T> new_shape;
    for (int i = 0; i < decreased_dims.size(); ++i) {
      if (decrease_flag[i] == 0) {
        new_shape.push_back(decreased_dims[i]);
      }
    }

    // NOTE(zoooo0820): Hack procssing to 1-D, when axes decrease to 0-D in
    // slice. This will remove in release 2.6.
    if (isEnvEnable("FLAGS_set_to_1d") && new_shape.size() == 0) {
      new_shape.push_back(1);
    }

    decreased_dims = phi::make_ddim(new_shape);
  }
  return decreased_dims;
}

template <typename T, typename Context>
void SliceRawKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const std::vector<int64_t>& axes,
                    const phi::IntArray& starts_array,
                    const phi::IntArray& ends_array,
                    const std::vector<int64_t>& infer_flags,
                    const std::vector<int64_t>& decrease_axis,
                    phi::DenseTensor* out) {
  VLOG(4) << "Call SDAA SliceRawKernel";

  auto starts = starts_array.GetData();
  auto ends = ends_array.GetData();
  std::vector<int64_t> strides(axes.size(), 1);

  PADDLE_ENFORCE_EQ(
      starts.size(),
      axes.size(),
      phi::errors::InvalidArgument(
          "The size of starts must be equal to the size of axes."));
  PADDLE_ENFORCE_EQ(ends.size(),
                    axes.size(),
                    phi::errors::InvalidArgument(
                        "The size of ends must be equal to the size of axes."));

  // Infer output dims
  const auto& in_dims = x.dims();
  auto out_dims = out->dims();
  auto slice_dims = out_dims;

  for (size_t i = 0; i < axes.size(); ++i) {
    // when start == -1 && end == start + 1
    if (starts[i] == -1 && ends[i] == 0 && infer_flags[i] == -1) {
      auto ret = std::find(decrease_axis.begin(), decrease_axis.end(), axes[i]);
      if (ret != decrease_axis.end()) {
        ends[i] = in_dims[axes[i]];
      }
    }
  }

  phi::funcs::UpdateSliceAttrs(in_dims, axes, &starts, &ends);
  slice_dims = phi::funcs::GetSliceDims<int64_t>(
      in_dims, axes, starts, ends, nullptr, nullptr);
  out_dims = custom_kernel::GetDecreasedDims(slice_dims, decrease_axis);
  out->Resize(out_dims);

  dev_ctx.template Alloc<T>(out);

  sdaa_ops::doSliceTensor(
      dev_ctx, x, axes, starts, ends, strides, decrease_axis, out);
}

template <typename T, typename Context>
void SliceGradRawKernel(const Context& dev_ctx,
                        const phi::DenseTensor& x,
                        const phi::DenseTensor& dout,
                        const std::vector<int64_t>& axes_t,
                        const phi::IntArray& starts_array,
                        const phi::IntArray& ends_array,
                        const std::vector<int64_t>& infer_flags,
                        const std::vector<int64_t>& decrease_axis,
                        phi::DenseTensor* dx) {
  VLOG(4) << "Call SDAA SliceGradRawKernel";

  std::vector<int> axes(axes_t.begin(), axes_t.end());
  auto starts_int = starts_array.GetData();
  auto ends_int = ends_array.GetData();

  std::vector<int> starts(starts_int.begin(), starts_int.end());
  std::vector<int> ends(ends_int.begin(), ends_int.end());

  const auto& in_dims = x.dims();
  int rank = in_dims.size();

  std::vector<int> offset(rank);
  std::vector<int> size(rank);
  custom_kernel::UpdateAttr(in_dims,
                            std::move(axes),
                            std::move(starts),
                            std::move(ends),
                            &offset,
                            &size);

  // paddings[0][i] represent the padding size before slice start position,
  // paddings[1][i] represent the padding size after slice end position
  std::vector<std::vector<int>> paddings(2, std::vector<int>(rank));
  for (int i = 0; i < rank; ++i) {
    paddings[0][i] = offset[i];
    paddings[1][i] = in_dims[i] - size[i] - offset[i];
  }

  phi::DenseTensor tmp_dout(dout);
  auto out_dims = dout.dims();

  auto decrease_size = decrease_axis.size();
  if (decrease_size > 0) {
    if (decrease_size == static_cast<size_t>(in_dims.size())) {
      out_dims = phi::make_ddim(std::vector<int>(decrease_size, 1));
    } else {
      std::vector<int> origin_out_shape(out_dims.size() + decrease_size, -1);
      for (size_t i = 0; i < decrease_size; ++i) {
        origin_out_shape[decrease_axis[i]] = 1;
      }
      int index = 0;
      for (size_t i = 0; i < origin_out_shape.size(); ++i) {
        if (origin_out_shape[i] == -1) {
          origin_out_shape[i] = out_dims[index];
          ++index;
        }
      }
      out_dims = phi::make_ddim(origin_out_shape);
    }
    tmp_dout.Resize(out_dims);
  }

  dev_ctx.template Alloc<T>(dx);
  sdaa_ops::doPaddingTensor(dev_ctx, tmp_dout, paddings, dx);
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(slice,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::SliceRawKernel,
                          bool,
                          float,
                          phi::dtype::float16,
                          double,
                          int32_t,
                          int64_t) {}

PD_REGISTER_PLUGIN_KERNEL(slice_grad,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::SliceGradRawKernel,
                          float,
                          phi::dtype::float16,
                          int32_t) {}
