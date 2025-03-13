// BSD 3- Clause License Copyright (c) 2024, Tecorigin Co., Ltd. All rights
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

#include "kernels/funcs/strided_copy_utils.h"

#include <string>

#include "kernels/funcs/contiguous/contiguous_register.h"

namespace custom_kernel {

namespace sdaa_copy {

bool pair_first_down(std::pair<int64_t, int64_t> pair1,
                     std::pair<int64_t, int64_t> pair2) {
  return pair1.first > pair2.first;
}

phi::DDim permute(const phi::DDim& dims, const int64_vec& axis) {
  std::vector<int> axis_(axis.begin(), axis.end());
  int64_vec out_format_dim = phi::vectorize<int64_t>(dims.transpose(axis_));
  phi::DDim out_dim = phi::make_ddim(out_format_dim);

  return out_dim;
}

bool is_permute(const phi::DenseTensor& input) {
  if (input.meta().is_contiguous()) {
    return true;
  }
  auto input_shape = input.dims();
  auto input_strides = input.strides();
  int64_t rank = input_shape.size();
  for (size_t i = 0; i < rank; ++i) {
    if (input_strides[i] == 0) {
      return false;
    }
  }
  std::vector<std::pair<int64_t, int64_t>> strides_size(
      rank, std::pair<int64_t, int64_t>(1, 1));
  for (int64_t i = 0; i < rank; ++i) {
    strides_size[i] =
        std::pair<int64_t, int64_t>(static_cast<int64_t>(input_strides[i]),
                                    static_cast<int64_t>(input_shape[i]));
  }
  std::sort(strides_size.begin(), strides_size.end(), pair_first_down);
  bool is_permute = true;
  int64_t z = 1;
  for (int64_t i = rank - 1; i >= 0; i--) {
    auto it = strides_size[i];
    if (it.second != 1) {
      if (it.first == z) {
        z *= it.second;
      } else {
        is_permute = false;
        break;
      }
    }
  }
  return is_permute;
}

bool is_same_shapes_sizes(const phi::DenseTensor& src,
                          const phi::DenseTensor& dst) {
  if (src.dims() != dst.dims()) {
    return false;
  }
  for (int64_t i = 0; i < src.dims().size(); i++) {
    if ((src.strides()[i] != dst.strides()[i]) && src.dims()[i] != 1 &&
        dst.dims()[i] != 1) {
      return false;
    }
  }
  return true;
}

inline bool is_total_same(const phi::DenseTensor& src,
                          const phi::DenseTensor& dst) {
  return is_permute(src) && (src.dtype() == dst.dtype()) &&
         is_same_shapes_sizes(src, dst);
}

inline bool check_sdaa_align(const phi::DenseTensor& t) {
  constexpr int kSDAAAlignSize = 4;
  bool align = reinterpret_cast<int64_t>(t.data()) % kSDAAAlignSize == 0;
  if (align) {
    return true;
  }
  return false;
}

bool strided_copy(const Context& dev_ctx,
                  const phi::DenseTensor& src,
                  phi::DenseTensor* dst) {
  auto src_place_type = src.place().GetType();
  auto dst_place_type = dst->place().GetType();
  bool sdaa_place = (src_place_type == phi::AllocationType::CUSTOM) &&
                    (dst_place_type == phi::AllocationType::CUSTOM);
  PADDLE_ENFORCE_EQ(
      sdaa_place,
      true,
      phi::errors::InvalidArgument(
          "The place of source and destination tensor must be SDAA"));

  if (src.numel() == 0) {
    return true;
  }

  VLOG(1) << "strided_copy src [dims: " << src.dims()
          << ", strides: " << src.strides() << ", dtype: " << src.dtype()
          << ", data_ptr: " << src.data() << "]; "
          << "dst [dims: " << dst->dims() << ", strides: " << dst->strides()
          << ", dtype: " << dst->dtype() << ", data_ptr: " << dst->data()
          << "]";

  if (check_sdaa_align(src) && check_sdaa_align(*dst) &&
      is_total_same(src, *dst)) {
    TensorCopy(dev_ctx, src, false, dst);
    return true;
  }

  std::string maybe_opt_name;
  if (sdaa_copy::register_opt::CopyOptRegister::GetInstance()->CanOptimize(
          maybe_opt_name, dev_ctx, src, dst)) {
    if (sdaa_copy::register_opt::CopyOptRegister::GetInstance()->Run(
            maybe_opt_name, dev_ctx, src, dst)) {
      return true;
    }
  }

  return false;
}

}  // namespace sdaa_copy

}  // namespace custom_kernel
