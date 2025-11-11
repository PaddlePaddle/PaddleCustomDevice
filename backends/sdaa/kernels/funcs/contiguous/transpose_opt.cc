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

#include "kernels/funcs/contiguous/contiguous_register.h"
#include "kernels/funcs/strided_copy_utils.h"
#include "paddle/phi/extension.h"

namespace custom_kernel {

namespace sdaa_copy {

vec_tuple get_permute_back_order(const phi::DenseTensor& input) {
  int64_vec input_shapes = phi::vectorize<int64_t>(input.dims());
  int64_vec input_strides = phi::vectorize<int64_t>(input.strides());
  int64_t rank = input.dims().size();

  std::vector<std::pair<int64_t, int64_t>> strides_shapes(
      rank, std::pair<int64_t, int64_t>(1, 1));

  for (int64_t i = 0; i < rank; i++) {
    strides_shapes[i] =
        std::pair<int64_t, int64_t>(input_strides[i], input_shapes[i]);
  }

  std::sort(
      strides_shapes.begin(), strides_shapes.end(), sdaa_copy::pair_first_down);

  int64_vec permute_back_order(rank);
  int64_vec permute_order(rank);
  int64_vec raw_order(rank);

  for (int64_t i = 0; i < rank; i++) {
    auto pair = strides_shapes[i];
    raw_order[i] = i;
    for (int64_t j = 0; j < rank; j++) {
      if (pair.first == input_strides[j] && pair.second == input_shapes[j]) {
        permute_back_order[i] = j;
        permute_order[j] = i;
        input_shapes[j] = -1;
        input_strides[j] = -1;
        break;
      }
    }
  }
  return {permute_back_order, permute_order, raw_order};
}

class TransposeContiguousOpt : public ContiguousOpt {
 public:
  bool Optimize(const Context& dev_ctx,
                const phi::DenseTensor& src,
                phi::DenseTensor* dst) override {
    VLOG(1) << "SDAA use tranpose to complete the strided_copy.";
    return transpose_to_contiguous(dev_ctx, src, dst);
  }

  bool CanOptimize(const Context& dev_ctx,
                   const phi::DenseTensor& src,
                   phi::DenseTensor* dst) override {
    if (!check_Transpose_dtype(src) || !check_Transpose_dtype(*dst)) {
      return false;
    }

    // condition 1: both non_overlapping and dense.
    if (!(sdaa_copy::is_permute(src) && sdaa_copy::is_permute(*dst))) {
      return false;
    }

    int64_t rank = src.dims().size();

    // condition 2: input and output tensor has same dims and different format,
    // meanwhile dims size should be greater than 2.
    if (rank < 2 ||
        !(src.dims() == dst->dims() && src.strides() != dst->strides())) {
      return false;
    }

    // condition 3: dims size should be less than 7.
    if (rank > 7) {
      return false;
    }

    return true;
  }

 private:
  bool check_Transpose_dtype(const phi::DenseTensor& t) {
    static std::vector<phi::DataType> TransposeDtype = {phi::DataType::FLOAT64,
                                                        phi::DataType::FLOAT32,
                                                        phi::DataType::FLOAT16,
                                                        phi::DataType::BFLOAT16,
                                                        phi::DataType::INT64,
                                                        phi::DataType::INT32,
                                                        phi::DataType::INT16,
                                                        phi::DataType::INT8,
                                                        phi::DataType::UINT8,
                                                        phi::DataType::BOOL};
    phi::DataType tensor_dtype = t.dtype();

    return std::find(TransposeDtype.begin(),
                     TransposeDtype.end(),
                     tensor_dtype) != TransposeDtype.end();
  }

  bool transpose_to_contiguous(const Context& dev_ctx,
                               const phi::DenseTensor& src,
                               phi::DenseTensor* dst) {
    // convert a non_overlapping_and_dense tensor to contiguous tensor.
    auto recover_contiguous =
        [](const Context& dev_ctx,
           const phi::DenseTensor& t,
           std::vector<int64_t>* permute_order) -> phi::DenseTensor {
      phi::DenseTensor view_contiguous;

      if (!t.meta().is_contiguous()) {
        vec_tuple order_info = sdaa_copy::get_permute_back_order(t);
        *permute_order = std::get<1>(order_info);

        phi::DDim new_dim =
            sdaa_copy::permute(t.dims(), std::get<0>(order_info));
        phi::DDim new_stride =
            sdaa_copy::permute(t.strides(), std::get<0>(order_info));
        view_contiguous = t;
        phi::DenseTensorMeta meta = t.meta();
        meta.dims = new_dim;
        meta.strides = new_stride;
        view_contiguous.set_meta(meta);
      } else {
        view_contiguous = t;
        std::iota((*permute_order).begin(), (*permute_order).end(), 0);
      }
      return view_contiguous;
    };

    auto src_rank = src.dims().size();
    auto dst_rank = dst->dims().size();
    int64_vec src_permute_order(src_rank);
    int64_vec dst_permute_order(dst_rank);

    phi::DenseTensor src_view_contiguous =
        recover_contiguous(dev_ctx, src, &src_permute_order);
    phi::DenseTensor dst_view_contiguous =
        recover_contiguous(dev_ctx, *dst, &dst_permute_order);

    std::vector<int> new_permute_order(dst_rank);
    for (int i = 0; i < dst_rank; i++) {
      new_permute_order[dst_permute_order[i]] = src_permute_order[i];
    }

    custom_kernel::sdaa_ops::doTransposeTensor(
        dev_ctx, src_view_contiguous, new_permute_order, &dst_view_contiguous);
    return true;
  }
};

REGISTER_COPY_OPT(transpose, TransposeContiguousOpt);

}  // namespace sdaa_copy

}  // namespace custom_kernel
