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

class CopyStrideContiguousOpt : public ContiguousOpt {
 public:
  bool Optimize(const Context& dev_ctx,
                const phi::DenseTensor& src,
                phi::DenseTensor* dst) override {
    VLOG(1) << "SDAA use CopyStride to complete the strided_copy.";
    auto shape = phi::vectorize<int>(src.dims());
    auto src_stride = phi::vectorize<int>(src.strides());
    auto dst_stride = phi::vectorize<int>(dst->strides());

    custom_kernel::sdaa_ops::doStrideCopy(
        dev_ctx, src, shape, src_stride, dst_stride, dst);

    return true;
  }

  bool CanOptimize(const Context& dev_ctx,
                   const phi::DenseTensor& src,
                   phi::DenseTensor* dst) override {
    if (!check_CopyStride_dtype(src) || !check_CopyStride_dtype(*dst)) {
      return false;
    }

    if ((src.dtype() != dst->dtype()) || (src.dims() != dst->dims())) {
      return false;
    }

    if (dst->dims().size() > 8) {
      return false;
    }

    auto is_bad_case = [](const phi::DenseTensor& t) {
      int64_vec stride = phi::vectorize<int64_t>(t.strides());
      return std::find(stride.begin(), stride.end(), 0) != stride.end();
    };

    if (src.dims().size() == 0 || dst->dims().size() == 0 ||
        is_bad_case(*dst)) {
      return false;
    }

    return true;
  }

 private:
  bool check_CopyStride_dtype(const phi::DenseTensor& t) {
    static std::vector<phi::DataType> CopyStrideDtype = {
        phi::DataType::FLOAT64,
        phi::DataType::FLOAT32,
        phi::DataType::FLOAT16,
        phi::DataType::BFLOAT16,
        phi::DataType::INT64,
        phi::DataType::INT32,
        phi::DataType::INT16,
        phi::DataType::UINT8,
        phi::DataType::BOOL};
    phi::DataType tensor_dtype = t.dtype();

    return std::find(CopyStrideDtype.begin(),
                     CopyStrideDtype.end(),
                     tensor_dtype) != CopyStrideDtype.end();
  }
};

REGISTER_COPY_OPT_LOW(copy_stride, CopyStrideContiguousOpt);

}  // namespace sdaa_copy

}  // namespace custom_kernel
