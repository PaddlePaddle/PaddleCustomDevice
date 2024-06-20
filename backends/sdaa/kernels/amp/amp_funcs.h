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

#pragma once

#include <vector>

#include "kernels/funcs/sdaa_baseop.h"
#include "kernels/funcs/sdaa_funcs.h"
#include "kernels/profiler/sdaa_wrapper.h"
#include "paddle/phi/extension.h"

namespace custom_kernel {

enum class AbnormalValueCheck { check_nan = 0, check_inf = 1 };

namespace amp_funcs {

inline UniformOps_t GetCustomUniformOpMode(
    const AbnormalValueCheck check_type) {
  UniformOps_t mode = UNIFORM_ISNAN;
  switch (check_type) {
    case AbnormalValueCheck::check_nan:
      mode = UNIFORM_ISNAN;
      break;
    case AbnormalValueCheck::check_inf:
      mode = UNIFORM_ISINF;
      break;
    default:
      break;
  }
  return mode;
}

inline DataTypes_t ToExtendDataType(const DataType& dtype) {
  DataTypes_t dt = DATA_FLOAT;
  switch (dtype) {
    case DataType::FLOAT16:
      dt = DATA_HALF;
      break;
    case DataType::FLOAT32:
      dt = DATA_FLOAT;
      break;
    case DataType::INT8:
      dt = DATA_INT8;
      break;
    case DataType::INT16:
      dt = DATA_INT16;
      break;
    case DataType::INT32:
      dt = DATA_INT32;
      break;
    default:
      break;
  }
  return dt;
}

template <typename T, typename Context>
void AbnormCheck(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 AbnormalValueCheck check_type,
                 phi::DenseTensor* out) {
  VLOG(4) << "call custom uniform op.";

  std::vector<int64_t> x_dims = phi::vectorize<int64_t>(x.dims());

  UniformOps_t mode = GetCustomUniformOpMode(check_type);
  DataTypes_t dt = ToExtendDataType(x.dtype());

  sdaaStream_t custom_stream = GetStreamFromCTX(dev_ctx);

  TCUS_CHECK(sdcops::uniform_ops_t_ret_int(x.data(),
                                           x_dims.data(),
                                           x_dims.size(),
                                           out->data(),
                                           mode,
                                           dt,
                                           custom_stream));
}

template <typename T, typename Context>
void AddOne(const Context& dev_ctx,
            const phi::DenseTensor* in_tensor,
            phi::DenseTensor* out_tensor) {
  phi::DenseTensor factor_tensor, in_tensor_f, out_tensor_f;

  factor_tensor.Resize({1});
  dev_ctx.template Alloc<float>(&factor_tensor);
  in_tensor_f.Resize({in_tensor->numel()});
  dev_ctx.template Alloc<float>(&in_tensor_f);
  out_tensor_f.Resize({out_tensor->numel()});
  dev_ctx.template Alloc<float>(&out_tensor_f);

  sdaa_ops::doFillTensor<float>(
      dev_ctx, static_cast<float>(1.0), phi::DataType::FLOAT32, &factor_tensor);

  sdaa_ops::doCastTensor(dev_ctx, *in_tensor, &in_tensor_f);

  sdaa_ops::doElementAdd(
      dev_ctx, in_tensor_f, factor_tensor, -1, &out_tensor_f);

  sdaa_ops::doCastTensor(dev_ctx, out_tensor_f, out_tensor);
}

template <typename T, typename Context>
void AbnormCheckAndScale(const Context& dev_ctx,
                         const std::vector<const phi::DenseTensor*>& xs,
                         const phi::DenseTensor& t_scale,
                         std::vector<phi::DenseTensor*> outs,
                         phi::DenseTensor* found_inf) {
  VLOG(4) << "call sdaa custom fusedVSCheckInvalid op";

  phi::DenseTensor found_inf_INT;
  found_inf_INT.Resize(found_inf->dims());
  dev_ctx.template Alloc<int32_t>(&found_inf_INT);

  PADDLE_ENFORCE_EQ(
      t_scale.dtype(),
      phi::DataType::FLOAT32,
      phi::errors::InvalidArgument("tecodnn only support the dtype is FP32 for "
                                   "scale in check_finite_and_unscale op, "
                                   "but got [%s].",
                                   t_scale.dtype()));

  int M = xs.size();

  std::vector<int64_t> every_tensor_num;
  std::vector<T*> input(2 * M);

  u_int8_t* in_pointer;
  u_int8_t* out_pointer;

  for (int i = 0; i < M; i++) {
    int64_t tensor_num = xs[i]->numel();
    every_tensor_num.push_back(tensor_num);
    auto* x = const_cast<phi::DenseTensor*>(xs[i]);
    input[i] = x->data<T>();
    input[i + M] = outs[i]->data<T>();
  }

  phi::DenseTensor total;
  int total_numel = M * sizeof(int64_t) + 2 * M * sizeof(void*);
  total.Resize({total_numel});
  dev_ctx.template Alloc<uint8_t>(&total);
  in_pointer = total.data<uint8_t>() + M * sizeof(int64_t);
  out_pointer = in_pointer + M * sizeof(void*);
  std::vector<uint8_t> total_host(total_numel);
  // NOTE(liaotianju): Merging two vectors into one block is efficient when in
  // async view, only do one big memcpy instead of two small ones
  memcpy(total_host.data(), every_tensor_num.data(), M * sizeof(int64_t));
  memcpy(total_host.data() + M * sizeof(int64_t),
         input.data(),
         2 * M * sizeof(void*));

  AsyncMemCpyH2D(nullptr,
                 static_cast<C_Stream>(dev_ctx.stream()),
                 total.data(),
                 total_host.data(),
                 total_numel);

  DataTypes_t dt = ToExtendDataType(phi::CppTypeToDataType<T>::Type());
  sdaaStream_t custom_stream = GetStreamFromCTX(dev_ctx);
  TCUS_CHECK(sdcops::multi_t_group_ops_t_fusedVSCheckInvalid_out(
      M,
      reinterpret_cast<int64_t*>(total.data()),
      reinterpret_cast<void**>(in_pointer),
      reinterpret_cast<void**>(out_pointer),
      t_scale.data<float>(),
      found_inf->data<bool>(),
      dt,
      custom_stream));
}

}  // namespace amp_funcs
}  // namespace custom_kernel
