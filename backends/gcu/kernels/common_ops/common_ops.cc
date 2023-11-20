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

#include "kernels/common_ops/common_ops.h"

#include <limits>
#include <map>

#include "backend/utils/utils.h"
#include "common/common.h"
#include "kernels/funcs/gcu_funcs.h"
#include "kernels/funcs/gcu_op_runner.h"
#include "paddle/phi/kernels/funcs/concat_funcs.h"

namespace custom_kernel {

phi::DenseTensor ConvertLayout(const phi::CustomContext& dev_ctx,
                               const phi::DenseTensor& tensor,
                               phi::DataLayout layout) {
  auto src_layout = LayoutToVector(tensor.layout());
  auto dst_layout = LayoutToVector(layout);
  auto tensor_dims = phi::vectorize(tensor.dims());
  std::vector<int64_t> out_permute_dims;
  std::vector<int64_t> out_convert_dims;
  LayoutConvertDims(
      tensor_dims, src_layout, dst_layout, out_permute_dims, out_convert_dims);

  phi::DenseTensor output_tensor;
  phi::DenseTensorMeta meta(
      tensor.dtype(), phi::make_ddim(out_convert_dims), layout);
  output_tensor.set_meta(meta);
  dev_ctx.Alloc(&output_tensor, tensor.dtype());

  transpose(dev_ctx, tensor, output_tensor, out_permute_dims);
  return output_tensor;
}

phi::DenseTensor ConvertNCHWToNHWC(const phi::CustomContext& dev_ctx,
                                   const phi::DenseTensor& tensor) {
  PADDLE_ENFORCE_EQ(
      tensor.layout(),
      phi::DataLayout::NCHW,
      phi::errors::InvalidArgument("the tensor layout is %s, not NCHW ",
                                   phi::DataLayoutToString(tensor.layout())));
  return ConvertLayout(dev_ctx, tensor, phi::DataLayout::NHWC);
}

phi::DenseTensor ConvertNHWCToNCHW(const phi::CustomContext& dev_ctx,
                                   const phi::DenseTensor& tensor) {
  PADDLE_ENFORCE_EQ(
      tensor.layout(),
      phi::DataLayout::NHWC,
      phi::errors::InvalidArgument("the tensor layout is %s, not NHWC ",
                                   phi::DataLayoutToString(tensor.layout())));
  return ConvertLayout(dev_ctx, tensor, phi::DataLayout::NCHW);
}

void transpose(const phi::CustomContext& dev_ctx,
               const phi::DenseTensor& src_tensor,
               phi::DenseTensor& dst_tensor,  // NOLINT
               const std::vector<int64_t>& permutation) {
  if (dst_tensor.numel() > 0) {
    auto tensor_gcu = GetHlirTensor(src_tensor);
    auto out_gcu = GetHlirTensor(dst_tensor);
    hlir::DispatchParam params;
    params.inputs = {tensor_gcu};
    params.outputs = {out_gcu};
    auto hlir_dims = hlir::ShapeMetaData<int64_t>(
        permutation, {static_cast<int64_t>(permutation.size())});
    params.metadata.setValue("permutation", hlir_dims);
    params.stream = static_cast<topsStream_t>(dev_ctx.stream());
    AOTOPS_DEBUG("transpose", params);
    GCUOPS_TRACE_START(transpose);
    auto func_ptr = GetOpFuncPtr(kTransPose, params);
    if (func_ptr) {
      auto pass = hlir::HlirDispatch::dispatch(func_ptr, params);
      PADDLE_ENFORCE(
          pass,
          phi::errors::InvalidArgument("dispatch %s failed!", kTransPose));
    } else {
      PADDLE_ENFORCE(
          false,
          phi::errors::InvalidArgument("not find aot func for %s", kTransPose));
    }
    FreeDispatchParam(params);
    GCUOPS_TRACE_END(transpose);
    GcuOpStreamSync(dev_ctx);
  }
}

phi::DenseTensor transpose(const phi::CustomContext& dev_ctx,
                           const phi::DenseTensor& src_tensor,
                           const std::vector<int64_t>& permutation) {
  // infer dst shape
  std::vector<int64_t> src_dims = phi::vectorize(src_tensor.dims());
  std::vector<int64_t> dst_dims = reorder_vector(src_dims, permutation);

  phi::DenseTensor dst_tensor;
  phi::DenseTensorMeta meta(src_tensor.dtype(), phi::make_ddim(dst_dims));
  dst_tensor.set_meta(meta);
  dev_ctx.Alloc(&dst_tensor, dst_tensor.dtype());

  transpose(dev_ctx, src_tensor, dst_tensor, permutation);
  return dst_tensor;
}

void dot_general_common(const phi::CustomContext& dev_ctx,
                        const phi::DenseTensor& lhs,
                        const phi::DenseTensor& rhs,
                        phi::DenseTensor& out,  // NOLINT
                        const std::vector<int64_t>& lhs_batch_dimension,
                        const std::vector<int64_t>& rhs_batch_dimension,
                        const std::vector<int64_t>& lhs_contracting_dimension,
                        const std::vector<int64_t>& rhs_contracting_dimension,
                        const double& alpha,
                        const double& beta) {
  auto lhs_gcu = GetHlirTensor(lhs);
  auto rhs_gcu = GetHlirTensor(rhs);
  auto out_gcu = GetHlirTensor(out);
  hlir::DispatchParam params;
  params.inputs = {lhs_gcu, rhs_gcu};
  params.outputs = {out_gcu};
  params.metadata.setValue("lhs_contracting_dimension",
                           HlirVector(lhs_contracting_dimension));
  params.metadata.setValue("rhs_contracting_dimension",
                           HlirVector(rhs_contracting_dimension));
  params.metadata.setValue("lhs_batch_dimension",
                           HlirVector(lhs_batch_dimension));
  params.metadata.setValue("rhs_batch_dimension",
                           HlirVector(rhs_batch_dimension));
  params.metadata.setValue(hlir::kAlpha, alpha);
  params.metadata.setValue(hlir::kBelta, beta);
  params.stream = static_cast<topsStream_t>(dev_ctx.stream());
  AOTOPS_DEBUG(kDotGeneral, params);
  GCUOPS_TRACE_START(dot_general);
  auto func_ptr = GetOpFuncPtr(kDotGeneral, params);
  if (func_ptr) {
    auto pass = hlir::HlirDispatch::dispatch(func_ptr, params);
    PADDLE_ENFORCE(
        pass, phi::errors::InvalidArgument("dispatch %s failed!", kDotGeneral));
  } else {
    PADDLE_ENFORCE(
        false,
        phi::errors::InvalidArgument("not find aot func for %s", kDotGeneral));
  }
  FreeDispatchParam(params);
  GCUOPS_TRACE_END(dot_general);
  GcuOpStreamSync(dev_ctx);
}

phi::DenseTensor dot_general_common(
    const phi::CustomContext& dev_ctx,
    const phi::DenseTensor& lhs,
    const phi::DenseTensor& rhs,
    const std::vector<int64_t>& lhs_batch_dimension,
    const std::vector<int64_t>& rhs_batch_dimension,
    const std::vector<int64_t>& lhs_contracting_dimension,
    const std::vector<int64_t>& rhs_contracting_dimension,
    const double& alpha,
    const double& beta) {
  // infer out shape
  std::vector<int64_t> lhs_dims = phi::vectorize(lhs.dims());
  std::vector<int64_t> rhs_dims = phi::vectorize(rhs.dims());
  std::vector<int64_t> out_dims;

  PADDLE_ENFORCE_EQ(
      lhs_batch_dimension.size(),
      rhs_batch_dimension.size(),
      phi::errors::InvalidArgument(
          "Must specify the same number of batch dimensions for lhs and rhs."));

  for (auto i = 0; i < lhs_batch_dimension.size(); ++i) {
    PADDLE_ENFORCE_EQ(
        lhs_dims[lhs_batch_dimension[i]],
        rhs_dims[rhs_batch_dimension[i]],
        phi::errors::InvalidArgument("Batch dimension sizes do not match."));
    out_dims.push_back(lhs_dims[lhs_batch_dimension[i]]);
  }
  // Ms
  for (int64_t i = 0; i < lhs_dims.size(); ++i) {
    if (!vector_contains(lhs_batch_dimension, i) &&
        !vector_contains(lhs_contracting_dimension, i))
      out_dims.push_back(lhs_dims[i]);
  }
  // Ns
  for (int64_t i = 0; i < rhs_dims.size(); ++i) {
    if (!vector_contains(rhs_batch_dimension, i) &&
        !vector_contains(rhs_contracting_dimension, i))
      out_dims.push_back(rhs_dims[i]);
  }

  phi::DenseTensor out;
  phi::DenseTensorMeta meta(lhs.dtype(), phi::make_ddim(out_dims));
  out.set_meta(meta);
  dev_ctx.Alloc(&out, out.dtype());

  dot_general_common(dev_ctx,
                     lhs,
                     rhs,
                     out,
                     lhs_batch_dimension,
                     rhs_batch_dimension,
                     lhs_contracting_dimension,
                     rhs_contracting_dimension,
                     alpha,
                     beta);
  return out;
}

void dot_common(const phi::CustomContext& dev_ctx,
                const phi::DenseTensor& lhs,
                const phi::DenseTensor& rhs,
                phi::DenseTensor& out,  // NOLINT
                const double& alpha,
                const double& beta) {
  auto lhs_gcu = GetHlirTensor(lhs);
  auto rhs_gcu = GetHlirTensor(rhs);
  auto out_gcu = GetHlirTensor(out);
  hlir::DispatchParam params;
  params.inputs = {lhs_gcu, rhs_gcu};
  params.outputs = {out_gcu};
  params.metadata.setValue(hlir::kAlpha, alpha);
  params.metadata.setValue(hlir::kBelta, beta);
  params.stream = static_cast<topsStream_t>(dev_ctx.stream());
  AOTOPS_DEBUG(kDot, params);
  GCUOPS_TRACE_START(dot);
  auto func_ptr = GetOpFuncPtr(kDot, params);
  if (func_ptr) {
    auto pass = hlir::HlirDispatch::dispatch(func_ptr, params);
    PADDLE_ENFORCE(pass,
                   phi::errors::InvalidArgument("dispatch %s failed!", kDot));
  } else {
    PADDLE_ENFORCE(
        false, phi::errors::InvalidArgument("not find aot func for %s", kDot));
  }
  FreeDispatchParam(params);
  GCUOPS_TRACE_END(dot);
  GcuOpStreamSync(dev_ctx);
}

void concat(const phi::CustomContext& dev_ctx,
            const std::vector<phi::DenseTensor>& input_tensors,
            int64_t axis,
            phi::DenseTensor& output) {  // NOLINT
  int rank = input_tensors.at(0).dims().size();
  PADDLE_ENFORCE_EQ(
      axis >= -rank && axis < rank,
      true,
      phi::errors::InvalidArgument(
          "The axis is expected to be in range of [%d, %d), but got %d",
          -rank,
          rank,
          axis));
  if (axis < 0) {
    axis = axis + rank;
  }

  auto input_num = input_tensors.size();
  std::vector<hlir::Tensor*> tensors_gcu;
  for (size_t idx = 0; idx < input_num; idx++)
    tensors_gcu.push_back(GetHlirTensor(input_tensors[idx]));
  auto out_gcu = GetHlirTensor(output);
  hlir::DispatchParam params;
  for (size_t idx = 0; idx < input_num; idx++)
    params.inputs.push_back(tensors_gcu[idx]);
  params.metadata.setValue("dimension", axis);
  params.metadata.setValue("kInputTensorNum", int32_t(input_num));
  params.metadata.setValue("kOutputTensorNum", int32_t(1));
  params.outputs = {out_gcu};
  params.stream = static_cast<topsStream_t>(dev_ctx.stream());
  AOTOPS_DEBUG(KConcat, params);
  GCUOPS_TRACE_START(concat);
  auto func_ptr = GetOpFuncPtr(KConcat, params);
  if (func_ptr) {
    auto pass = hlir::HlirDispatch::dispatch(func_ptr, params);
    PADDLE_ENFORCE(
        pass, phi::errors::InvalidArgument("dispatch %s failed!", KConcat));
  } else {
    PADDLE_ENFORCE(
        false,
        phi::errors::InvalidArgument("not find aot func for %s", KConcat));
  }
  FreeDispatchParam(params);
  GCUOPS_TRACE_END(concat);
  GcuOpStreamSync(dev_ctx);
}

// no check, please do not use it directly
phi::DenseTensor concat(const phi::CustomContext& dev_ctx,
                        const std::vector<phi::DenseTensor>& input_tensors,
                        int64_t axis) {  // NOLINT
  phi::DenseTensor output;
  output.set_meta(input_tensors.at(0).meta());

  int rank = input_tensors.at(0).dims().size();
  PADDLE_ENFORCE_EQ(
      axis >= -rank && axis < rank,
      true,
      phi::errors::InvalidArgument(
          "The axis is expected to be in range of [%d, %d), but got %d",
          -rank,
          rank,
          axis));
  if (axis < 0) {
    axis = axis + rank;
  }

  // calculate out dims
  std::vector<phi::DDim> x_dims;
  x_dims.reserve(input_tensors.size());
  for (auto& x_t : input_tensors) {
    x_dims.emplace_back(x_t.dims());
  }
  phi::DDim out_dim = phi::funcs::ComputeAndCheckShape(true, x_dims, axis);

  output.Resize(out_dim);
  dev_ctx.Alloc(&output, output.dtype());

  concat(dev_ctx, input_tensors, axis, output);

  return output;
}

std::vector<int64_t> infer_broadcast_dim_map(const std::vector<int64_t>& a,
                                             const std::vector<int64_t>& b) {
  auto dims_a = a.size();
  auto ndim = b.size();
  PADDLE_ENFORCE(dims_a <= ndim,
                 phi::errors::InvalidArgument("%d vs %d", dims_a, ndim));
  if (dims_a == 0) {
    return ndim == 0 ? std::vector<int64_t>{0}
                     : std::vector<int64_t>{static_cast<int64_t>(ndim - 1)};
  }
  std::vector<int64_t> dim_maps(dims_a);
  for (int64_t i = dims_a - 1; i >= 0; --i) {
    int64_t offset = dims_a - 1 - i;
    int64_t dim_b = ndim - 1 - offset;
    int64_t size_a = a[i];
    int64_t size_b = b[dim_b];
    PADDLE_ENFORCE(
        size_a == size_b || size_a == 1,
        phi::errors::InvalidArgument("The size of tensor a (%d"
                                     ") must match the size of tensor b (%d"
                                     ") at non-singleton dimension %d",
                                     size_a,
                                     size_b,
                                     i));
    dim_maps[i] = dim_b;
  }
  return dim_maps;
}

phi::DenseTensor& broadcast(const phi::CustomContext& dev_ctx,
                            const phi::DenseTensor& src,
                            phi::DenseTensor& dst) {  // NOLINT
  auto src_sizes = phi::vectorize(src.dims());
  auto dst_sizes = phi::vectorize(dst.dims());
  auto dim_map = infer_broadcast_dim_map(src_sizes, dst_sizes);
  if (src.numel() > 0) {
    auto src_gcu = GetHlirTensor(src);
    auto out_gcu = GetHlirTensor(dst);
    hlir::DispatchParam params;
    params.inputs = {src_gcu};
    params.outputs = {out_gcu};
    params.metadata.setValue("dim_map", HlirVector(dim_map));
    params.metadata.setValue(hlir::kAlpha, 1.0);
    params.metadata.setValue(hlir::kBelta, 0.0);
    params.stream = static_cast<topsStream_t>(dev_ctx.stream());
    AOTOPS_DEBUG(kBroadcast, params);
    GCUOPS_TRACE_START(broadcast);
    auto func_ptr = GetOpFuncPtr(kBroadcast, params);
    if (func_ptr) {
      auto pass = hlir::HlirDispatch::dispatch(func_ptr, params);
      PADDLE_ENFORCE(
          pass,
          phi::errors::InvalidArgument("dispatch %s failed!", kBroadcast));
    } else {
      PADDLE_ENFORCE(
          false,
          phi::errors::InvalidArgument("not find aot func for %s", kBroadcast));
    }
    FreeDispatchParam(params);
    GcuOpStreamSync(dev_ctx);
    GCUOPS_TRACE_END(broadcast);
  } else {
    VLOG(1) << "broadcast(src, dst): src numel = " << src.numel()
            << ", dst numel = " << dst.numel();
  }

  return dst;
}

phi::DenseTensor& fill(const phi::CustomContext& dev_ctx,
                       phi::DenseTensor& dims,  // NOLINT
                       const phi::DenseTensor& value) {
  VLOG(1) << "fill value.dims() " << (value.dims().size()) << " dims.numel() "
          << dims.numel();
  PADDLE_ENFORCE(
      value.dims().size() == 0 || value.dims() == phi::make_ddim({1}),
      phi::errors::InvalidArgument(
          "fill only supports 0-dimension value tensor but got tensor with "
          "%d dimensions.",
          value.dims().size()));
  if (dims.numel() > 0) {
    return broadcast(dev_ctx, value, dims);
  } else {
    return dims;
  }
}

phi::DenseTensor& slice(const phi::CustomContext& dev_ctx,
                        const phi::DenseTensor& input,
                        const std::vector<int64_t>& axes,
                        const std::vector<int64_t>& starts,
                        const std::vector<int64_t>& ends,
                        const std::vector<int64_t>& steps,
                        phi::DenseTensor& output) {  // NOLINT
  if (input.capacity() > 0 && output.capacity() > 0) {
    auto input_gcu = GetHlirTensor(input);
    auto out_gcu = GetHlirTensor(output);
    hlir::DispatchParam params;
    params.inputs = {input_gcu};
    params.outputs = {out_gcu};
    params.metadata.setValue("axes", HlirVector(axes));
    params.metadata.setValue("starts", HlirVector(starts));
    params.metadata.setValue("ends", HlirVector(ends));
    params.metadata.setValue("steps", HlirVector(steps));
    params.stream = static_cast<topsStream_t>(dev_ctx.stream());
    AOTOPS_DEBUG(kSlice, params);
    GCUOPS_TRACE_START(slice);
    auto func_ptr = GetOpFuncPtr(kSlice, params);
    if (func_ptr) {
      auto pass = hlir::HlirDispatch::dispatch(func_ptr, params);
      PADDLE_ENFORCE(
          pass, phi::errors::InvalidArgument("dispatch %s failed!", kSlice));
    } else {
      PADDLE_ENFORCE(
          false,
          phi::errors::InvalidArgument("not find aot func for %s", kSlice));
    }
    FreeDispatchParam(params);
    GcuOpStreamSync(dev_ctx);
    GCUOPS_TRACE_END(slice);
  }

  return output;
}

phi::DenseTensor reverse(const phi::CustomContext& dev_ctx,
                         const phi::DenseTensor& input,
                         const std::vector<int64_t>& reverse_dims) {  // NOLINT
  phi::DenseTensor output = EmptyTensor(dev_ctx, input.meta());
  if (input.capacity() > 0 && output.capacity() > 0) {
    auto input_gcu = GetHlirTensor(input);
    auto out_gcu = GetHlirTensor(output);
    hlir::DispatchParam params;
    params.inputs = {input_gcu};
    params.outputs = {out_gcu};
    params.metadata.setValue("dims", HlirVector(reverse_dims));
    params.stream = static_cast<topsStream_t>(dev_ctx.stream());
    AOTOPS_DEBUG(kReverse, params);
    GCUOPS_TRACE_START(reverse);
    auto func_ptr = GetOpFuncPtr(kReverse, params);
    if (func_ptr) {
      auto pass = hlir::HlirDispatch::dispatch(func_ptr, params);
      PADDLE_ENFORCE(
          pass, phi::errors::InvalidArgument("dispatch %s failed!", kReverse));
    } else {
      PADDLE_ENFORCE(
          false,
          phi::errors::InvalidArgument("not find aot func for %s", kReverse));
    }
    FreeDispatchParam(params);
    GCUOPS_TRACE_END(reverse);
    GcuOpStreamSync(dev_ctx);
  }

  return output;
}

#define CREATE_SCALAR_CASE(SAVER, DTAT_TYPE, VAL)                           \
  case DTAT_TYPE: {                                                         \
    typedef typename ::phi::DataTypeToCppType<DTAT_TYPE>::type T;           \
    SAVER[DTAT_TYPE] = CreateScalarTensor<T>(dev_ctx, static_cast<T>(VAL)); \
    break;                                                                  \
  }

#define CREATE_SCALAR_ALL_TYPE(SAVER, VAL)                         \
  if (SAVER.count(dtype) <= 0) {                                   \
    switch (dtype) {                                               \
      CREATE_SCALAR_CASE(SAVER, phi::DataType::INT8, VAL)          \
      CREATE_SCALAR_CASE(SAVER, phi::DataType::INT16, VAL)         \
      CREATE_SCALAR_CASE(SAVER, phi::DataType::INT32, VAL)         \
      CREATE_SCALAR_CASE(SAVER, phi::DataType::INT64, VAL)         \
      CREATE_SCALAR_CASE(SAVER, phi::DataType::UINT8, VAL)         \
      CREATE_SCALAR_CASE(SAVER, phi::DataType::FLOAT16, VAL)       \
      CREATE_SCALAR_CASE(SAVER, phi::DataType::BFLOAT16, VAL)      \
      CREATE_SCALAR_CASE(SAVER, phi::DataType::FLOAT32, VAL)       \
      CREATE_SCALAR_CASE(SAVER, phi::DataType::FLOAT64, VAL)       \
      CREATE_SCALAR_CASE(SAVER, phi::DataType::BOOL, VAL)          \
      default: {                                                   \
        PADDLE_ENFORCE(false,                                      \
                       phi::errors::InvalidArgument(               \
                           "Invalid scalar type %s",               \
                           phi::DataTypeToString(dtype).c_str())); \
      }                                                            \
    }                                                              \
  }

phi::DenseTensor& zeros(const phi::CustomContext& dev_ctx,
                        phi::DenseTensor& src) {  // NOLINT
  static std::map<phi::DataType, phi::DenseTensor> all_zeros;
  auto dtype = src.dtype();
  CREATE_SCALAR_ALL_TYPE(all_zeros, 0)
  return fill(dev_ctx, src, all_zeros.at(dtype));
}

phi::DenseTensor& ones(const phi::CustomContext& dev_ctx,
                       phi::DenseTensor& src) {  // NOLINT
  static std::map<phi::DataType, phi::DenseTensor> all_ones;
  auto dtype = src.dtype();
  CREATE_SCALAR_ALL_TYPE(all_ones, 1)
  return fill(dev_ctx, src, all_ones.at(dtype));
}

phi::DenseTensor zeros_like(const phi::CustomContext& dev_ctx,
                            const phi::DenseTensor& src) {
  auto dst = EmptyTensor(dev_ctx, src.meta());
  zeros(dev_ctx, dst);
  return dst;
}

phi::DenseTensor ones_like(const phi::CustomContext& dev_ctx,
                           const phi::DenseTensor& src) {
  auto dst = EmptyTensor(dev_ctx, src.meta());
  ones(dev_ctx, dst);
  return dst;
}

#undef CREATE_SCALAR_ALL_TYPE
#undef CREATE_SCALAR_CASE

phi::DenseTensor& neg_infs(const phi::CustomContext& dev_ctx,
                           phi::DenseTensor& src) {  // NOLINT
  static std::map<phi::DataType, phi::DenseTensor> all_infs;
  auto dtype = src.dtype();

#define CREATE_INFINITY_CASE(SAVER, DTAT_TYPE)                               \
  case DTAT_TYPE: {                                                          \
    typedef typename ::phi::DataTypeToCppType<DTAT_TYPE>::type T;            \
    SAVER[DTAT_TYPE] =                                                       \
        CreateScalarTensor<T>(dev_ctx, -std::numeric_limits<T>::infinity()); \
    break;                                                                   \
  }

  if (all_infs.count(dtype) <= 0) {
    switch (dtype) {
      CREATE_INFINITY_CASE(all_infs, phi::DataType::INT8)
      CREATE_INFINITY_CASE(all_infs, phi::DataType::INT16)
      CREATE_INFINITY_CASE(all_infs, phi::DataType::INT32)
      CREATE_INFINITY_CASE(all_infs, phi::DataType::INT64)
      CREATE_INFINITY_CASE(all_infs, phi::DataType::UINT8)
      CREATE_INFINITY_CASE(all_infs, phi::DataType::FLOAT16)
      CREATE_INFINITY_CASE(all_infs, phi::DataType::BFLOAT16)
      CREATE_INFINITY_CASE(all_infs, phi::DataType::FLOAT32)
      CREATE_INFINITY_CASE(all_infs, phi::DataType::FLOAT64)
      CREATE_INFINITY_CASE(all_infs, phi::DataType::BOOL)
      default: {
        PADDLE_ENFORCE(
            false,
            phi::errors::InvalidArgument("Invalid scalar type %s",
                                         phi::DataTypeToString(dtype).c_str()));
      }
    }
  }

#undef CREATE_INFINITY_CASE

  return fill(dev_ctx, src, all_infs.at(dtype));
}

phi::DenseTensor broadcast_in_dim(const phi::CustomContext& dev_ctx,
                                  const phi::DenseTensor& src,
                                  std::vector<int64_t> output_dims,
                                  std::vector<int64_t> broadcast_dimensions) {
  phi::DenseTensor dst =
      EmptyTensor(dev_ctx, src.dtype(), phi::make_ddim(output_dims));
  if (src.numel() > 0) {
    auto src_gcu = GetHlirTensor(src);
    auto out_gcu = GetHlirTensor(dst);
    hlir::DispatchParam params;
    params.inputs = {src_gcu};
    params.outputs = {out_gcu};
    params.metadata.setValue(
        "broadcast_dimensions",
        hlir::ShapeMetaData<int64_t>(
            broadcast_dimensions,
            {static_cast<int64_t>(broadcast_dimensions.size())}));
    params.stream = static_cast<topsStream_t>(dev_ctx.stream());
    AOTOPS_DEBUG(kBroadcastInDim, params);
    GCUOPS_TRACE_START(broadcast_in_dim);
    auto func_ptr = GetOpFuncPtr(kBroadcastInDim, params);
    if (func_ptr) {
      auto pass = hlir::HlirDispatch::dispatch(func_ptr, params);
      PADDLE_ENFORCE(
          pass,
          phi::errors::InvalidArgument("dispatch %s failed!", kBroadcastInDim));
    } else {
      PADDLE_ENFORCE(false,
                     phi::errors::InvalidArgument("not find aot func for %s",
                                                  kBroadcastInDim));
    }
    FreeDispatchParam(params);
    GcuOpStreamSync(dev_ctx);
    GCUOPS_TRACE_END(broadcast_in_dim);
  } else {
    VLOG(1) << "broadcast(src, dst): src numel = " << src.numel()
            << ", dst numel = " << dst.numel();
  }

  return dst;
}

phi::DenseTensor broadcast_to(const phi::CustomContext& dev_ctx,
                              const phi::DenseTensor& src,
                              std::vector<int64_t> output_dims) {
  auto input_dims = phi::vectorize(src.dims());

  PADDLE_ENFORCE_LE(
      input_dims.size(),
      output_dims.size(),
      phi::errors::InvalidArgument(
          "Input shape %s"
          " must have rank less than or equal to the output shape %s",
          backend::VectorToString(input_dims).c_str(),
          backend::VectorToString(output_dims).c_str()));

  std::vector<int64_t> broadcast_dims;
  std::vector<int64_t> broadcast_shape;
  auto input_it = input_dims.rbegin();
  for (auto output_it = output_dims.rbegin(); output_it != output_dims.rend();
       ++output_it) {
    if (input_it != input_dims.rend()) {
      if (!(*output_it == 0 && *input_it == 0) &&
          !(*input_it != 0 && *output_it % *input_it == 0)) {
        PADDLE_ENFORCE(false,
                       phi::errors::InvalidArgument(
                           "Invalid shape broadcast from %s to %s",
                           backend::VectorToString(input_dims).c_str(),
                           backend::VectorToString(output_dims).c_str()));
      }

      broadcast_dims.push_back(broadcast_shape.size());
      if (*output_it == *input_it || *input_it == 1) {
        broadcast_shape.push_back(*output_it);
      } else if (*output_it != *input_it) {
        // Add dimensions [I, O/I], which we will later flatten to just
        // [O]. We must do this in two phases since XLA broadcasting does not
        // support tiling.
        broadcast_shape.push_back(*input_it);
        broadcast_shape.push_back(*output_it / *input_it);
      }
      ++input_it;
    } else {
      broadcast_shape.push_back(*output_it);
    }
  }
  PADDLE_ENFORCE_EQ(input_it,
                    input_dims.rend(),
                    phi::errors::InvalidArgument(
                        "Invalid shape broadcast from %s to %s",
                        backend::VectorToString(input_dims).c_str(),
                        backend::VectorToString(output_dims).c_str()));

  std::reverse(broadcast_dims.begin(), broadcast_dims.end());
  int broadcast_shape_size = broadcast_shape.size();
  for (int64_t& broadcast_dim : broadcast_dims) {
    broadcast_dim = broadcast_shape_size - broadcast_dim - 1;
  }
  std::reverse(broadcast_shape.begin(), broadcast_shape.end());

  auto output = broadcast_in_dim(dev_ctx, src, broadcast_shape, broadcast_dims);
  if (broadcast_shape != output_dims) {
    output = reshape(dev_ctx, output, output_dims);
  }
  return output;
}

phi::DenseTensor& reshape(const phi::CustomContext& dev_ctx,
                          const phi::DenseTensor& src,
                          phi::DenseTensor& dst) {  // NOLINT
  PADDLE_ENFORCE_EQ(src.capacity(),
                    dst.capacity(),
                    phi::errors::InvalidArgument(
                        "src tensor shape is %s dst tensor shape is %s",
                        src.dims().to_str().c_str(),
                        dst.dims().to_str().c_str()));

  if (src.capacity() > 0 && dst.capacity() > 0) {
    auto input_gcu = GetHlirTensor(src);
    auto out_gcu = GetHlirTensor(dst);
    hlir::DispatchParam params;
    params.inputs = {input_gcu};
    params.outputs = {out_gcu};
    params.metadata.setValue("dims", HlirVector(out_gcu->dimensions));
    params.stream = static_cast<topsStream_t>(dev_ctx.stream());
    AOTOPS_DEBUG(kReshape, params);
    GCUOPS_TRACE_START(reshape);
    auto func_ptr = GetOpFuncPtr(kReshape, params);
    if (func_ptr) {
      auto pass = hlir::HlirDispatch::dispatch(func_ptr, params);
      PADDLE_ENFORCE(
          pass, phi::errors::InvalidArgument("dispatch %s failed!", kReshape));
    } else {
      PADDLE_ENFORCE(
          false,
          phi::errors::InvalidArgument("not find aot func for %s", kReshape));
    }
    FreeDispatchParam(params);
    GcuOpStreamSync(dev_ctx);
    GCUOPS_TRACE_END(reshape);
  }
  return dst;
}

phi::DenseTensor reshape(const phi::CustomContext& dev_ctx,
                         const phi::DenseTensor& src,
                         const std::vector<int64_t>& output_dims) {
  auto out = EmptyTensor(dev_ctx, src.dtype(), phi::make_ddim(output_dims));
  return reshape(dev_ctx, src, out);
}

phi::DenseTensor& iota(const phi::CustomContext& dev_ctx,
                       phi::DenseTensor& output,  // NOLINT
                       int64_t dim) {
  if (output.capacity() > 0) {
    auto out_gcu = GetHlirTensor(output);
    hlir::DispatchParam params;
    params.outputs = {out_gcu};
    params.metadata.setValue("iota_dimension", dim);
    params.stream = static_cast<topsStream_t>(dev_ctx.stream());
    AOTOPS_DEBUG(kIota, params);
    GCUOPS_TRACE_START(iota);
    auto func_ptr = GetOpFuncPtr(kIota, params);
    if (func_ptr) {
      auto pass = hlir::HlirDispatch::dispatch(func_ptr, params);
      PADDLE_ENFORCE(
          pass, phi::errors::InvalidArgument("dispatch %s failed!", kIota));
    } else {
      PADDLE_ENFORCE(
          false,
          phi::errors::InvalidArgument("not find aot func for %s", kIota));
    }
    FreeDispatchParam(params);
    GCUOPS_TRACE_END(iota);
    GcuOpStreamSync(dev_ctx);
  }
  return output;
}

phi::DenseTensor select(const phi::CustomContext& dev_ctx,
                        const phi::DenseTensor& pred,
                        const phi::DenseTensor& on_true,
                        const phi::DenseTensor& on_false) {
  auto out = EmptyTensor(dev_ctx, on_true.meta());
  auto condition_gcu = GetHlirTensor(pred);
  auto in1_gcu = GetHlirTensor(on_true);
  auto in2_gcu = GetHlirTensor(on_false);
  auto out_gcu = GetHlirTensor(out);

  hlir::DispatchParam params;
  params.inputs = {condition_gcu, in1_gcu, in2_gcu};
  params.outputs = {out_gcu};
  params.stream = static_cast<topsStream_t>(dev_ctx.stream());
  AOTOPS_DEBUG(kSelect, params);
  GCUOPS_TRACE_START(select);
  auto func_ptr = GetOpFuncPtr(kSelect, params);
  if (func_ptr) {
    auto pass = hlir::HlirDispatch::dispatch(func_ptr, params);
    PADDLE_ENFORCE(
        pass, phi::errors::InvalidArgument("dispatch %s failed!", kSelect));
  } else {
    PADDLE_ENFORCE(
        false,
        phi::errors::InvalidArgument("not find aot func for %s", kSelect));
  }
  FreeDispatchParam(params);
  GCUOPS_TRACE_END(select);
  GcuOpStreamSync(dev_ctx);

  return out;
}

phi::DenseTensor expand(const phi::CustomContext& dev_ctx,
                        const phi::DenseTensor& input,
                        const std::vector<int64_t> expand_shape) {
  auto input_shape = phi::vectorize(input.dims());

  auto large_shape =
      (expand_shape.size() > input_shape.size()) ? expand_shape : input_shape;
  auto small_shape =
      (expand_shape.size() > input_shape.size()) ? input_shape : expand_shape;

  std::vector<int64_t> result_dims{};
  std::vector<int64_t> broadcast_dims{};
  const int64_t j = large_shape.size() - small_shape.size();
  for (int64_t i = 0; i < j; ++i) {
    result_dims.emplace_back(large_shape[i]);
  }
  for (int64_t i = 0; i < static_cast<int64_t>(small_shape.size()); ++i) {
    int64_t dim = large_shape[i + j];
    if (dim < small_shape[i]) {
      result_dims.emplace_back(small_shape[i]);
    } else {
      result_dims.emplace_back(dim);
    }
    broadcast_dims.emplace_back(i + j);
  }

  return broadcast_in_dim(dev_ctx, input, result_dims, broadcast_dims);
}

phi::DenseTensor stack(const phi::CustomContext& dev_ctx,
                       const std::vector<phi::DenseTensor>& inputs,
                       int64_t axis) {
  PADDLE_ENFORCE_GT(inputs.size(), 0, "Stack must have at least 1 input");
  auto input_rank = inputs[0].dims().size();
  PADDLE_ENFORCE(
      -(input_rank + 1) <= axis && axis <= (input_rank + 1),
      phi::errors::InvalidArgument(
          "Stack axis range is [-(R+1), (R+1)], as [%d, %d], but got axis: %d",
          -(input_rank + 1),
          input_rank + 1,
          axis));

  if (axis < 0) {
    axis += input_rank + 1;
  }
  std::vector<phi::DenseTensor> reshaped_inputs;
  for (const auto& input : inputs) {
    auto input_shape = phi::vectorize(input.dims());
    input_shape.insert(input_shape.begin() + axis, 1);
    reshaped_inputs.push_back(reshape(dev_ctx, input, input_shape));
  }

  auto out = concat(dev_ctx, reshaped_inputs, axis);
  return out;
}

void stack(const phi::CustomContext& dev_ctx,
           const std::vector<phi::DenseTensor>& inputs,
           int64_t axis,
           phi::DenseTensor& output) {  // NOLINT
  PADDLE_ENFORCE_GT(inputs.size(), 0, "Stack must have at least 1 input");
  auto input_rank = inputs[0].dims().size();
  PADDLE_ENFORCE(
      -(input_rank + 1) <= axis && axis <= (input_rank + 1),
      phi::errors::InvalidArgument(
          "Stack axis range is [-(R+1), (R+1)], as [%d, %d], but got axis: %d",
          -(input_rank + 1),
          input_rank + 1,
          axis));

  if (axis < 0) {
    axis += input_rank + 1;
  }
  std::vector<phi::DenseTensor> reshaped_inputs;
  for (const auto& input : inputs) {
    auto input_shape = phi::vectorize(input.dims());
    input_shape.insert(input_shape.begin() + axis, 1);
    reshaped_inputs.push_back(reshape(dev_ctx, input, input_shape));
  }
  concat(dev_ctx, reshaped_inputs, axis, output);
}

phi::DenseTensor softmax_compute(const phi::CustomContext& dev_ctx,
                                 const phi::DenseTensor& input,
                                 int64_t axis) {
  auto output = EmptyTensor(dev_ctx, input.meta());
  softmax_compute(dev_ctx, input, axis, output);
  return output;
}

void softmax_compute(const phi::CustomContext& dev_ctx,
                     const phi::DenseTensor& input,
                     int64_t axis,
                     phi::DenseTensor& output) {  // NOLINT
  auto src_gcu = GetHlirTensor(input);
  auto out_gcu = GetHlirTensor(output);
  hlir::DispatchParam params;
  params.inputs = {src_gcu};
  params.outputs = {out_gcu};
  params.metadata.setValue("axis", static_cast<int32_t>(axis));
  params.stream = static_cast<topsStream_t>(dev_ctx.stream());
  AOTOPS_DEBUG(kSoftmax, params);
  GCUOPS_TRACE_START(softmax);
  auto func_ptr = GetOpFuncPtr(kSoftmax, params);
  if (func_ptr) {
    auto pass = hlir::HlirDispatch::dispatch(func_ptr, params);
    PADDLE_ENFORCE(
        pass, phi::errors::InvalidArgument("dispatch %s failed!", kSoftmax));
  } else {
    PADDLE_ENFORCE(
        false,
        phi::errors::InvalidArgument("not find aot func for %s", kSoftmax));
  }
  FreeDispatchParam(params);
  GCUOPS_TRACE_END(softmax);
  GcuOpStreamSync(dev_ctx);
}

static std::vector<int64_t> cal_sections(
    const std::vector<int64_t>& input_shape,
    int axis,
    int num,
    const std::vector<int64_t>& sections) {
  std::vector<int64_t> real_sections(sections);

  if (num <= 0) {
    const int64_t unk_dim_val = -1;
    int64_t unk_dim_idx = -1, num_of_unk = 0;
    int64_t sum_of_section = 0;
    for (size_t i = 0; i < real_sections.size(); ++i) {
      if (real_sections[i] == unk_dim_val) {
        num_of_unk += 1;
        unk_dim_idx = i;
      } else {
        sum_of_section += real_sections[i];
      }
    }
    PADDLE_ENFORCE_LE(
        num_of_unk,
        1,
        phi::errors::InvalidArgument("[TopsGraphBuilder]Split noly support at "
                                     "most 1 unk dim, but got: %d",
                                     num_of_unk));
    if (num_of_unk > 0) {
      real_sections[unk_dim_idx] = input_shape[axis] - sum_of_section;
    }
  } else {
    real_sections = std::vector<int64_t>(num, input_shape[axis] / num);
  }

  return real_sections;
}

std::vector<phi::DenseTensor> split(const phi::CustomContext& dev_ctx,
                                    const phi::DenseTensor& x,
                                    int axis,
                                    int num,
                                    const std::vector<int64_t>& sections) {
  CHECK(sections.size() > 0 || num > 0)
      << "[TopsGraphBuilder]Split must have one of split or num";
  if (num <= 0) {
    CHECK(sections.size() > 0) << "[TopsGraphBuilder]Split NOT set num, "
                                  "so must set split input";
  }

  const auto& input_shape = phi::vectorize(x.dims());
  const int64_t input_rank = input_shape.size();
  if (axis < 0) axis += input_rank;
  auto real_sections = cal_sections(input_shape, axis, num, sections);

  std::vector<phi::DenseTensor> splits;
  std::vector<phi::DenseTensor*> outs;
  // reserve vector spaec to make address fixed
  splits.reserve(real_sections.size());
  for (auto& section : real_sections) {
    auto split_dims = input_shape;
    split_dims[axis] = section;
    phi::DenseTensor split_tensor =
        EmptyTensor(dev_ctx, x.dtype(), phi::make_ddim(split_dims));

    splits.push_back(split_tensor);
    outs.push_back(&(splits.back()));
  }

  split(dev_ctx, x, axis, num, real_sections, outs);
  return splits;
}

void split(const phi::CustomContext& dev_ctx,
           const phi::DenseTensor& x,
           int axis,
           int num,
           std::vector<int64_t> sections,
           std::vector<phi::DenseTensor*> outs) {
  CHECK(sections.size() > 0 || num > 0)
      << "[TopsGraphBuilder]Split must have one of split or num";
  if (num <= 0) {
    CHECK(sections.size() > 0) << "[TopsGraphBuilder]Split NOT set num, "
                                  "so must set split input";
  }

  const auto& input_shape = phi::vectorize(x.dims());
  const int64_t input_rank = input_shape.size();
  if (axis < 0) axis += input_rank;
  auto real_sections = cal_sections(input_shape, axis, num, sections);

  int64_t start = 0;
  int64_t limit = 0;
  for (size_t i = 0; i < real_sections.size(); ++i) {
    start += (i == 0) ? 0 : real_sections[i - 1];
    limit += real_sections[i];

    *outs[i] = slice(dev_ctx, x, {axis}, {start}, {limit}, {1}, (*outs[i]));
  }
}

void cast(const phi::CustomContext& dev_ctx,
          const phi::DenseTensor& x,
          phi::DataType dtype,
          phi::DenseTensor* out) {
  if (x.dtype() == dtype) {
    out->set_meta(x.meta());
    dev_ctx.Alloc(out, out->dtype());
    TensorCopy(dev_ctx, x, false, out);
    return;
  }

  if (dtype == phi::DataType::FLOAT32) {
    dev_ctx.Alloc<float>(out);
  } else if (dtype == phi::DataType::FLOAT64) {
    dev_ctx.Alloc<double>(out);
  } else if (dtype == phi::DataType::FLOAT16) {
    dev_ctx.Alloc<phi::dtype::float16>(out);
  } else if (dtype == phi::DataType::INT16) {
    dev_ctx.Alloc<int16_t>(out);
  } else if (dtype == phi::DataType::INT32) {
    dev_ctx.Alloc<int32_t>(out);
  } else if (dtype == phi::DataType::INT64) {
    dev_ctx.Alloc<int64_t>(out);
  } else if (dtype == phi::DataType::BOOL) {
    dev_ctx.Alloc<bool>(out);
  } else if (dtype == phi::DataType::UINT8) {
    dev_ctx.Alloc<uint8_t>(out);
  } else if (dtype == phi::DataType::INT8) {
    dev_ctx.Alloc<int8_t>(out);
  } else if (dtype == phi::DataType::COMPLEX64) {
    dev_ctx.Alloc<phi::dtype::complex<float>>(out);
  } else if (dtype == phi::DataType::COMPLEX128) {
    dev_ctx.Alloc<phi::dtype::complex<double>>(out);
  } else {
    phi::errors::InvalidArgument("Unsupported cast dtype %s", dtype);
  }
  TensorNameMap input_names;
  input_names["X"] = {"x"};

  TensorValueMap inputs;
  inputs["X"] = {const_cast<DenseTensor*>(&x)};

  TensorNameMap output_names;
  output_names["Out"] = {"out"};

  TensorValueMap outputs;
  outputs["Out"] = {out};

  GcuAttributeMap attrs;
  attrs["in_dtype"] = static_cast<int>(x.dtype());
  attrs["out_dtype"] = static_cast<int>(dtype);

  GcuRunner(input_names, inputs, output_names, outputs, attrs, "cast", dev_ctx);
  // }
}

phi::DenseTensor cast(const phi::CustomContext& dev_ctx,
                      const phi::DenseTensor& x,
                      phi::DataType dtype) {
  phi::DenseTensor out;
  auto meta = x.meta();
  meta.dtype = dtype;
  out.set_meta(meta);
  cast(dev_ctx, x, dtype, &out);
  return out;
}

phi::DenseTensor& one_hot(const phi::CustomContext& dev_ctx,
                          const phi::DenseTensor& x,
                          int64_t axis,
                          int64_t depth,
                          phi::DenseTensor& out) {  // NOLINT
  std::vector<int64_t> label_dims = phi::vectorize(out.dims());
  std::vector<int64_t> reshape_dims = label_dims;
  reshape_dims.at(axis) = 1;
  auto orders = EmptyTensor(dev_ctx, x.dtype(), phi::make_ddim(label_dims));
  orders = custom_kernel::iota(dev_ctx, orders, axis);
  auto x_reshape = reshape(dev_ctx, x, reshape_dims);
  auto x_boradcast = broadcast_to(dev_ctx, x_reshape, label_dims);
  auto idx_compare = equal_compute(dev_ctx, x_boradcast, orders);
  auto ones = ones_like(dev_ctx, orders);
  auto zeros = zeros_like(dev_ctx, orders);
  auto label_int32 = select(dev_ctx, idx_compare, ones, zeros);
  cast(dev_ctx, label_int32, out.dtype(), &out);
  return out;
}

phi::DenseTensor one_hot(const phi::CustomContext& dev_ctx,
                         const phi::DenseTensor& x,
                         int64_t axis,
                         int64_t depth) {
  // calculate out dims
  std::vector<int64_t> out_dims = phi::vectorize(x.dims());
  out_dims.insert(out_dims.begin() + axis, depth);
  auto out =
      EmptyTensor(dev_ctx, phi::DataType::FLOAT32, phi::make_ddim(out_dims));
  return one_hot(dev_ctx, x, axis, depth, out);
}

}  // namespace custom_kernel
