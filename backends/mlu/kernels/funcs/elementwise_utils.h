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

#pragma once

#include "kernels/funcs/mlu_baseop.h"

namespace custom_kernel {

inline void GetBroadcastDimsArrays(const phi::DDim& x_dims,
                                   const phi::DDim& y_dims,
                                   int* x_dims_array,
                                   int* y_dims_array,
                                   int* out_dims_array,
                                   const int max_dim,
                                   const int axis) {
  PADDLE_ENFORCE_GE(
      axis,
      0,
      phi::errors::InvalidArgument(
          "Axis should be great than or equal to 0, but received axis is %d.",
          axis));
  PADDLE_ENFORCE_LT(axis,
                    max_dim,
                    phi::errors::InvalidArgument(
                        "Axis should be less than %d, but received axis is %d.",
                        max_dim,
                        axis));
  if (x_dims.size() > y_dims.size()) {
    std::fill(y_dims_array, y_dims_array + axis, 1);
    if (axis + y_dims.size() < max_dim) {
      std::fill(y_dims_array + axis + y_dims.size(), y_dims_array + max_dim, 1);
    }
    std::copy(x_dims.Get(), x_dims.Get() + x_dims.size(), x_dims_array);
    std::copy(y_dims.Get(), y_dims.Get() + y_dims.size(), y_dims_array + axis);
  } else {
    std::fill(x_dims_array, x_dims_array + axis, 1);
    if (axis + x_dims.size() < max_dim) {
      std::fill(x_dims_array + axis + x_dims.size(), x_dims_array + max_dim, 1);
    }
    std::copy(x_dims.Get(), x_dims.Get() + x_dims.size(), x_dims_array + axis);
    std::copy(y_dims.Get(), y_dims.Get() + y_dims.size(), y_dims_array);
  }

  for (int i = 0; i < max_dim; i++) {
    PADDLE_ENFORCE_EQ(
        x_dims_array[i] == y_dims_array[i] || x_dims_array[i] <= 1 ||
            y_dims_array[i] <= 1,
        true,
        phi::errors::InvalidArgument(
            "Broadcast dimension mismatch. Operands could "
            "not be broadcast together with the shape of X = [%s] and "
            "the shape of Y = [%s]. Received [%d] in X is not equal to "
            "[%d] in Y at i:%d.",
            x_dims,
            y_dims,
            x_dims_array[i],
            y_dims_array[i],
            i));
    if ((x_dims_array[i] > 1 || y_dims_array[i] > 1) ||
        (x_dims_array[i] == 1 && y_dims_array[i] == 1)) {
      out_dims_array[i] = (std::max)(x_dims_array[i], y_dims_array[i]);
    } else {
      out_dims_array[i] = -1;
    }
  }
}

inline void GetReduceAxes(const int axis,
                          const phi::DDim& src_ddims,
                          const phi::DDim& target_ddims,
                          std::vector<int>* axes) {
  int64_t src_dim_size = src_ddims.size();
  int64_t target_dim_size = target_ddims.size();
  for (int64_t i = 0; i < src_dim_size; ++i) {
    if (i < axis || i >= target_dim_size + axis) {
      axes->push_back(i);
      continue;
    }
    if (src_ddims[i] > target_ddims[i - axis]) {
      axes->push_back(i);
    }
  }
}

inline void GetReduceAxesAndDstDims(const int axis,
                                    const phi::DDim& src_ddims,
                                    const phi::DDim& target_ddims,
                                    std::vector<int>* reduce_axes,
                                    std::vector<int>* dst_dims_vec) {
  int64_t src_dim_size = src_ddims.size();
  int64_t target_dim_size = target_ddims.size();

  int src_axis = (target_dim_size < src_dim_size ? axis : 0);
  for (int ax = 0; ax < src_dim_size; ++ax) {
    if ((ax < src_axis || ax >= src_axis + target_dim_size) ||
        (src_ddims[ax] > 1 && target_ddims[ax - src_axis] == 1)) {
      reduce_axes->push_back(ax);
    } else {
      dst_dims_vec->push_back(src_ddims[ax]);
    }
  }
  if (dst_dims_vec->size() == 0) {
    // target_var is scalar
    dst_dims_vec->push_back(1);
  }
}

template <typename T>
void MLUOpTensorKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::DenseTensor& y,
                       int axis,
                       const cnnlOpTensorDesc_t op_tensor_type,
                       phi::DenseTensor* out) {
  PADDLE_ENFORCE_EQ((op_tensor_type == CNNL_OP_TENSOR_ADD) ||
                        (op_tensor_type == CNNL_OP_TENSOR_SUB) ||
                        (op_tensor_type == CNNL_OP_TENSOR_MUL),
                    true,
                    phi::errors::Unavailable(
                        "This kernel of MLU only support ADD, SUB, MUL."));
  dev_ctx.template Alloc<T>(out);

  const auto& x_dims = x.dims();
  const auto& y_dims = y.dims();
  axis =
      (axis < 0 ? (std::abs(x_dims.size() - y_dims.size()) + axis + 1) : axis);
  int max_dim = std::max(x_dims.size(), y_dims.size());
  std::vector<int> x_dims_array(max_dim);
  std::vector<int> y_dims_array(max_dim);
  std::vector<int> out_dims_array(max_dim);
  GetBroadcastDimsArrays(x_dims,
                         y_dims,
                         x_dims_array.data(),
                         y_dims_array.data(),
                         out_dims_array.data(),
                         max_dim,
                         axis);

  MLUCnnlTensorDesc x_desc(max_dim, x_dims_array.data(), ToCnnlDataType<T>());
  MLUCnnlTensorDesc y_desc(max_dim, y_dims_array.data(), ToCnnlDataType<T>());
  MLUCnnlTensorDesc out_desc(*out);
  MLUCnnlOpTensorDesc op_tensor_desc(
      op_tensor_type, ToCnnlDataType<T>(), CNNL_NOT_PROPAGATE_NAN);

  MLUCnnl::OpTensor(dev_ctx,
                    op_tensor_desc.get(),
                    x_desc.get(),
                    GetBasePtr(&x),
                    y_desc.get(),
                    GetBasePtr(&y),
                    out_desc.get(),
                    GetBasePtr(out),
                    ToCnnlDataType<T>());
}

// ------------------ BinaryOp -----------------
enum BINARY_FUNCTOR {
  DIV,
  DIVNONAN,
  MAXIMUM,
  MINIMUM,
  POW,
};

template <BINARY_FUNCTOR func>
void MLUBinary(const Context& dev_ctx,
               cnnlComputationPreference_t prefer,
               const cnnlTensorDescriptor_t x_desc,
               const void* x,
               const cnnlTensorDescriptor_t y_desc,
               const void* y,
               const cnnlTensorDescriptor_t out_desc,
               void* out);

template <>
inline void MLUBinary<DIV>(const Context& dev_ctx,
                           cnnlComputationPreference_t prefer,
                           const cnnlTensorDescriptor_t x_desc,
                           const void* x,
                           const cnnlTensorDescriptor_t y_desc,
                           const void* y,
                           const cnnlTensorDescriptor_t out_desc,
                           void* out) {
  MLUCnnl::Div(dev_ctx, prefer, x_desc, x, y_desc, y, out_desc, out);
}

template <>
inline void MLUBinary<MAXIMUM>(
    const Context& dev_ctx,
    cnnlComputationPreference_t prefer,  // useless, only for compatible
    const cnnlTensorDescriptor_t x_desc,
    const void* x,
    const cnnlTensorDescriptor_t y_desc,
    const void* y,
    const cnnlTensorDescriptor_t out_desc,
    void* out) {
  MLUCnnl::Maximum(dev_ctx, x_desc, x, y_desc, y, out_desc, out);
}

template <>
inline void MLUBinary<MINIMUM>(const Context& dev_ctx,
                               cnnlComputationPreference_t prefer,
                               const cnnlTensorDescriptor_t in1_desc,
                               const void* in1,
                               const cnnlTensorDescriptor_t in2_desc,
                               const void* in2,
                               const cnnlTensorDescriptor_t out_desc,
                               void* out) {
  MLUCnnl::Minimum(dev_ctx, in1_desc, in1, in2_desc, in2, out_desc, out);
}

template <>
inline void MLUBinary<POW>(const Context& dev_ctx,
                           cnnlComputationPreference_t prefer,
                           const cnnlTensorDescriptor_t x_desc,
                           const void* x,
                           const cnnlTensorDescriptor_t y_desc,
                           const void* y,
                           const cnnlTensorDescriptor_t out_desc,
                           void* out) {
  MLUCnnl::Pow(dev_ctx, prefer, x_desc, x, y_desc, y, out_desc, out);
}

template <BINARY_FUNCTOR Functor, typename T>
void MLUBinaryOp(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 const phi::DenseTensor& y,
                 int axis,
                 phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  const auto& x_dims = x.dims();
  const auto& y_dims = y.dims();
  axis =
      (axis < 0 ? (std::abs(x_dims.size() - y_dims.size()) + axis + 1) : axis);
  int max_dim = std::max(x_dims.size(), y_dims.size());
  std::vector<int> x_dims_array(max_dim);
  std::vector<int> y_dims_array(max_dim);
  std::vector<int> out_dims_array(max_dim);
  GetBroadcastDimsArrays(x_dims,
                         y_dims,
                         x_dims_array.data(),
                         y_dims_array.data(),
                         out_dims_array.data(),
                         max_dim,
                         axis);

  MLUCnnlTensorDesc x_desc(max_dim, x_dims_array.data(), ToCnnlDataType<T>());
  MLUCnnlTensorDesc y_desc(max_dim, y_dims_array.data(), ToCnnlDataType<T>());
  MLUCnnlTensorDesc out_desc(*out, CNNL_LAYOUT_ARRAY, ToCnnlDataType<T>());

  cnnlComputationPreference_t prefer_type = CNNL_COMPUTATION_HIGH_PRECISION;
  MLUBinary<Functor>(dev_ctx,
                     prefer_type,
                     x_desc.get(),
                     GetBasePtr(&x),
                     y_desc.get(),
                     GetBasePtr(&y),
                     out_desc.get(),
                     GetBasePtr(out));
}

// ------------------ UnaryOp -----------------
enum UNARY_FUNCTOR {
  NEG,
  RECIPROCAL,
};

template <UNARY_FUNCTOR func>
void MLUUnary(const Context& dev_ctx,
              cnnlComputationPreference_t prefer,
              const cnnlTensorDescriptor_t input_desc,
              const void* input,
              const cnnlTensorDescriptor_t output_desc,
              void* output);

template <>
inline void MLUUnary<NEG>(const Context& dev_ctx,
                          cnnlComputationPreference_t prefer,
                          const cnnlTensorDescriptor_t input_desc,
                          const void* input,
                          const cnnlTensorDescriptor_t output_desc,
                          void* output) {
  MLUCnnl::Neg(dev_ctx, input_desc, input, output_desc, output);
}

template <>
inline void MLUUnary<RECIPROCAL>(const Context& dev_ctx,
                                 cnnlComputationPreference_t prefer,
                                 const cnnlTensorDescriptor_t input_desc,
                                 const void* input,
                                 const cnnlTensorDescriptor_t output_desc,
                                 void* output) {
  MLUCnnl::Reciprocal(dev_ctx, input_desc, input, output_desc, output);
}

template <UNARY_FUNCTOR Functor, typename Tin, typename Tout = Tin>
void MLUUnaryOp(const Context& dev_ctx,
                const phi::DenseTensor& x,
                phi::DenseTensor* out) {
  dev_ctx.template Alloc<Tout>(out);

  MLUCnnlTensorDesc x_desc(x, CNNL_LAYOUT_ARRAY, ToCnnlDataType<Tin>());
  MLUCnnlTensorDesc out_desc(*out, CNNL_LAYOUT_ARRAY, ToCnnlDataType<Tout>());

  cnnlComputationPreference_t prefer_type = CNNL_COMPUTATION_HIGH_PRECISION;
  MLUUnary<Functor>(dev_ctx,
                    prefer_type,
                    x_desc.get(),
                    GetBasePtr(&x),
                    out_desc.get(),
                    GetBasePtr(out));
}

// ------------------ MLUElementwiseGradOp -----------------
enum MINMAX_GRAD_FUNCTOR {
  MAXIMUM_GRAD,
  MINIMUM_GRAD,
};
template <MINMAX_GRAD_FUNCTOR Functor, typename Tin, typename Tout = Tin>
void MLUMinMaxGradHelper(const Context& dev_ctx,
                         const phi::DenseTensor& x,
                         const phi::DenseTensor& y,
                         const phi::DenseTensor& dout,
                         int axis,
                         phi::DenseTensor* dx,
                         phi::DenseTensor* dy) {
  const auto& x_dims = x.dims();
  const auto& y_dims = y.dims();
  axis =
      (axis < 0 ? (std::abs(x_dims.size() - y_dims.size()) + axis + 1) : axis);
  int max_dim = std::max(x_dims.size(), y_dims.size());
  std::vector<int> x_dims_array(max_dim);
  std::vector<int> y_dims_array(max_dim);
  std::vector<int> out_dims_array(max_dim);
  GetBroadcastDimsArrays(x_dims,
                         y_dims,
                         x_dims_array.data(),
                         y_dims_array.data(),
                         out_dims_array.data(),
                         max_dim,
                         axis);

  // mask = Logic(x, y) only support min & max
  cnnlLogicOp_t logic =
      Functor == MAXIMUM_GRAD ? CNNL_LOGIC_OP_GE : CNNL_LOGIC_OP_LE;
  Tensor mask;
  mask.Resize(phi::make_ddim(out_dims_array));
  dev_ctx.template Alloc<Tin>(&mask);

  cnnlDataType_t data_type = ToCnnlDataType<Tin>();
  MLUCnnlTensorDesc x_desc(max_dim, x_dims_array.data(), data_type);
  MLUCnnlTensorDesc y_desc(max_dim, y_dims_array.data(), data_type);
  MLUCnnlTensorDesc mask_desc(max_dim, out_dims_array.data(), data_type);
  MLUCnnl::Logic(dev_ctx,
                 logic,
                 x_desc.get(),
                 GetBasePtr(&x),
                 y_desc.get(),
                 GetBasePtr(&y),
                 mask_desc.get(),
                 GetBasePtr(&mask));

  // dx = Mul(dz, mask)
  Tensor dx_temp;
  dx_temp.Resize(dout.dims());
  dev_ctx.template Alloc<Tout>(&dx_temp);
  MLUCnnlTensorDesc dout_desc(dout);
  MLUCnnlOpTensorDesc mul_op_desc(
      CNNL_OP_TENSOR_MUL, data_type, CNNL_NOT_PROPAGATE_NAN);
  MLUCnnl::OpTensor(dev_ctx,
                    mul_op_desc.get(),
                    dout_desc.get(),
                    GetBasePtr(&dout),
                    dout_desc.get(),
                    GetBasePtr(&mask),
                    dout_desc.get(),
                    GetBasePtr(&dx_temp),
                    data_type);

  // dy = Sub(dz, dx)
  Tensor dy_temp;
  dy_temp.Resize(dout.dims());
  dev_ctx.template Alloc<Tout>(&dy_temp);
  MLUCnnlOpTensorDesc sub_op_desc(
      CNNL_OP_TENSOR_SUB, data_type, CNNL_NOT_PROPAGATE_NAN);
  MLUCnnl::OpTensor(dev_ctx,
                    sub_op_desc.get(),
                    dout_desc.get(),
                    GetBasePtr(&dout),
                    dout_desc.get(),
                    GetBasePtr(&dx_temp),
                    dout_desc.get(),
                    GetBasePtr(&dy_temp),
                    data_type);

  if (dx) {
    if (dx->dims() != dout.dims()) {
      dev_ctx.template Alloc<Tout>(dx);
      std::vector<int> reduce_axes;
      GetReduceAxes(axis, dx_temp.dims(), dx->dims(), &reduce_axes);
      MLUCnnlReduceDesc reduction_desc(reduce_axes,
                                       CNNL_REDUCE_ADD,
                                       data_type,
                                       CNNL_NOT_PROPAGATE_NAN,
                                       CNNL_REDUCE_NO_INDICES,
                                       CNNL_32BIT_INDICES);
      MLUCnnlTensorDesc dx_desc(*dx);
      MLUCnnl::Reduce(dev_ctx,
                      true /*need_workspace*/,
                      reduction_desc.get(),
                      nullptr,
                      dout_desc.get(),
                      GetBasePtr(&dx_temp),
                      0,
                      nullptr,
                      nullptr,
                      dx_desc.get(),
                      GetBasePtr(dx));
    } else {
      *dx = dx_temp;
    }
  }

  if (dy) {
    if (dy->dims() != dout.dims()) {
      dev_ctx.template Alloc<Tout>(dy);
      std::vector<int> reduce_axes;
      GetReduceAxes(axis, dy_temp.dims(), dy->dims(), &reduce_axes);
      MLUCnnlReduceDesc reduction_desc(reduce_axes,
                                       CNNL_REDUCE_ADD,
                                       data_type,
                                       CNNL_NOT_PROPAGATE_NAN,
                                       CNNL_REDUCE_NO_INDICES,
                                       CNNL_32BIT_INDICES);
      MLUCnnlTensorDesc dy_desc(*dy);
      MLUCnnl::Reduce(dev_ctx,
                      true /*need_workspace*/,
                      reduction_desc.get(),
                      nullptr,
                      dout_desc.get(),
                      GetBasePtr(&dy_temp),
                      0,
                      nullptr,
                      nullptr,
                      dy_desc.get(),
                      GetBasePtr(dy));
    } else {
      *dy = dy_temp;
    }
  }
}

}  // namespace custom_kernel
