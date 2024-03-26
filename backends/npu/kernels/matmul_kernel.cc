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

#include "kernels/funcs/format_utils.h"
#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"

namespace custom_kernel {
static inline aclTensor* Create_Acltensor(const phi::DenseTensor& paddletensor,
                                          const bool transpose) {
  static const auto aclCreateTensor = GET_OP_API_FUNC(aclCreateTensor);
  auto tensor_dtype = paddletensor.dtype();
  auto acl_data_type = ConvertToNpuDtype(tensor_dtype);
  const auto dimNum =
      paddletensor.dims().size() == 0 ? 1 : paddletensor.dims().size();
  std::vector<int64_t> storageDims(dimNum - 1);
  storageDims.push_back(paddletensor.numel() * sizeof(tensor_dtype));
  aclFormat format = ACL_FORMAT_ND;
  switch (dimNum) {
    break;
    case 4:
      format = ACL_FORMAT_NCHW;
      break;
    case 5:
      format = ACL_FORMAT_NCDHW;
      break;
    default:
      format = ACL_FORMAT_ND;
  }
  auto shape = phi::vectorize(paddletensor.dims());
  auto strides = shape;
  if (!strides.empty()) {
    strides.erase(strides.begin());
  }
  strides.push_back(1);
  for (int i = static_cast<int>(strides.size()) - 2; i >= 0; i--) {
    strides[i] = strides[i] * strides[i + 1];
  }
  if (transpose) {
    std::swap(shape[shape.size() - 1], shape[shape.size() - 2]);
    std::swap(strides[strides.size() - 1], strides[strides.size() - 2]);
  }
  auto acl_tensor = aclCreateTensor(shape.data(),
                                    dimNum,
                                    acl_data_type,
                                    strides.data(),
                                    0,
                                    format,
                                    shape.data(),
                                    dimNum,
                                    const_cast<void*>(paddletensor.data()));
  return acl_tensor;
}

static std::vector<int64_t> MatmulInferShape(const phi::DenseTensor& x,
                                             const phi::DenseTensor& y,
                                             bool trans_x,
                                             bool trans_y) {
  std::vector<int64_t> dims_x = phi::vectorize(x.dims());
  std::vector<int64_t> dims_y = phi::vectorize(y.dims());
  auto ndims_x = dims_x.size();
  auto ndims_y = dims_y.size();
  PADDLE_ENFORCE_GT(ndims_x,
                    0UL,
                    phi::errors::InvalidArgument(
                        "The Input(x) dims size must be greater than 0,"
                        " but reviced dims size is 0. "));
  PADDLE_ENFORCE_GT(ndims_y,
                    0UL,
                    phi::errors::InvalidArgument(
                        "The Input(y) dims size must be greater than 0,"
                        " but reviced dims size is 0. "));

  bool x_broadcasted = false, y_broadcasted = false;
  if (ndims_x == 1) {
    dims_x.insert(dims_x.begin(), 1);
    ndims_x = 2;
    x_broadcasted = true;
  }

  if (ndims_y == 1) {
    dims_y.push_back(1);
    ndims_y = 2;
    y_broadcasted = true;
  }

  size_t M = 0, N = 0;
  if (trans_x) {
    M = dims_x[ndims_x - 1];
  } else {
    M = dims_x[ndims_x - 2];
  }
  if (trans_y) {
    N = dims_y[ndims_y - 2];
  } else {
    N = dims_y[ndims_y - 1];
  }

  std::vector<int64_t> new_dims;
  if (ndims_x > ndims_y) {
    new_dims.assign(dims_x.begin(), dims_x.end() - 2);
  } else if (ndims_x < ndims_y) {
    new_dims.assign(dims_y.begin(), dims_y.end() - 2);
  } else {
    new_dims.reserve(ndims_x);
    for (size_t i = 0; i < ndims_x - 2; ++i) {
      new_dims.push_back(std::max(dims_x[i], dims_y[i]));
    }
  }
  if (!x_broadcasted) {
    new_dims.push_back(M);
  }
  if (!y_broadcasted) {
    new_dims.push_back(N);
  }
  return new_dims;
}

template <typename Context>
static phi::DenseTensor TensorCast(const Context& dev_ctx,
                                   const phi::DenseTensor& tensor) {
  if (tensor.dtype() != phi::DataType::FLOAT64) {
    return tensor;
  }
  phi::DenseTensor tensor_cast;
  phi::DenseTensorMeta x_temp_cast_meta = {phi::DataType::FLOAT16,
                                           tensor.dims()};
  tensor_cast.set_meta(x_temp_cast_meta);
  dev_ctx.Alloc(&tensor_cast, phi::DataType::FLOAT16);
  int aclDtype = ConvertToNpuDtype(phi::DataType::FLOAT16);
  EXEC_NPU_CMD(aclnnCast, dev_ctx, tensor, aclDtype, tensor_cast);
  return tensor_cast;
}

template <typename Context>
static phi::DenseTensor OutCast(const Context& dev_ctx, phi::DenseTensor* out) {
  if (out->dtype() != phi::DataType::FLOAT64) {
    return *out;
  }
  phi::DenseTensor out_fp16;
  phi::DenseTensorMeta out_temp_cast_meta = {phi::DataType::FLOAT16,
                                             out->dims()};
  out_fp16.set_meta(out_temp_cast_meta);
  dev_ctx.Alloc(&out_fp16, phi::DataType::FLOAT16);
  return out_fp16;
}

template <typename Context>
static void OutRevert(const Context& dev_ctx,
                      const phi::DenseTensor& out_temp,
                      phi::DenseTensor* out) {
  if (out->dtype() == phi::DataType::FLOAT64) {
    int aclDtype = ConvertToNpuDtype(out->dtype());
    EXEC_NPU_CMD(aclnnCast, dev_ctx, out_temp, aclDtype, out);
  }
}

template <typename T, typename Context>
static void AclnnMatmulForward(const Context& dev_ctx,
                               const phi::DenseTensor& X,
                               const phi::DenseTensor& Y,
                               phi::DenseTensor* out,
                               const bool transpose_x,
                               const bool transpose_y) {
  // cubeMathType(int8_t，计算输入)：
  // 0:KEEP_DTYE保持输入的数据类型进行计算
  // 1:ALLOW_FP32_DOWN_PRECISON，允许将输入数据降精度计算
  // 2:USE_FP16，允许转换为数据类型FLOAT16进行计算。当输入数据类型是FLOAT，转换为FLOAT16计算。
  // 3:USE_HF32，允许转换为数据类型HFLOAT32计算
  // 当前取0，维持原类型即可
  int8_t cube_math_type = 0;
  dev_ctx.template Alloc<T>(out);
  std::vector<int64_t> x_dims = phi::vectorize(X.dims());
  std::vector<int64_t> y_dims = phi::vectorize(Y.dims());
  int x_ndim = x_dims.size();
  int y_ndim = y_dims.size();
  phi::DenseTensor x_temp(X), y_temp(Y);
  if (x_ndim == 1) {
    x_dims.insert(x_dims.begin(), 1);
    x_temp.Resize(phi::make_ddim(x_dims));
  }
  if (y_ndim == 1) {
    y_dims.push_back(1);
    y_temp.Resize(phi::make_ddim(y_dims));
  }
  // 如果是float64转换成fp16计算，其余数据类型透传
  auto x_temp_cast = TensorCast<Context>(dev_ctx, x_temp);
  auto y_temp_cast = TensorCast<Context>(dev_ctx, y_temp);
  // 如果out是float64,也要转成fp16，其余数据类型透传
  auto out_temp = OutCast<Context>(dev_ctx, out);
  // transpose场景下通过acl接口实现view机制
  aclTensor* x_acltensor = Create_Acltensor(x_temp_cast, transpose_x);
  aclTensor* y_acltensor = Create_Acltensor(y_temp_cast, transpose_y);
  EXEC_NPU_CMD(
      aclnnMatmul, dev_ctx, x_acltensor, y_acltensor, out_temp, cube_math_type);
  // 如果是float64,输出结果最后转成float64输出，其余数据类型透传
  OutRevert<Context>(dev_ctx, out_temp, out);
}

inline static std::vector<int64_t> GetReduceDims(
    const std::vector<int64_t>& x_dims,
    const std::vector<int64_t>& y_dims,
    const std::vector<int64_t>& brd_dims) {
  std::vector<int64_t> axes;
  int64_t size = brd_dims.size();
  int64_t diff = brd_dims.size() - x_dims.size();
  // 输入输出shape相同,有1则需要reduce
  if (x_dims.size() == brd_dims.size() && x_dims.size() == y_dims.size()) {
    for (auto i = 0; i < x_dims.size(); i++) {
      if (x_dims[i] == 1) {
        axes.push_back(i);
      }
    }
    return axes;
  }
  for (int64_t i = 0; i < size; ++i) {
    if (i < diff) {
      axes.push_back(i);
      continue;
    }
    if ((brd_dims[i] > x_dims[i - diff])) {
      axes.push_back(i);
    }
  }
  return axes;
}

template <typename T, typename Context>
static phi::DenseTensor MakeReduceTempOut(const Context& dev_ctx,
                                          std::vector<int64_t> reduce_axes,
                                          const phi::DenseTensor& x_tensor,
                                          const phi::DenseTensor& y_tensor,
                                          phi::DenseTensor* d_tensor,
                                          bool transpose_x,
                                          bool transpose_y) {
  phi::DenseTensor d_tmp;
  if (!reduce_axes.empty()) {
    auto dx_tmp_shape =
        MatmulInferShape(x_tensor, y_tensor, transpose_x, transpose_y);
    d_tmp.Resize(phi::make_ddim(dx_tmp_shape));
    dev_ctx.template Alloc<T>(&d_tmp);
  } else {
    d_tmp = *d_tensor;
  }
  return d_tmp;
}

template <typename Context>
static void OuttempReduceOut(const Context& dev_ctx,
                             const std::vector<int64_t>& reduce_axes,
                             const phi::DenseTensor& out_temp,
                             phi::DenseTensor* out) {
  bool keep_dims = false;
  phi::DenseTensor out_resized(*out);
  if (!reduce_axes.empty()) {
    if (out_temp.dims().size() == out->dims().size()) {
      keep_dims = true;
    }
    auto dtype = ConvertToNpuDtype(out->dtype());
    EXEC_NPU_CMD(aclnnReduceSum,
                 dev_ctx,
                 out_temp,
                 reduce_axes,
                 keep_dims,
                 dtype,
                 out_resized);
  }
}

template <typename T, typename Context>
static void AclnnMatmulBackward(const Context& dev_ctx,
                                const phi::DenseTensor& X,
                                const phi::DenseTensor& Y,
                                const phi::DenseTensor& dout,
                                const bool transpose_x,
                                const bool transpose_y,
                                phi::DenseTensor* dx,
                                phi::DenseTensor* dy) {
  int8_t cube_math_type = 0;
  auto x_dims = X.dims();
  auto y_dims = Y.dims();
  auto dout_dims = dout.dims();
  int x_ndim = x_dims.size();
  int y_ndim = y_dims.size();
  int out_ndim = dout_dims.size();
  // 输入输出shape全为1，反向用乘法
  if (x_ndim == 1 && y_ndim == 1) {
    if (dx) {
      dev_ctx.template Alloc<T>(dx);
      // aclnnMul支持float64，此处不用转数据类型
      EXEC_NPU_CMD(aclnnMul, dev_ctx, dout, Y, *dx);
    }
    if (dy) {
      dev_ctx.template Alloc<T>(dy);
      // aclnnMul支持float64，此处不用转数据类型
      EXEC_NPU_CMD(aclnnMul, dev_ctx, dout, X, *dy);
    }
    return;
  }
  phi::DenseTensor x_temp(X), y_temp(Y), dout_temp(dout);
  if (x_ndim == 1) {
    auto x_dims_vector = phi::vectorize(x_dims);
    auto out_dims_vector = phi::vectorize(dout_dims);
    x_dims_vector.insert(x_dims_vector.begin(), 1);
    out_dims_vector.insert(out_dims_vector.end() - 1, 1);
    x_dims = phi::make_ddim(x_dims_vector);
    x_temp.Resize(x_dims);
    dout_dims = phi::make_ddim(out_dims_vector);
    dout_temp.Resize(dout_dims);
  }
  if (y_ndim == 1) {
    auto y_dims_vector = phi::vectorize(y_dims);
    auto out_dims_vector = phi::vectorize(dout_dims);
    y_dims_vector.push_back(1);
    out_dims_vector.push_back(1);
    y_dims = phi::make_ddim(y_dims_vector);
    y_temp.Resize(y_dims);
    dout_dims = phi::make_ddim(out_dims_vector);
    dout_temp.Resize(dout_dims);
  }

  if (dx) {
    dev_ctx.template Alloc<T>(dx);
    phi::DenseTensor dx_tmp;
    phi::DenseTensor dx_resized(*dx);
    dx_resized.Resize(x_dims);
    // 如果需要reduce sum，先计算出需要reduce sum的轴
    auto reduce_dims = GetReduceDims(phi::vectorize(x_dims),
                                     phi::vectorize(y_dims),
                                     phi::vectorize(dout_dims));
    if (transpose_x) {
      // 如果需要reduce，先申请一个reduce前的临时tensor,不需要reduce时透传
      dx_tmp = MakeReduceTempOut<T, Context>(dev_ctx,
                                             reduce_dims,
                                             y_temp,
                                             dout_temp,
                                             &dx_resized,
                                             transpose_y,
                                             true);
      // 如果out是float64,也要转成fp16，其余数据类型透传
      auto dx_tmp_out = OutCast<Context>(dev_ctx, &dx_tmp);
      // 如果是float64转换成fp16计算，其余数据类型透传
      auto Y_temp_cast = TensorCast<Context>(dev_ctx, y_temp);
      // 如果是float64转换成fp16计算，其余数据类型透传
      auto dout_temp_cast = TensorCast<Context>(dev_ctx, dout_temp);
      // transpose场景下通过acl接口实现view机制
      aclTensor* x_acltensor = Create_Acltensor(Y_temp_cast, transpose_y);
      aclTensor* y_acltensor = Create_Acltensor(dout_temp_cast, true);
      EXEC_NPU_CMD(aclnnMatmul,
                   dev_ctx,
                   x_acltensor,
                   y_acltensor,
                   dx_tmp_out,
                   cube_math_type);
      // 如果是float64,输出结果最后转成float64输出，其余数据类型透传
      OutRevert<Context>(dev_ctx, dx_tmp_out, &dx_tmp);
    } else {
      dx_tmp = MakeReduceTempOut<T, Context>(dev_ctx,
                                             reduce_dims,
                                             dout_temp,
                                             y_temp,
                                             &dx_resized,
                                             false,
                                             !transpose_y);
      // 如果out是float64,也要转成fp16，其余数据类型透传
      auto dx_tmp_out = OutCast<Context>(dev_ctx, &dx_tmp);
      // 如果是float64转换成fp16计算，其余数据类型透传
      auto Y_temp_cast = TensorCast<Context>(dev_ctx, y_temp);
      // 如果是float64转换成fp16计算，其余数据类型透传
      auto dout_temp_cast = TensorCast<Context>(dev_ctx, dout_temp);
      aclTensor* x_acltensor = Create_Acltensor(dout_temp_cast, false);
      aclTensor* y_acltensor = Create_Acltensor(Y_temp_cast, !transpose_y);
      EXEC_NPU_CMD(aclnnMatmul,
                   dev_ctx,
                   x_acltensor,
                   y_acltensor,
                   dx_tmp_out,
                   cube_math_type);
      // 如果是float64,输出结果最后转成float64输出，其余数据类型透传
      OutRevert<Context>(dev_ctx, dx_tmp_out, &dx_tmp);
    }
    // reduce场景下，生成对应的reduce tensor，否则透传
    OuttempReduceOut<Context>(dev_ctx, reduce_dims, dx_tmp, &dx_resized);
  }
  if (dy) {
    dev_ctx.template Alloc<T>(dy);
    phi::DenseTensor dy_tmp;
    phi::DenseTensor dy_resized(*dy);
    dy_resized.Resize(y_dims);
    // 如果需要reduce sum，先计算出需要reduce sum的轴
    auto reduce_dims = GetReduceDims(phi::vectorize(y_dims),
                                     phi::vectorize(x_dims),
                                     phi::vectorize(dout_dims));
    if (transpose_y) {
      dy_tmp = MakeReduceTempOut<T, Context>(dev_ctx,
                                             reduce_dims,
                                             dout_temp,
                                             x_temp,
                                             &dy_resized,
                                             true,
                                             transpose_x);
      // 如果out是float64,也要转成fp16，其余数据类型透传
      auto dy_tmp_out = OutCast<Context>(dev_ctx, &dy_tmp);
      // 如果是float64转换成fp16计算，其余数据类型透传
      auto X_temp_cast = TensorCast<Context>(dev_ctx, x_temp);
      // 如果是float64转换成fp16计算，其余数据类型透传
      auto dout_temp_cast = TensorCast<Context>(dev_ctx, dout_temp);
      aclTensor* x_acltensor = Create_Acltensor(dout_temp_cast, true);
      aclTensor* y_acltensor = Create_Acltensor(X_temp_cast, transpose_x);
      EXEC_NPU_CMD(aclnnMatmul,
                   dev_ctx,
                   x_acltensor,
                   y_acltensor,
                   dy_tmp_out,
                   cube_math_type);
      // 如果是float64,输出结果最后转成float64输出，其余数据类型透传
      OutRevert<Context>(dev_ctx, dy_tmp_out, &dy_tmp);
      // AclnnMatmulForward<T, Context>(dev_ctx, dout, X, dy, true,
      // transpose_x);
    } else {
      dy_tmp = MakeReduceTempOut<T, Context>(dev_ctx,
                                             reduce_dims,
                                             x_temp,
                                             dout_temp,
                                             &dy_resized,
                                             !transpose_x,
                                             false);
      // 如果out是float64,也要转成fp16，其余数据类型透传
      auto dy_tmp_out = OutCast<Context>(dev_ctx, &dy_tmp);
      // 如果是float64转换成fp16计算，其余数据类型透传
      auto X_temp_cast = TensorCast<Context>(dev_ctx, x_temp);
      // 如果是float64转换成fp16计算，其余数据类型透传
      auto dout_temp_cast = TensorCast<Context>(dev_ctx, dout_temp);
      aclTensor* x_acltensor = Create_Acltensor(X_temp_cast, !transpose_x);
      aclTensor* y_acltensor = Create_Acltensor(dout_temp_cast, false);
      EXEC_NPU_CMD(aclnnMatmul,
                   dev_ctx,
                   x_acltensor,
                   y_acltensor,
                   dy_tmp_out,
                   cube_math_type);
      // 如果是float64,输出结果最后转成float64输出，其余数据类型透传
      OutRevert<Context>(dev_ctx, dy_tmp_out, &dy_tmp);
    }
    // reduce场景下，生成对应的reduce tensor，否则透传
    OuttempReduceOut<Context>(dev_ctx, reduce_dims, dy_tmp, &dy_resized);
  }
}

template <typename T, typename Context>
void NPUIdentityKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const int format,
                       phi::DenseTensor* out);

bool IsBaseFormat(const phi::DenseTensor& tensor) {
  auto format = tensor.layout();
  return format == phi::DataLayout::NCHW || format == phi::DataLayout::NCDHW;
}

bool IsNotTransformedNZFormat(const phi::DenseTensor& x,
                              const phi::DenseTensor& y) {
  auto isAligin = [&]() {
    return (!(static_cast<uint64_t>(x.dims()[0]) & 0x0000000F)) &&
           (!(static_cast<uint64_t>(x.dims()[1]) & 0x0000000F)) &&
           (!(static_cast<uint64_t>(y.dims()[0]) & 0x0000000F)) &&
           (!(static_cast<uint64_t>(y.dims()[1]) & 0x0000000F));
  };
  return x.dtype() != phi::DataType::FLOAT16 ||
         y.dtype() != phi::DataType::FLOAT16 || !FLAGS_npu_storage_format ||
         (isAligin() && IsBaseFormat(x) && IsBaseFormat(y));
}

template <typename T, typename Context>
static void MatMulForNZFormat(const Context& dev_ctx,
                              const aclrtStream& stream,
                              const phi::DenseTensor& X,
                              const phi::DenseTensor& Y,
                              phi::DenseTensor* out,
                              const bool transpose_x,
                              const bool transpose_y,
                              bool is_batch) {
  phi::DenseTensor out_tmp;
  phi::DenseTensorMeta meta;
  auto out_dim = out->dims();
  dev_ctx.template Alloc<T>(out);
  VLOG(6) << "Alloc Matmul output in ACL_FORMAT_FRACTAL_NZ format";
  if (out->dims().size() == X.dims().size() - 1) {
    std::vector<int64_t> out_tmp_dims = phi::vectorize(out->dims());
    out_tmp_dims.push_back(1);
    meta = {X.dtype(), phi::make_ddim(out_tmp_dims)};
    out_tmp.set_meta(meta);
  } else {
    meta = {X.dtype(), out->dims()};
    out_tmp.set_meta(meta);
  }
  AllocNPUTensor<T>(dev_ctx, ACL_FORMAT_FRACTAL_NZ, &out_tmp);

  std::function<void(const std::vector<phi::DenseTensor>&,
                     const std::vector<phi::DenseTensor>&,
                     const NPUAttributeMap& attrs,
                     const phi::CustomContext&)>
      functional;
  NPUAttributeMap attr_input = {};
  if (is_batch) {
    attr_input = {{"adj_x1", transpose_x}, {"adj_x2", transpose_y}};
    functional = [&](const std::vector<phi::DenseTensor>& inputs,
                     const std::vector<phi::DenseTensor>& outputs,
                     const NPUAttributeMap& attrs,
                     const phi::CustomContext& dev_ctx) {
      const auto& runner = NpuOpRunner("BatchMatMul", inputs, outputs, attrs);
      runner.Run(dev_ctx.stream());
    };
  } else {
    attr_input = {{"transpose_x1", transpose_x}, {"transpose_x2", transpose_y}};
    functional = [&](const std::vector<phi::DenseTensor>& inputs,
                     const std::vector<phi::DenseTensor>& outputs,
                     const NPUAttributeMap& attrs,
                     const phi::CustomContext& dev_ctx) {
      const auto& runner = NpuOpRunner("MatMul", inputs, outputs, attrs);
      runner.Run(dev_ctx.stream());
    };
  }
  if (X.dtype() == phi::DataType::FLOAT64 &&
      Y.dtype() == phi::DataType::FLOAT64) {
    // To optimize the performace, we transform the datatype from fp64 tp fp16.
    // This is because ascend "matmul" op will transform fp32 to fp16 during
    // actual calculation，
    NpuOpRunner::TypeAdapter({X, Y},
                             {out_tmp},
                             attr_input,
                             dev_ctx,
                             functional,
                             {phi::DataType::FLOAT16, phi::DataType::FLOAT16},
                             {phi::DataType::FLOAT16});
  } else {
    functional({X, Y}, {out_tmp}, attr_input, dev_ctx);
  }
  custom_kernel::NPUIdentityKernel<T, Context>(
      dev_ctx, out_tmp, ConvertToNpuFormat(out->layout()), out);
  out->Resize(out_dim);
}

template <typename T, typename Context>
static void MatMulForNotNZFormat(const Context& dev_ctx,
                                 const aclrtStream& stream,
                                 const phi::DenseTensor& X,
                                 const phi::DenseTensor& Y,
                                 phi::DenseTensor* out,
                                 const bool transpose_x,
                                 const bool transpose_y,
                                 bool is_batch) {
  dev_ctx.template Alloc<T>(out);
  NPUAttributeMap attr_input = {};
  std::function<void(const std::vector<phi::DenseTensor>&,
                     const std::vector<phi::DenseTensor>&,
                     const NPUAttributeMap&,
                     const phi::CustomContext&)>
      functional;
  if (is_batch) {
    attr_input = {{"adj_x1", transpose_x}, {"adj_x2", transpose_y}};
    functional = [&](const std::vector<phi::DenseTensor>& inputs,
                     const std::vector<phi::DenseTensor>& outputs,
                     const NPUAttributeMap& attrs,
                     const phi::CustomContext& dev_ctx) {
      const auto& runner = NpuOpRunner("BatchMatMul", inputs, outputs, attrs);
      runner.Run(dev_ctx.stream());
    };
  } else {
    attr_input = {{"transpose_x1", transpose_x}, {"transpose_x2", transpose_y}};
    functional = [&](const std::vector<phi::DenseTensor>& inputs,
                     const std::vector<phi::DenseTensor>& outputs,
                     const NPUAttributeMap& attrs,
                     const phi::CustomContext& dev_ctx) {
      const auto& runner = NpuOpRunner("MatMul", inputs, outputs, attrs);
      runner.Run(dev_ctx.stream());
    };
  }
  if (X.dtype() == phi::DataType::FLOAT64 &&
      Y.dtype() == phi::DataType::FLOAT64) {
    // To optimize the performace, we transform the datatype from fp64 tp fp16.
    // This is because ascend "matmul" op will transform fp32 to fp16 during
    // actual calculation，
    NpuOpRunner::TypeAdapter({X, Y},
                             {*out},
                             attr_input,
                             dev_ctx,
                             functional,
                             {phi::DataType::FLOAT16, phi::DataType::FLOAT16},
                             {phi::DataType::FLOAT16});
  } else {
    functional({X, Y}, {*out}, attr_input, dev_ctx);
  }
}

template <typename T, typename Context>
static void MatMul(const Context& dev_ctx,
                   const aclrtStream& stream,
                   const phi::DenseTensor& X,
                   const phi::DenseTensor& Y,
                   phi::DenseTensor* out,
                   const bool transpose_x,
                   const bool transpose_y,
                   bool is_batch) {
  if (IsNotTransformedNZFormat(X, Y)) {
    DO_COMPATIBILITY(
        aclnnMatmul,
        (MatMulForNotNZFormat<T>(
            dev_ctx, stream, X, Y, out, transpose_x, transpose_y, is_batch)));
    // MatMulForNotNZFormat<T>(
    //     dev_ctx, stream, X, Y, out, transpose_x, transpose_y, is_batch);
  } else {
    MatMulForNZFormat<T>(
        dev_ctx, stream, X, Y, out, transpose_x, transpose_y, is_batch);
  }
}

template <typename T, typename Context>
static void ReduceDims(const Context& dev_ctx,
                       const aclrtStream& stream,
                       const std::vector<int64_t>& dims,
                       const std::vector<int64_t>& brd_dims,
                       const phi::DenseTensor& in,
                       phi::DenseTensor* out) {
  std::vector<int64_t> axes;
  int64_t size = brd_dims.size();
  int64_t diff = brd_dims.size() - dims.size();
  for (int64_t i = 0; i < size; ++i) {
    if (i < diff) {
      axes.push_back(i);
      continue;
    }
    if (brd_dims[i] > dims[i - diff]) {
      axes.push_back(i);
    }
  }
  dev_ctx.template Alloc<T>(out);
  NpuOpRunner runner;
  runner.SetType("ReduceSum");
  runner.AddInput(in);
  runner.AddInput(dev_ctx, std::move(axes));
  runner.AddOutput(*out);
  runner.AddAttr("keep_dims", false);
  runner.Run(stream);
}

template <typename T, typename Context>
void DotImpl(const Context& dev_ctx,
             const phi::DenseTensor& x,
             const phi::DenseTensor& y,
             phi::DenseTensor* out) {
  auto stream = dev_ctx.stream();
  PADDLE_ENFORCE_EQ(x.numel(),
                    y.numel(),
                    phi::errors::InvalidArgument(
                        "X's numbers must be equal to Y's numbers,"
                        "when X/Y's dims =1. But received X has [%d] elements,"
                        "received Y has [%d] elements",
                        x.numel(),
                        y.numel()));
  out->Resize({1});
  dev_ctx.template Alloc<T>(out);
  if (x.dtype() == phi::DataType::FLOAT64 ||
      y.dtype() == phi::DataType::FLOAT64) {
    auto op_func = [](const std::vector<phi::DenseTensor>& inputs,
                      const std::vector<phi::DenseTensor>& outputs,
                      const NPUAttributeMap& attrs,
                      const phi::CustomContext& dev_ctx) {
      const auto& runner = NpuOpRunner("Dot", inputs, outputs, attrs);
      runner.Run(dev_ctx.stream());
    };
    NpuOpRunner::TypeAdapter({x, y},
                             {*out},
                             {},
                             dev_ctx,
                             op_func,
                             {phi::DataType::FLOAT16, phi::DataType::FLOAT16},
                             {phi::DataType::FLOAT16});
  } else {
    const auto& runner = NpuOpRunner("Dot", {x, y}, {*out});
    runner.Run(stream);
  }
}

template <typename T, typename Context>
void MatmulAclOpImpl(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const phi::DenseTensor& y,
                     bool transpose_x,
                     bool transpose_y,
                     phi::DenseTensor* out) {
  std::vector<int64_t> x_dims = phi::vectorize(x.dims());
  std::vector<int64_t> y_dims = phi::vectorize(y.dims());
  std::vector<int64_t> out_dims = phi::vectorize(out->dims());
  int x_ndim = x_dims.size();
  int y_ndim = y_dims.size();
  int out_ndim = out_dims.size();

  auto stream = dev_ctx.stream();

  // Case 1: [K] x [K] = [1]
  if (x_ndim == 1 && y_ndim == 1) {
    DotImpl<T, Context>(dev_ctx, x, y, out);
    return;
  }
  // Resize dim 1 to 2
  phi::DenseTensor x_temp(x), y_temp(y);
  if (x_ndim == 1) {
    x_dims.insert(x_dims.begin(), 1);
    out_dims.insert(out_dims.end() - 1, 1);
    x_temp.Resize(phi::make_ddim(x_dims));
    x_ndim = 2;
    out_ndim += 1;
  }
  if (y_ndim == 1) {
    y_dims.push_back(1);
    out_dims.push_back(1);
    y_temp.Resize(phi::make_ddim(y_dims));
    y_ndim = 2;
    out_ndim += 1;
  }

  const int K = transpose_x ? x_dims[x_ndim - 2] : x_dims[x_ndim - 1];
  if (transpose_y) {
    PADDLE_ENFORCE_EQ(
        y_dims[y_ndim - 1],
        K,
        phi::errors::InvalidArgument("Input(Y) has error dim."
                                     "Y'dims[%d] must be equal to %d"
                                     "But received Y'dims[%d] is %d",
                                     y_ndim - 1,
                                     K,
                                     y_ndim - 1,
                                     y_dims[y_ndim - 1]));
  } else {
    PADDLE_ENFORCE_EQ(
        y_dims[y_ndim - 2],
        K,
        phi::errors::InvalidArgument("Input(Y) has error dim."
                                     "Y'dims[%d] must be equal to %d"
                                     "But received Y'dims[%d] is %d",
                                     y_ndim - 2,
                                     K,
                                     y_ndim - 2,
                                     y_dims[y_ndim - 2]));
  }
  bool is_batch = false;
  // Case 2: [M, K] x [K, N] = [M, N]
  if (x_ndim == 2 && y_ndim == 2) {
    MatMul<T>(dev_ctx,
              stream,
              x_temp,
              y_temp,
              out,
              transpose_x,
              transpose_y,
              is_batch);
    return;
  }

  // Case 3: [B, M, K] x [K, N] =  [B, M, N], when transpose_x = false
  // Equal: [B * M, K] x [K, N] = [B * M, N] => [B, M, N]
  if (transpose_x == false && y_ndim == 2) {
    std::vector<int64_t> vec_dim = {x_temp.numel() / K, K};
    x_temp.Resize(phi::make_ddim(vec_dim));
    MatMul<T>(dev_ctx,
              stream,
              x_temp,
              y_temp,
              out,
              transpose_x,
              transpose_y,
              is_batch);
    return;
  }

  // Case 4: [B, M, K] x  [B, K, N] = [B, M, N]
  std::vector<int64_t> x_broadcast_dims(out_ndim, 1);
  std::vector<int64_t> y_broadcast_dims(out_ndim, 1);
  std::copy(out_dims.begin(), out_dims.end() - 2, x_broadcast_dims.begin());
  std::copy(out_dims.begin(), out_dims.end() - 2, y_broadcast_dims.begin());
  std::copy(x_dims.end() - 2, x_dims.end(), x_broadcast_dims.end() - 2);
  std::copy(y_dims.end() - 2, y_dims.end(), y_broadcast_dims.end() - 2);

  phi::DenseTensor x_temp_brd;
  phi::DenseTensorMeta x_temp_brd_meta = {x.dtype(), {}};
  x_temp_brd.set_meta(x_temp_brd_meta);
  if (x_dims == x_broadcast_dims) {
    x_temp_brd = x;
    x_temp_brd.Resize(phi::make_ddim(x_broadcast_dims));
  } else {
    x_temp_brd.Resize(phi::make_ddim(x_broadcast_dims));
    dev_ctx.template Alloc<T>(&x_temp_brd);
    NpuOpRunner runner_brd;
    runner_brd.SetType("BroadcastTo")
        .AddInput(x_temp)
        .AddInput(dev_ctx, std::move(x_broadcast_dims))
        .AddOutput(x_temp_brd)
        .Run(stream);
  }

  phi::DenseTensor y_temp_brd;
  phi::DenseTensorMeta y_temp_brd_meta = {y.dtype(), {}};
  y_temp_brd.set_meta(y_temp_brd_meta);
  if (y_dims == y_broadcast_dims) {
    y_temp_brd = y;
    y_temp_brd.Resize(phi::make_ddim(y_broadcast_dims));
  } else {
    y_temp_brd.Resize(phi::make_ddim(y_broadcast_dims));
    dev_ctx.template Alloc<T>(&y_temp_brd);
    NpuOpRunner runner_brd;
    runner_brd.SetType("BroadcastTo")
        .AddInput(y_temp)
        .AddInput(dev_ctx, std::move(y_broadcast_dims))
        .AddOutput(y_temp_brd)
        .Run(stream);
  }
  is_batch = true;
  MatMul<T>(dev_ctx,
            stream,
            x_temp_brd,
            y_temp_brd,
            out,
            transpose_x,
            transpose_y,
            is_batch);
}

template <typename T, typename Context>
void MatmulGradAclOpImpl(const Context& dev_ctx,
                         const phi::DenseTensor& x,
                         const phi::DenseTensor& y,
                         const phi::DenseTensor& dout,
                         bool transpose_x,
                         bool transpose_y,
                         phi::DenseTensor* dx,
                         phi::DenseTensor* dy) {
  std::vector<int64_t> x_dims = phi::vectorize(x.dims());
  std::vector<int64_t> y_dims = phi::vectorize(y.dims());
  std::vector<int64_t> out_dims = phi::vectorize(dout.dims());
  int x_ndim = x_dims.size();
  int y_ndim = y_dims.size();
  int out_ndim = out_dims.size();

  auto stream = dev_ctx.stream();

  // Case 1: [K] x [K] = [1]
  if (x_ndim == 1 && y_ndim == 1) {
    phi::DenseTensor dout_temp;
    phi::DenseTensorMeta dout_temp_meta = {dout.dtype(), x.dims()};
    dout_temp.set_meta(dout_temp_meta);
    dev_ctx.template Alloc<T>(&dout_temp);
    NpuOpRunner runner;
    runner.SetType("BroadcastTo")
        .AddInput(dout)
        .AddInput(dev_ctx, std::move(x_dims))
        .AddOutput(dout_temp)
        .Run(stream);

    if (dx) {
      dev_ctx.template Alloc<T>(dx);
      const auto& runner_dx = NpuOpRunner("Mul", {dout_temp, y}, {*dx}, {});
      runner_dx.Run(stream);
    }
    if (dy) {
      dev_ctx.template Alloc<T>(dy);
      const auto& runner_dy = NpuOpRunner("Mul", {dout_temp, x}, {*dy}, {});
      runner_dy.Run(stream);
    }
    return;
  }
  bool is_batch = false;
  // Resize dim 1 to 2
  phi::DenseTensor x_temp(x), y_temp(y), dout_temp(dout);
  if (x_ndim == 1) {
    x_dims.insert(x_dims.begin(), 1);
    out_dims.insert(out_dims.end() - 1, 1);
    x_temp.Resize(phi::make_ddim(x_dims));
    dout_temp.Resize(phi::make_ddim(out_dims));
    x_ndim = 2;
    out_ndim += 1;
  }
  if (y_ndim == 1) {
    y_dims.push_back(1);
    out_dims.push_back(1);
    y_temp.Resize(phi::make_ddim(y_dims));
    dout_temp.Resize(phi::make_ddim(out_dims));
    y_ndim = 2;
    out_ndim += 1;
  }

  // Case 2: [M, K] x [K, N] = [M, N]
  if (out_ndim == 2) {
    if (dx) {
      dx->Resize(phi::make_ddim(x_dims));
      if (transpose_x) {
        MatMul<T>(dev_ctx,
                  stream,
                  y_temp,
                  dout_temp,
                  dx,
                  transpose_y,
                  true,
                  is_batch);
      } else {
        MatMul<T>(dev_ctx,
                  stream,
                  dout_temp,
                  y_temp,
                  dx,
                  false,
                  !transpose_y,
                  is_batch);
      }
      dx->Resize(x.dims());
    }
    if (dy) {
      dy->Resize(phi::make_ddim(y_dims));
      if (transpose_y) {
        MatMul<T>(dev_ctx,
                  stream,
                  dout_temp,
                  x_temp,
                  dy,
                  true,
                  transpose_x,
                  is_batch);
      } else {
        MatMul<T>(dev_ctx,
                  stream,
                  x_temp,
                  dout_temp,
                  dy,
                  !transpose_x,
                  false,
                  is_batch);
      }
      dy->Resize(y.dims());
    }
    return;
  }

  const int K = transpose_x ? x_dims[x_ndim - 2] : x_dims[x_ndim - 1];
  const int N = transpose_y ? y_dims[y_ndim - 2] : y_dims[y_ndim - 1];
  // Case 3: [B, M, K] x [K, N] =  [B, M, N], when transpose_x = false
  // Equal: [B * M, K] x [K, N] = [B * M, N] => [B, M, N]
  if (transpose_x == false && y_ndim == 2) {
    std::vector<int64_t> x_vec_dim = {x_temp.numel() / K, K};
    dout_temp.Resize(
        phi::make_ddim(std::vector<int64_t>{dout_temp.numel() / N, N}));
    if (dx) {
      dx->Resize(phi::make_ddim(x_vec_dim));
      MatMul<T>(dev_ctx,
                stream,
                dout_temp,
                y_temp,
                dx,
                false,
                !transpose_y,
                is_batch);
      dx->Resize(x.dims());
    }
    if (dy) {
      x_temp.Resize(phi::make_ddim(x_vec_dim));
      if (transpose_y) {
        MatMul<T>(
            dev_ctx, stream, dout_temp, x_temp, dy, true, false, is_batch);
      } else {
        MatMul<T>(
            dev_ctx, stream, x_temp, dout_temp, dy, true, false, is_batch);
      }
    }
    return;
  }

  // Case 4: [B, M, K] x  [B, K, N] = [B, M, N]
  std::vector<int64_t> x_broadcast_dims(out_ndim, 1);
  std::vector<int64_t> y_broadcast_dims(out_ndim, 1);
  std::copy(out_dims.begin(), out_dims.end() - 2, x_broadcast_dims.begin());
  std::copy(out_dims.begin(), out_dims.end() - 2, y_broadcast_dims.begin());
  std::copy(x_dims.end() - 2, x_dims.end(), x_broadcast_dims.end() - 2);
  std::copy(y_dims.end() - 2, y_dims.end(), y_broadcast_dims.end() - 2);

  phi::DenseTensor x_temp_brd;
  phi::DenseTensorMeta x_temp_brd_meta = {x.dtype(), {}};
  x_temp_brd.set_meta(x_temp_brd_meta);
  if (x_dims == x_broadcast_dims) {
    x_temp_brd = x;
    x_temp_brd.Resize(phi::make_ddim(x_broadcast_dims));
  } else {
    x_temp_brd.Resize(phi::make_ddim(x_broadcast_dims));
    dev_ctx.template Alloc<T>(&x_temp_brd);
    NpuOpRunner runner_brd;
    runner_brd.SetType("BroadcastTo")
        .AddInput(x_temp)
        .AddInput(dev_ctx, std::move(x_broadcast_dims))
        .AddOutput(x_temp_brd)
        .Run(stream);
  }

  phi::DenseTensor y_temp_brd;
  phi::DenseTensorMeta y_temp_brd_meta = {y.dtype(), {}};
  y_temp_brd.set_meta(y_temp_brd_meta);
  if (y_dims == y_broadcast_dims) {
    y_temp_brd = y;
    y_temp_brd.Resize(phi::make_ddim(y_broadcast_dims));
  } else {
    y_temp_brd.Resize(phi::make_ddim(y_broadcast_dims));
    dev_ctx.template Alloc<T>(&y_temp_brd);
    NpuOpRunner runner_brd;
    runner_brd.SetType("BroadcastTo")
        .AddInput(y_temp)
        .AddInput(dev_ctx, std::move(y_broadcast_dims))
        .AddOutput(y_temp_brd)
        .Run(stream);
  }
  is_batch = true;
  if (dx) {
    if (x_dims == x_broadcast_dims) {
      if (transpose_x) {
        MatMul<T>(dev_ctx,
                  stream,
                  y_temp_brd,
                  dout_temp,
                  dx,
                  transpose_y,
                  true,
                  is_batch);
      } else {
        MatMul<T>(dev_ctx,
                  stream,
                  dout_temp,
                  y_temp_brd,
                  dx,
                  false,
                  !transpose_y,
                  is_batch);
      }
    } else {
      phi::DenseTensor dx_temp;
      phi::DenseTensorMeta dx_temp_meta = {x.dtype(),
                                           phi::make_ddim(x_broadcast_dims)};
      dx_temp.set_meta(dx_temp_meta);
      if (transpose_x) {
        MatMul<T>(dev_ctx,
                  stream,
                  y_temp_brd,
                  dout_temp,
                  &dx_temp,
                  transpose_y,
                  true,
                  is_batch);
      } else {
        MatMul<T>(dev_ctx,
                  stream,
                  dout_temp,
                  y_temp_brd,
                  &dx_temp,
                  false,
                  !transpose_y,
                  is_batch);
      }
      ReduceDims<T>(dev_ctx, stream, x_dims, x_broadcast_dims, dx_temp, dx);
    }
  }
  if (dy) {
    if (y_dims == y_broadcast_dims) {
      if (transpose_y) {
        MatMul<T>(dev_ctx,
                  stream,
                  dout_temp,
                  x_temp_brd,
                  dy,
                  true,
                  transpose_x,
                  is_batch);
      } else {
        MatMul<T>(dev_ctx,
                  stream,
                  x_temp_brd,
                  dout_temp,
                  dy,
                  !transpose_x,
                  false,
                  is_batch);
      }
    } else {
      phi::DenseTensor dy_temp;
      phi::DenseTensorMeta dy_temp_meta = {y.dtype(),
                                           phi::make_ddim(y_broadcast_dims)};
      dy_temp.set_meta(dy_temp_meta);
      if (transpose_y) {
        MatMul<T>(dev_ctx,
                  stream,
                  dout_temp,
                  x_temp_brd,
                  &dy_temp,
                  true,
                  transpose_x,
                  is_batch);
      } else {
        MatMul<T>(dev_ctx,
                  stream,
                  x_temp_brd,
                  dout_temp,
                  &dy_temp,
                  !transpose_x,
                  false,
                  is_batch);
      }
      ReduceDims<T>(dev_ctx, stream, y_dims, y_broadcast_dims, dy_temp, dy);
    }
  }
}

template <typename T, typename Context>
void MatmulKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::DenseTensor& y,
                  bool transpose_x,
                  bool transpose_y,
                  phi::DenseTensor* out) {
  // 没有aclnn时默认走aclop
  DO_COMPATIBILITY(aclnnMatmul,
                   (MatmulAclOpImpl<T, Context>(
                       dev_ctx, x, y, transpose_x, transpose_y, out)));
  // 私有格式走aclop
  if (x.storage_properties_initialized() ||
      y.storage_properties_initialized()) {
    MatmulAclOpImpl<T, Context>(dev_ctx, x, y, transpose_x, transpose_y, out);
    return;
  }
  AclnnMatmulForward<T, Context>(dev_ctx, x, y, out, transpose_x, transpose_y);
}

template <typename T, typename Context>
void MatmulGradKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      const phi::DenseTensor& y,
                      const phi::DenseTensor& dout,
                      bool transpose_x,
                      bool transpose_y,
                      phi::DenseTensor* dx,
                      phi::DenseTensor* dy) {
  DO_COMPATIBILITY(aclnnMatmul,
                   (MatmulGradAclOpImpl<T, Context>(
                       dev_ctx, x, y, dout, transpose_x, transpose_y, dx, dy)));
  // 私有格式走aclop
  if (x.storage_properties_initialized() ||
      y.storage_properties_initialized()) {
    MatmulGradAclOpImpl<T, Context>(
        dev_ctx, x, y, dout, transpose_x, transpose_y, dx, dy);
    return;
  }
  AclnnMatmulBackward<T, Context>(
      dev_ctx, x, y, dout, transpose_x, transpose_y, dx, dy);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(matmul,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::MatmulKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(matmul_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::MatmulGradKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          double) {}
