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
    case 3:
      format = ACL_FORMAT_NCL;
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

template <typename T, typename Context>
static void AclnnMatmulForward(const Context& dev_ctx,
                               const phi::DenseTensor& X,
                               const phi::DenseTensor& Y,
                               phi::DenseTensor* out,
                               const bool transpose_x,
                               const bool transpose_y) {
  dev_ctx.template Alloc<T>(out);
  auto out_dim = phi::vectorize(out->dims());
  aclTensor* x_acltensor = Create_Acltensor(X, transpose_x);
  aclTensor* y_acltensor = Create_Acltensor(Y, transpose_y);
  int8_t cube_math_type = 0;
  EXEC_NPU_CMD(
      aclnnMatmul, dev_ctx, x_acltensor, y_acltensor, *out, cube_math_type);
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
  std::vector<int64_t> x_dims = phi::vectorize(X.dims());
  std::vector<int64_t> y_dims = phi::vectorize(Y.dims());
  std::vector<int64_t> out_dims = phi::vectorize(dout.dims());
  int x_ndim = x_dims.size();
  int y_ndim = y_dims.size();
  int out_ndim = out_dims.size();
  // Case 1: [K] x [K] = [1]
  if (x_ndim == 1 && y_ndim == 1) {
    if (dx) {
      dev_ctx.template Alloc<T>(dx);
      EXEC_NPU_CMD(aclnnMul, dev_ctx, dout, Y, *dx);
    }
    if (dy) {
      dev_ctx.template Alloc<T>(dy);
      EXEC_NPU_CMD(aclnnMul, dev_ctx, dout, X, *dy);
    }
    return;
  }
  if (dx) {
    if (transpose_x) {
      AclnnMatmulForward<T, Context>(dev_ctx, Y, dout, dx, transpose_y, true);
    } else {
      AclnnMatmulForward<T, Context>(dev_ctx, dout, Y, dx, false, !transpose_y);
    }
    dx->Resize(X.dims());
  }
  if (dy) {
    if (transpose_y) {
      AclnnMatmulForward<T, Context>(dev_ctx, dout, X, dy, true, transpose_x);
    } else {
      AclnnMatmulForward<T, Context>(dev_ctx, X, dout, dy, !transpose_x, false);
    }
    dy->Resize(Y.dims());
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
    MatMulForNotNZFormat<T>(
        dev_ctx, stream, X, Y, out, transpose_x, transpose_y, is_batch);
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
  bool is_nz = IsNotTransformedNZFormat(x, y);
  // NZ私有格式走ACLOP
  if (is_nz) {
    MatmulAclOpImpl<T, Context>(dev_ctx, x, y, transpose_x, transpose_y, out);
    return;
  }
  // ACLNN调用
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
  // 没有aclnn时默认走aclop
  DO_COMPATIBILITY(aclnnMatmul,
                   (MatmulGradAclOpImpl<T, Context>(
                       dev_ctx, x, y, dout, transpose_x, transpose_y, dx, dy)));
  // NZ私有格式走ACLOP
  bool is_nz = IsNotTransformedNZFormat(x, y);
  if (is_nz) {
    MatmulGradAclOpImpl<T, Context>(
        dev_ctx, x, y, dout, transpose_x, transpose_y, dx, dy);
    return;
  }
  // ACLNN调用
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
