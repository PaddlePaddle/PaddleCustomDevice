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
#include <vector>

#include "kernels/funcs/sdaa_baseop.h"
#include "paddle/phi/extension.h"

namespace custom_kernel {

template <typename T = int>
inline void UpdatePadding(std::vector<T>* paddings,
                          const bool global_pooling,
                          const bool adaptive,
                          const std::string padding_algorithm,
                          const phi::DDim data_dims,
                          const std::vector<T>& strides,
                          const std::vector<T>& kernel_size) {
  // set padding size == data_dims.size() * 2
  auto data_shape = phi::vectorize<T>(data_dims);
  if (static_cast<int>(paddings->size()) == data_dims.size()) {
    for (int i = 0; i < data_dims.size(); ++i) {
      T copy_pad = *(paddings->begin() + 2 * i);
      paddings->insert(paddings->begin() + 2 * i + 1, copy_pad);
    }
  } else {
    PADDLE_ENFORCE_EQ(data_dims.size() * 2,
                      paddings->size(),
                      phi::errors::InvalidArgument(
                          "Paddings size %d should be the same or twice as the "
                          "pooling size %d.",
                          paddings->size(),
                          data_dims.size() * 2));
  }

  // when padding_algorithm is "VALID" or "SAME"
  if (padding_algorithm == "SAME") {
    for (int i = 0; i < data_dims.size(); ++i) {
      T out_size = (data_dims[i] + strides[i] - 1) / strides[i];
      T pad_sum =
          std::max((out_size - 1) * strides[i] + kernel_size[i] - data_shape[i],
                   static_cast<T>(0));
      T pad_0 = pad_sum / 2;
      T pad_1 = pad_sum - pad_0;
      *(paddings->begin() + i * 2) = pad_0;
      *(paddings->begin() + i * 2 + 1) = pad_1;
    }
  } else if (padding_algorithm == "VALID") {
    for (auto it = paddings->begin(); it != paddings->end(); it++) {
      *it = 0;
    }
  }

  // if global_pooling == true or adaptive == true, padding will be ignore
  if (global_pooling || adaptive) {
    for (auto it = paddings->begin(); it != paddings->end(); it++) {
      *it = 0;
    }
  }
}

template <typename T = int>
inline void UpdateKernelSize(std::vector<T>* kernel_size,
                             const phi::DDim data_dims) {
  kernel_size->resize(static_cast<size_t>(data_dims.size()));
  for (size_t i = 0; i < kernel_size->size(); ++i) {
    *(kernel_size->begin() + i) = static_cast<T>(data_dims[i]);
  }
}

tecodnnAdaptivePoolingMode_t GetTecodnnAdaptivePoolingMode(
    const std::string& pooling_type) {
  tecodnnAdaptivePoolingMode_t AdaptivePoolingMode;
  if (pooling_type == "max") {
    AdaptivePoolingMode = TECODNN_ADAPTIVE_POOLING_MAX;
  } else {
    AdaptivePoolingMode = TECODNN_ADAPTIVE_POOLING_AVG;
  }

  return AdaptivePoolingMode;
}

tecodnnPoolingMode_t GetTecodnnPoolingMode(const std::string& pooling_type,
                                           bool exclusive,
                                           bool ceil_mode = false) {
  if (pooling_type == "max") {
    PADDLE_ENFORCE_EQ(
        exclusive,
        true,
        phi::errors::InvalidArgument(
            "MaxPool only support exclusive==false, but got true"));
    PADDLE_ENFORCE_EQ(
        ceil_mode,
        false,
        phi::errors::InvalidArgument(
            "MaxPool in sdaa only support ceil_mode==False, but got true."));
    return TECODNN_POOLING_MAX;
  } else if (pooling_type == "avg") {
    tecodnnPoolingMode_t pooling_mode;
    if (ceil_mode) {
      PADDLE_ENFORCE_EQ(
          exclusive,
          true,
          phi::errors::InvalidArgument(
              "tecodnn not support ceil_mode=true when exclusive=false."));
      pooling_mode = TECODNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING_CEIL_TRUE;
    } else {
      pooling_mode = exclusive ? TECODNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING
                               : TECODNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
    }
    return pooling_mode;
  } else {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "Pooling mode error, check whether pooling type is correct."));
  }
}

/*The tensor format of this function must be NHWC*/
template <typename T, typename Context>
void doPoolingForward(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      const std::string& pooling_type,
                      const std::vector<int>& pool2dParameters,
                      bool adaptive,
                      bool exclusive,
                      bool ceil_mode,
                      phi::DenseTensor* out) {
  std::vector<int> x_dims = phi::vectorize<int>(x.dims());
  std::vector<int> out_dims = phi::vectorize<int>(out->dims());

  tecodnnPoolingDescriptor_t pooling_Desc;
  tecodnnPoolingMode_t PoolingMode;
  tecodnnAdaptivePoolingMode_t adaptivePoolingMode;

  if (adaptive) {
    adaptivePoolingMode = GetTecodnnAdaptivePoolingMode(pooling_type);
  } else {
    PoolingMode = GetTecodnnPoolingMode(pooling_type, exclusive, ceil_mode);
  }

  tecodnnHandle_t tecodnn_handle = GetHandleFromCTX(dev_ctx);

  if (!adaptive) {
    TECODNN_CHECK(tecodnnCreatePoolingDescriptor(&pooling_Desc));
  }

  tecodnnTensorDescriptor_t x_Desc =
      sdaa_ops::GetTecodnnTensorDesc(x_dims, x.dtype(), TensorFormat::NHWC);
  tecodnnTensorDescriptor_t out_Desc = sdaa_ops::GetTecodnnTensorDesc(
      out_dims, out->dtype(), TensorFormat::NHWC);

  if (!adaptive) {
    TECODNN_CHECK(tecodnnSetPooling2dDescriptor(pooling_Desc,
                                                PoolingMode,
                                                TECODNN_NOT_PROPAGATE_NAN,
                                                pool2dParameters[0],
                                                pool2dParameters[1],
                                                pool2dParameters[4],
                                                pool2dParameters[5],
                                                pool2dParameters[2],
                                                pool2dParameters[3]));
  }

  const float alpha = 1.0f, beta = 0.0f;
  if (!adaptive) {
    TECODNN_CHECK(tecodnnPoolingForward(tecodnn_handle,
                                        pooling_Desc,
                                        &alpha,
                                        x_Desc,
                                        x.data(),
                                        &beta,
                                        out_Desc,
                                        out->data()));
  } else {
    TECODNN_CHECK(tecodnnAdaptivePoolingForward(tecodnn_handle,
                                                adaptivePoolingMode,
                                                &alpha,
                                                x_Desc,
                                                x.data(),
                                                &beta,
                                                out_Desc,
                                                out->data()));
  }

  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(out_Desc));
  if (!adaptive) {
    TECODNN_CHECK(tecodnnDestroyPoolingDescriptor(pooling_Desc));
  }
}

/*The tensor format of this function must be NHWC*/
template <typename T, typename Context>
void doPoolingBackward(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::DenseTensor& out,
                       const phi::DenseTensor& out_grad,
                       const std::string& pooling_type,
                       const std::vector<int>& pool2dParameters,
                       bool adaptive,
                       bool exclusive,
                       bool ceil_mode,
                       phi::DenseTensor* x_grad) {
  std::vector<int> x_dims = phi::vectorize<int>(x.dims());
  std::vector<int> out_dims = phi::vectorize<int>(out.dims());
  std::vector<int> out_grad_dims = phi::vectorize<int>(out_grad.dims());
  std::vector<int> in_x_grad_dims = phi::vectorize<int>(x_grad->dims());

  tecodnnPoolingDescriptor_t pooling_Desc;
  tecodnnPoolingMode_t PoolingMode;
  tecodnnAdaptivePoolingMode_t adaptivePoolingMode;

  if (adaptive) {
    adaptivePoolingMode = GetTecodnnAdaptivePoolingMode(pooling_type);
  } else {
    PoolingMode = GetTecodnnPoolingMode(pooling_type, exclusive, ceil_mode);
  }

  tecodnnHandle_t tecodnn_handle = GetHandleFromCTX(dev_ctx);

  if (!adaptive) {
    TECODNN_CHECK(tecodnnCreatePoolingDescriptor(&pooling_Desc));
  }

  tecodnnTensorDescriptor_t x_Desc =
      sdaa_ops::GetTecodnnTensorDesc(x_dims, x.dtype(), TensorFormat::NHWC);
  tecodnnTensorDescriptor_t out_Desc =
      sdaa_ops::GetTecodnnTensorDesc(out_dims, out.dtype(), TensorFormat::NHWC);
  tecodnnTensorDescriptor_t out_grad_Desc = sdaa_ops::GetTecodnnTensorDesc(
      out_grad_dims, out_grad.dtype(), TensorFormat::NHWC);
  tecodnnTensorDescriptor_t x_grad_Desc = sdaa_ops::GetTecodnnTensorDesc(
      in_x_grad_dims, x_grad->dtype(), TensorFormat::NHWC);

  if (!adaptive) {
    TECODNN_CHECK(tecodnnSetPooling2dDescriptor(pooling_Desc,
                                                PoolingMode,
                                                TECODNN_NOT_PROPAGATE_NAN,
                                                pool2dParameters[0],
                                                pool2dParameters[1],
                                                pool2dParameters[4],
                                                pool2dParameters[5],
                                                pool2dParameters[2],
                                                pool2dParameters[3]));
  }

  const float alpha = 1.0f, beta = 0.0f;
  if (!adaptive) {
    TECODNN_CHECK(tecodnnPoolingBackward(tecodnn_handle,
                                         pooling_Desc,
                                         &alpha,
                                         out_Desc,
                                         out.data(),
                                         out_grad_Desc,
                                         out_grad.data(),
                                         x_Desc,
                                         x.data(),
                                         &beta,
                                         x_grad_Desc,
                                         x_grad->data()));
  } else {
    TECODNN_CHECK(tecodnnAdaptivePoolingBackward(tecodnn_handle,
                                                 adaptivePoolingMode,
                                                 &alpha,
                                                 out_grad_Desc,
                                                 out_grad.data(),
                                                 x_Desc,
                                                 x.data(),
                                                 &beta,
                                                 x_grad_Desc,
                                                 x_grad->data()));
  }

  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_grad_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(out_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(out_grad_Desc));
  if (!adaptive) {
    TECODNN_CHECK(tecodnnDestroyPoolingDescriptor(pooling_Desc));
  }
}

template <typename T, typename Context>
void Pool2dKernel(const Context& dev_ctx,
                  const phi::DenseTensor& in_x,
                  const phi::IntArray& kernel_size,
                  const std::vector<int>& strides_t,
                  const std::vector<int>& paddings_t,
                  bool ceil_mode,
                  bool exclusive,
                  const std::string& data_format,
                  const std::string& pooling_type,
                  bool global_pooling,
                  bool adaptive,
                  const std::string& padding_algorithm,
                  phi::DenseTensor* out) {
  VLOG(4) << "CALL SDAA Pool2dKernel";

  dev_ctx.template Alloc<T>(out);

  if (ceil_mode && !exclusive) {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "tecodnn not support ceil_mode=true when exclusive=false."));
  }

  std::vector<int> ksize(kernel_size.GetData().begin(),
                         kernel_size.GetData().end());
  auto strides = strides_t;
  auto paddings = paddings_t;
  bool channel_last = data_format == "NHWC";

  auto in_x_dims = in_x.dims();
  auto out_dims = out->dims();
  phi::DDim data_dims;

  bool transfer2NHWC = false;

  if (channel_last) {
    // NHWC
    data_dims =
        phi::slice_ddim(in_x_dims, 1, in_x_dims.size() - 1);  // require H, W
  } else {
    // NCHW
    data_dims =
        phi::slice_ddim(in_x_dims, 2, in_x_dims.size());  // require H, W
    transfer2NHWC = true;
  }
  UpdatePadding(&paddings,
                global_pooling,
                adaptive,
                padding_algorithm,
                data_dims,
                strides,
                ksize);

  if (global_pooling) {
    UpdateKernelSize(&ksize, data_dims);
  }

  int kernel_H = ksize[0];
  int kernel_W = ksize[1];
  int stride_H = strides[0];
  int stride_W = strides[1];

  int pad_H = 0, pad_W = 0;
  if (paddings.size() == 2) {
    pad_H = paddings[0];
    pad_W = paddings[1];
  } else if (paddings.size() == 4) {
    if (paddings[0] != paddings[1]) {
      PADDLE_ENFORCE_EQ(
          paddings[0],
          paddings[1],
          phi::errors::InvalidArgument("padding_height_top size should be the "
                                       "same as padding_height_bottom."
                                       "But recieved: padding_height_top size "
                                       "is %d, padding_height_bottom is %d",
                                       paddings[0],
                                       paddings[1]));
    } else if (paddings[2] != paddings[3]) {
      PADDLE_ENFORCE_EQ(
          paddings[2],
          paddings[3],
          phi::errors::InvalidArgument("padding_width_left size should be the "
                                       "same as padding_width_right."
                                       "But recieved: padding_width_left size "
                                       "is %d, padding_width_right is %d",
                                       paddings[2],
                                       paddings[3]));
    } else {
      pad_H = paddings[0];
      pad_W = paddings[2];
    }
  } else {
    PADDLE_ENFORCE_EQ(1,
                      0,
                      phi::errors::InvalidArgument(
                          "tecodnn only support padding size equal to 2 or 4."
                          "But recived: padding size is %d",
                          paddings.size()));
  }

  std::vector<int> pool2dParameters = {
      kernel_H, kernel_W, stride_H, stride_W, pad_H, pad_W};

  if (transfer2NHWC) {
    // allocate memory for NHWC tensorformat, needs to dims permute
    phi::DDim in_x_NHWC_dims =
        sdaa_ops::doDimPermute(in_x, Convert_TF::NCHW2NHWC);
    phi::DDim out_NHWC_dims =
        sdaa_ops::doDimPermute(*out, Convert_TF::NCHW2NHWC);

    phi::DenseTensor in_x_NHWC, out_NHWC;
    phi::DenseTensorMeta in_x_NHWC_meta = {in_x.dtype(), in_x_NHWC_dims};
    phi::DenseTensorMeta out_NHWC_meta = {out->dtype(), out_NHWC_dims};
    in_x_NHWC.set_meta(in_x_NHWC_meta);
    out_NHWC.set_meta(out_NHWC_meta);

    dev_ctx.template Alloc<T>(&in_x_NHWC);
    dev_ctx.template Alloc<T>(&out_NHWC);

    sdaa_ops::doTransformTensor(
        dev_ctx, in_x, Convert_TF::NCHW2NHWC, &in_x_NHWC);

    doPoolingForward<T, Context>(dev_ctx,
                                 in_x_NHWC,
                                 pooling_type,
                                 pool2dParameters,
                                 adaptive,
                                 exclusive,
                                 ceil_mode,
                                 &out_NHWC);

    sdaa_ops::doTransformTensor(dev_ctx, out_NHWC, Convert_TF::NHWC2NCHW, out);
  } else {
    doPoolingForward<T, Context>(dev_ctx,
                                 in_x,
                                 pooling_type,
                                 pool2dParameters,
                                 adaptive,
                                 exclusive,
                                 ceil_mode,
                                 out);
  }
}

template <typename T, typename Context>
void Pool2dGradKernel(const Context& dev_ctx,
                      const phi::DenseTensor& in_x,
                      const phi::DenseTensor& out,
                      const phi::DenseTensor& out_grad,
                      const phi::IntArray& kernel_size,
                      const std::vector<int>& strides_t,
                      const std::vector<int>& paddings_t,
                      bool ceil_mode,
                      bool exclusive,
                      const std::string& data_format,
                      const std::string& pooling_type,
                      bool global_pooling,
                      bool adaptive,
                      const std::string& padding_algorithm,
                      phi::DenseTensor* in_x_grad) {
  VLOG(4) << "CALL SDAA Pool2dGradKernel";

  dev_ctx.template Alloc<T>(in_x_grad);

  if (ceil_mode && !exclusive) {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "tecodnn not support ceil_mode=true when exclusive=false."));
  }

  std::vector<int> ksize(kernel_size.GetData().begin(),
                         kernel_size.GetData().end());
  auto strides = strides_t;
  auto paddings = paddings_t;

  bool channel_last = data_format == "NHWC";

  // updata paddings
  auto in_x_dims = in_x.dims();
  auto out_dims = out.dims();
  phi::DDim data_dims;
  phi::DDim out_data_dims;

  bool transfer2NHWC = false;

  if (channel_last) {
    data_dims = phi::slice_ddim(in_x_dims, 1, in_x_dims.size() - 1);
    out_data_dims = phi::slice_ddim(out_dims, 1, out_dims.size() - 1);
  } else {
    data_dims = phi::slice_ddim(in_x_dims, 2, in_x_dims.size());
    out_data_dims = phi::slice_ddim(out_dims, 2, out_dims.size());
    transfer2NHWC = true;
  }

  UpdatePadding(&paddings,
                global_pooling,
                adaptive,
                padding_algorithm,
                data_dims,
                strides,
                ksize);

  if (global_pooling) {
    UpdateKernelSize(&ksize, data_dims);
  }

  int kernel_H = ksize[0];
  int kernel_W = ksize[1];
  int stride_H = strides[0];
  int stride_W = strides[1];

  int pad_H = 0, pad_W = 0;
  if (paddings.size() == 2) {
    pad_H = paddings[0];
    pad_W = paddings[1];
  } else if (paddings.size() == 4) {
    if (paddings[0] != paddings[1]) {
      PADDLE_ENFORCE_EQ(
          paddings[0],
          paddings[1],
          phi::errors::InvalidArgument("padding_height_top size should be the "
                                       "same as padding_height_bottom."
                                       "But recieved: padding_height_top size "
                                       "is %d, padding_height_bottom is %d",
                                       paddings[0],
                                       paddings[1]));
    } else if (paddings[2] != paddings[3]) {
      PADDLE_ENFORCE_EQ(
          paddings[2],
          paddings[3],
          phi::errors::InvalidArgument("padding_width_left size should be the "
                                       "same as padding_width_right."
                                       "But recieved: padding_width_left size "
                                       "is %d, padding_width_right is %d",
                                       paddings[2],
                                       paddings[3]));
    } else {
      pad_H = paddings[0];
      pad_W = paddings[2];
    }
  } else {
    PADDLE_ENFORCE_EQ(1,
                      0,
                      phi::errors::InvalidArgument(
                          "tecodnn only support padding size equal to 2 or 4."
                          "But recived: padding size is %d",
                          paddings.size()));
  }

  std::vector<int> pool2dParameters = {
      kernel_H, kernel_W, stride_H, stride_W, pad_H, pad_W};

  if (transfer2NHWC) {
    // allocate memory for NHWC tensorformat, need to dims permute
    phi::DDim in_x_NHWC_dims =
        sdaa_ops::doDimPermute(in_x, Convert_TF::NCHW2NHWC);
    phi::DDim out_NHWC_dims =
        sdaa_ops::doDimPermute(out, Convert_TF::NCHW2NHWC);
    phi::DDim out_grad_NHWC_dims =
        sdaa_ops::doDimPermute(out_grad, Convert_TF::NCHW2NHWC);
    phi::DDim in_x_grad_NHWC_dims =
        sdaa_ops::doDimPermute(*in_x_grad, Convert_TF::NCHW2NHWC);

    phi::DenseTensor in_x_NHWC, in_x_grad_NHWC, out_NHWC, out_grad_NHWC;
    phi::DenseTensorMeta in_x_NHWC_meta = {in_x.dtype(), in_x_NHWC_dims};
    phi::DenseTensorMeta out_NHWC_meta = {out_NHWC.dtype(), out_NHWC_dims};
    phi::DenseTensorMeta out_grad_NHWC_meta = {out_grad.dtype(),
                                               out_grad_NHWC_dims};
    phi::DenseTensorMeta in_x_grad_NHWC_meta = {in_x_grad->dtype(),
                                                in_x_grad_NHWC_dims};

    in_x_NHWC.set_meta(in_x_NHWC_meta);
    out_NHWC.set_meta(out_NHWC_meta);
    out_grad_NHWC.set_meta(out_grad_NHWC_meta);
    in_x_grad_NHWC.set_meta(in_x_grad_NHWC_meta);

    dev_ctx.template Alloc<T>(&in_x_NHWC);
    dev_ctx.template Alloc<T>(&in_x_grad_NHWC);
    dev_ctx.template Alloc<T>(&out_NHWC);
    dev_ctx.template Alloc<T>(&out_grad_NHWC);

    sdaa_ops::doTransformTensor(
        dev_ctx, in_x, Convert_TF::NCHW2NHWC, &in_x_NHWC);
    sdaa_ops::doTransformTensor(dev_ctx, out, Convert_TF::NCHW2NHWC, &out_NHWC);
    sdaa_ops::doTransformTensor(
        dev_ctx, out_grad, Convert_TF::NCHW2NHWC, &out_grad_NHWC);
    doPoolingBackward<T, Context>(dev_ctx,
                                  in_x_NHWC,
                                  out_NHWC,
                                  out_grad_NHWC,
                                  pooling_type,
                                  pool2dParameters,
                                  adaptive,
                                  exclusive,
                                  ceil_mode,
                                  &in_x_grad_NHWC);

    sdaa_ops::doTransformTensor(
        dev_ctx, in_x_grad_NHWC, Convert_TF::NHWC2NCHW, in_x_grad);
  } else {
    doPoolingBackward<T, Context>(dev_ctx,
                                  in_x,
                                  out,
                                  out_grad,
                                  pooling_type,
                                  pool2dParameters,
                                  adaptive,
                                  exclusive,
                                  ceil_mode,
                                  in_x_grad);
  }
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(pool2d,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::Pool2dKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(pool2d_grad,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::Pool2dGradKernel,
                          float,
                          phi::dtype::float16) {}
