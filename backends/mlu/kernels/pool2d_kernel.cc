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

#include "kernels/funcs/mlu_baseop.h"
#include "kernels/funcs/mlu_funcs.h"
#include "runtime/runtime.h"

namespace custom_kernel {

namespace {

cnnlPoolingMode_t ToCnnlPoolingMode(const std::string &pooling_type,
                                    bool exclusive,
                                    bool adaptive) {
  cnnlPoolingMode_t pooling_mode;
  if (pooling_type == "max") {
    pooling_mode = CNNL_POOLING_MAX;
  } else if (pooling_type == "avg") {
    if (exclusive && !adaptive) {
      pooling_mode = CNNL_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
    } else {
      pooling_mode = CNNL_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
    }
  } else {
    PADDLE_THROW(phi::errors::InvalidArgument("Unknown pooling_type: %s",
                                                   pooling_type));
  }
  return pooling_mode;
}
}  // namespace

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
  dev_ctx.template Alloc<T>(out);

  std::vector<int> ksize(kernel_size.GetData().begin(),
                         kernel_size.GetData().end());
  auto strides = strides_t;
  auto paddings = paddings_t;
  const bool channel_last = data_format == "NHWC";
  // default
  cnnlTensorLayout_t cnnl_layout = CNNL_LAYOUT_NCHW;
  auto out_dims = out->dims();
  int64_t out_h = out_dims[2];
  int64_t out_w = out_dims[3];
  auto in_x_dims = in_x.dims();
  phi::DDim data_dims = phi::slice_ddim(in_x_dims, 2, in_x_dims.size());
  
  if (channel_last) {
      cnnl_layout = CNNL_LAYOUT_NHWC;
      out_h = out_dims[1];
      out_w = out_dims[2];
      data_dims = phi::slice_ddim(in_x_dims, 1, in_x_dims.size() - 1);
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
  
  MLUCnnlTensorDesc in_x_desc(in_x, cnnl_layout, ToCnnlDataType<T>());
  MLUCnnlTensorDesc out_desc(*out, cnnl_layout, ToCnnlDataType<T>());
  
  VLOG(4) << "[Pool2d] pooling type: " << pooling_type
          << " exclusive: " << exclusive
          << " adaptive: " << adaptive;
  cnnlPoolingMode_t pool_mode =
      ToCnnlPoolingMode(pooling_type, exclusive, adaptive);
  
  // transpose NCHW to NHWC since cnnl pool2d has worse performance in that
  // layout.
  Tensor trans_in_x;
  Tensor trans_out;
  if (channel_last) {
    trans_in_x = in_x;
    trans_out = *out;
  } else {
    std::vector<int> perm{0, 2, 3, 1};
    TransposeFromMLUTensor<T>(
        dev_ctx, perm, &in_x, &trans_in_x, true /*need_reshape_or_alloc*/);
    phi::DDim trans_out_dims = phi::make_ddim(
        {out_dims[0], out_dims[2], out_dims[3], out_dims[1]});
    trans_out.Resize(trans_out_dims);
    dev_ctx.template Alloc<T>(&trans_out);
  }
  MLUCnnlTensorDesc trans_in_x_desc(
      trans_in_x, CNNL_LAYOUT_NHWC, ToCnnlDataType<T>());
  MLUCnnlTensorDesc trans_out_desc(
      trans_out, CNNL_LAYOUT_NHWC, ToCnnlDataType<T>());
  
  if (!adaptive) {
    MLUCnnlPoolingDesc pool_desc(pool_mode,
                                 CNNL_NOT_PROPAGATE_NAN,
                                 ksize[0],
                                 ksize[1],
                                 paddings[0],
                                 paddings[1],
                                 paddings[2],
                                 paddings[3],
                                 strides[0],
                                 strides[1],
                                 1 /*row_dilation*/,
                                 1 /*col_dilation*/,
                                 ceil_mode);

    size_t extra_input_size = 0;
    cnnlHandle_t handle = GetHandleFromCTX(dev_ctx);
    cnnlGetPoolingExtraInputSize(
        handle, pool_mode, out_w, out_h, &extra_input_size);
    VLOG(4) << "[Pool2d] extra_input_size: " << extra_input_size;

    if (extra_input_size > 0) {
      Tensor extra_host_tensor, extra_device_tensor;
      extra_host_tensor.Resize({static_cast<int64_t>(extra_input_size)});
      extra_device_tensor.Resize({static_cast<int64_t>(extra_input_size)});
      int8_t* h_extra_tensor_ptr = dev_ctx.template HostAlloc<int8_t>(&extra_host_tensor);
      dev_ctx.template Alloc<int8_t>(&extra_device_tensor);

      // Get extra host data by cnnl and then copy to mlu buffer
      cnnlInitPoolingExtraInput(handle,
                                pool_desc.get(),
                                trans_in_x_desc.get(),
                                trans_out_desc.get(),
                                static_cast<void*>(h_extra_tensor_ptr));
      TensorCopy(dev_ctx, extra_host_tensor, false, &extra_device_tensor);
      dev_ctx.Wait();

      MLUCnnl::PoolingForward(
        dev_ctx,
        pool_mode,
        out_h,
        out_w,
        pool_desc.get(),
        nullptr /*alpha*/,
        trans_in_x_desc.get(),
        GetBasePtr(&trans_in_x),
        nullptr /*beta*/,
        GetBasePtr(&extra_device_tensor) /*params_shape_ptr*/,
        trans_out_desc.get(),
        GetBasePtr(&trans_out));
    } else {
    MLUCnnl::PoolingForward(dev_ctx,
                            pool_mode,
                            out_h,
                            out_w,
                            pool_desc.get(),
                            nullptr /*alpha*/,
                            trans_in_x_desc.get(),
                            GetBasePtr(&trans_in_x),
                            nullptr /*beta*/,
                            nullptr /*params_shape_ptr*/,
                            trans_out_desc.get(),
                            GetBasePtr(&trans_out));
    }
  } else {
    MLUCnnl::AdaptivePoolingForward(dev_ctx,
                                    pool_mode,
                                    trans_in_x_desc.get(),
                                    GetBasePtr(&trans_in_x),
                                    trans_out_desc.get(),
                                    GetBasePtr(&trans_out),
                                    nullptr,
                                    nullptr);
  }
  if (!channel_last) {
      std::vector<int> perm{0, 3, 1, 2};
      TransposeFromMLUTensor<T>(
          dev_ctx, perm, &trans_out, out, false /*need_reshape_or_alloc*/);
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
  dev_ctx.template Alloc<T>(in_x_grad);

  std::vector<int> ksize(kernel_size.GetData().begin(),
                         kernel_size.GetData().end());
  auto strides = strides_t;
  auto paddings = paddings_t;

  const bool channel_last = data_format == "NHWC";

  auto in_x_dims = in_x.dims();
  phi::DDim data_dims = phi::slice_ddim(in_x_dims, 2, in_x_dims.size());
  if (channel_last) {
    data_dims = phi::slice_ddim(in_x_dims, 1, in_x_dims.size() - 1);
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
  // inputs need with NHWC layout
  Tensor trans_in_x;
  Tensor trans_out;
  Tensor trans_out_grad;
  Tensor trans_in_x_grad;
  if (channel_last) {
    trans_in_x = in_x;
    trans_out = out;
    trans_out_grad = out_grad;
    trans_in_x_grad = *in_x_grad;
  } else {
    std::vector<int> perm{0, 2, 3, 1};
    TransposeFromMLUTensor<T>(
        dev_ctx, perm, &in_x, &trans_in_x, true /*need_reshape_or_alloc*/);
    TransposeFromMLUTensor<T>(
        dev_ctx, perm, &out, &trans_out, true /*need_reshape_or_alloc*/);
    TransposeFromMLUTensor<T>(
        dev_ctx, perm, &out_grad, &trans_out_grad, true /*need_reshape_or_alloc*/);
    auto in_x_grad_dims = in_x_grad->dims();
    phi::DDim trans_in_grad_dims = phi::make_ddim(
        {in_x_grad_dims[0], in_x_grad_dims[2], in_x_grad_dims[3], in_x_grad_dims[1]});
    trans_in_x_grad.Resize(trans_in_grad_dims);
    dev_ctx.template Alloc<T>(&trans_in_x_grad);
  }
  MLUCnnlTensorDesc trans_in_x_desc(
      trans_in_x, CNNL_LAYOUT_NHWC, ToCnnlDataType<T>());
  MLUCnnlTensorDesc trans_out_desc(
      trans_out, CNNL_LAYOUT_NHWC, ToCnnlDataType<T>());
  MLUCnnlTensorDesc trans_out_grad_desc(
      trans_out_grad, CNNL_LAYOUT_NHWC, ToCnnlDataType<T>());
  MLUCnnlTensorDesc trans_in_x_grad_desc(
      trans_in_x_grad, CNNL_LAYOUT_NHWC, ToCnnlDataType<T>());  
  cnnlPoolingMode_t pool_mode =
      ToCnnlPoolingMode(pooling_type, exclusive, adaptive);
  MLUCnnlPoolingDesc pool_desc(pool_mode,
                               CNNL_NOT_PROPAGATE_NAN,
                               ksize[0],
                               ksize[1],
                               paddings[0],
                               paddings[1],
                               paddings[2],
                               paddings[3],
                               strides[0],
                               strides[1],
                               1 /*row_dilation*/,
                               1 /*col_dilation*/,
                               ceil_mode);  
  if (pooling_type == "max") {
    Tensor index_tensor;
    index_tensor.Resize(trans_out_grad.dims());
    dev_ctx.template Alloc<int>(&index_tensor);
    MLUCnnlTensorDesc index_tensor_desc(
        index_tensor, CNNL_LAYOUT_NHWC, ToCnnlDataType<int>());
    MLUCnnl::PoolingIndex(dev_ctx,
                          pool_desc.get(),
                          trans_in_x_desc.get(),
                          GetBasePtr(&trans_in_x),
                          index_tensor_desc.get(),
                          GetBasePtr(&index_tensor));
    if (adaptive) {
      MLUCnnl::AdaptivePoolingBackward(dev_ctx,
                                       pool_mode,
                                       trans_out_grad_desc.get(),
                                       GetBasePtr(&trans_out_grad),
                                       index_tensor_desc.get(),
                                       GetBasePtr(&index_tensor),
                                       trans_in_x_grad_desc.get(),
                                       GetBasePtr(&trans_in_x_grad));
    } else {
      MLUCnnl::PoolingBackward(dev_ctx,
                               pool_desc.get(),
                               nullptr /*alpha*/,
                               index_tensor_desc.get(),
                               GetBasePtr(&index_tensor),
                               trans_out_grad_desc.get(),
                               GetBasePtr(&trans_out_grad),
                               trans_in_x_desc.get(),
                               GetBasePtr(&trans_in_x),
                               nullptr /*beta*/,
                               trans_in_x_grad_desc.get(),
                               GetBasePtr(&trans_in_x_grad));
    }
  } else {
    if (adaptive) {
      MLUCnnl::AdaptivePoolingBackward(dev_ctx,
                                       pool_mode,
                                       trans_out_grad_desc.get(),
                                       GetBasePtr(&trans_out_grad),
                                       nullptr /*index_tensor_desc.get()*/,
                                       nullptr /*GetBasePtr(&index_tensor)*/,
                                       trans_in_x_grad_desc.get(),
                                       GetBasePtr(&trans_in_x_grad));
    } else {
      MLUCnnl::PoolingBackward(dev_ctx,
                               pool_desc.get(),
                               nullptr /*alpha*/,
                               nullptr,
                               nullptr,
                               trans_out_grad_desc.get(),
                               GetBasePtr(&trans_out_grad),
                               nullptr,
                               nullptr,
                               nullptr /*beta*/,
                               trans_in_x_grad_desc.get(),
                               GetBasePtr(&trans_in_x_grad));
    }
  }
  if (!channel_last) {
    std::vector<int> perm{0, 3, 1, 2};
    TransposeFromMLUTensor<T>(dev_ctx,
                              perm,
                              &trans_in_x_grad,
                              in_x_grad,
                              false /*need_reshape_or_alloc*/);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(pool2d,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::Pool2dKernel,
                          float,
                          phi::dtype::float16) {}
PD_REGISTER_PLUGIN_KERNEL(pool2d_grad,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::Pool2dGradKernel,
                          float,
                          phi::dtype::float16) {}
