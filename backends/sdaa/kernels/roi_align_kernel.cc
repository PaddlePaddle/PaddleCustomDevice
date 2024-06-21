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

#include <vector>

#include "kernels/funcs/sdaa_baseop.h"
#include "paddle/phi/extension.h"

namespace custom_kernel {

template <typename T, typename Context>
void RoiAlignKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& boxes,
                    const paddle::optional<phi::DenseTensor>& boxes_num,
                    int pooled_height,
                    int pooled_width,
                    float spatial_scale,
                    int sampling_ratio,
                    bool aligned,
                    phi::DenseTensor* out) {
  if (boxes.dims()[0] == 0) {
    dev_ctx.template Alloc<T>(out);
    return;
  }

  PADDLE_ENFORCE_LT(
      sampling_ratio,
      1024,
      phi::errors::InvalidArgument(
          "sampling_ratio must be less than 1024, but provided is %d",
          sampling_ratio));

  dev_ctx.template Alloc<T>(out);

  int boxes_batch_size;
  std::vector<T> roi_batch_id_data((boxes.dims()[0]));
  int batch_size = x.dims()[0];

  if (boxes_num) {
    boxes_batch_size = boxes_num->numel();
    PADDLE_ENFORCE_EQ(
        boxes_batch_size,
        batch_size,
        phi::errors::InvalidArgument(
            "The batch size of rois and the batch size of images "
            " must be the same. But received the batch size of rois is %d, "
            "and the batch size of images is %d",
            boxes_batch_size,
            batch_size));

    std::vector<int> boxes_num_data;
    auto& boxes_num_tensor = *boxes_num.get_ptr();
    TensorToVector(dev_ctx, boxes_num_tensor, dev_ctx, &boxes_num_data);
    int start = 0;
    // transfrom boxes_num to roi_batch_id_data
    for (int n = 0; n < boxes_batch_size; ++n) {
      for (int i = start; i < start + boxes_num_data[n]; ++i) {
        roi_batch_id_data[i] = static_cast<float>(n);
      }
      start += boxes_num_data[n];
    }
  } else {
    auto lod = boxes.lod();
    PADDLE_ENFORCE_EQ(
        lod.empty(),
        false,
        phi::errors::InvalidArgument("Input(ROIs) Tensor of ROIAlignOp "
                                     "does not contain LoD information."));
    auto boxes_lod = lod.back();
    boxes_batch_size = boxes_lod.size() - 1;
    PADDLE_ENFORCE_EQ(
        boxes_batch_size,
        batch_size,
        phi::errors::InvalidArgument(
            "The boxes_batch_size and imgs "
            "batch_size must be the same. But received boxes_batch_size = %d, "
            "batch_size = %d",
            boxes_batch_size,
            batch_size));
    int boxes_num_with_lod = boxes_lod[boxes_batch_size];
    PADDLE_ENFORCE_EQ(
        boxes.dims()[0],
        boxes_num_with_lod,
        phi::errors::InvalidArgument(
            "The actual number of rois and the number of rois "
            "provided from Input(RoIsLoD) in RoIAlign must be the same."
            " But received actual number of rois is %d, and the number "
            "of rois from RoIsLoD is %d",
            boxes.dims()[0],
            boxes_num_with_lod));
    for (int n = 0; n < boxes_batch_size; ++n) {
      for (std::size_t i = boxes_lod[n]; i < boxes_lod[n + 1]; ++i) {
        roi_batch_id_data[i] = n;
      }
    }
  }

  phi::DenseTensor boxes_num_t;
  TensorFromVector<T>(dev_ctx, roi_batch_id_data, dev_ctx, &boxes_num_t);
  boxes_num_t.Resize({boxes.dims()[0], 1});

  // x and boxes must be the same dtype
  phi::DenseTensor boxes_t;
  if (boxes.dtype() == x.dtype()) {
    boxes_t = boxes;
  } else {
    boxes_t.Resize(boxes.dims());
    dev_ctx.Alloc(&boxes_t, x.dtype());
    sdaa_ops::doCastTensor(dev_ctx, boxes, &boxes_t);
  }

  std::vector<const phi::DenseTensor*> boxes_list;
  boxes_list.emplace_back(&boxes_num_t);
  boxes_list.emplace_back(&boxes_t);
  phi::DenseTensor boxes_N5;
  boxes_N5.Resize({boxes.dims()[0], 5});
  dev_ctx.template Alloc<T>(&boxes_N5);

  // concat boxes and rois location to [N,5] tensor
  int axis = 1;
  sdaa_ops::doConcatTensor(dev_ctx, boxes_list, axis, &boxes_N5);

  // tecodnnRoiAlignForward only support input and output of NHWC format
  phi::DDim x_t_dims = sdaa_ops::doDimPermute(x, Convert_TF::NCHW2NHWC);
  phi::DDim out_t_dims = sdaa_ops::doDimPermute(*out, Convert_TF::NCHW2NHWC);
  phi::DenseTensor x_t, out_t;
  phi::DenseTensorMeta x_t_meta = {x.dtype(), x_t_dims, phi::DataLayout::NHWC},
                       out_t_meta = {
                           out->dtype(), out_t_dims, phi::DataLayout::NHWC};
  x_t.set_meta(x_t_meta);
  out_t.set_meta(out_t_meta);
  dev_ctx.template Alloc<T>(&x_t);
  dev_ctx.template Alloc<T>(&out_t);
  sdaa_ops::doTransformTensor(dev_ctx, x, Convert_TF::NCHW2NHWC, &x_t);

  tecodnnHandle_t tecodnnHanndle = GetHandleFromCTX(dev_ctx);
  tecodnnTensorDescriptor_t boxes_N5_Desc =
      sdaa_ops::GetTecodnnTensorDesc(phi::vectorize<int>(boxes_N5.dims()),
                                     boxes_N5.dtype(),
                                     TensorFormat::Undefined);
  tecodnnTensorDescriptor_t x_Desc = sdaa_ops::GetTecodnnTensorDesc(
      phi::vectorize<int>(x_t.dims()), x_t.dtype(), TensorFormat::NHWC);
  tecodnnTensorDescriptor_t out_Desc = sdaa_ops::GetTecodnnTensorDesc(
      phi::vectorize<int>(out_t.dims()), out_t.dtype(), TensorFormat::NHWC);

  TECODNN_CHECK(tecodnnRoiAlignForward(tecodnnHanndle,
                                       static_cast<int>(aligned),
                                       spatial_scale,
                                       sampling_ratio,
                                       boxes_N5_Desc,
                                       boxes_N5.data(),
                                       x_Desc,
                                       x_t.data(),
                                       out_Desc,
                                       out_t.data()));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(boxes_N5_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(out_Desc));

  sdaa_ops::doTransformTensor(dev_ctx, out_t, Convert_TF::NHWC2NCHW, out);
}

template <typename T, typename Context>
void RoiAlignGradKernel(const Context& dev_ctx,
                        const phi::DenseTensor& x,
                        const phi::DenseTensor& boxes,
                        const paddle::optional<phi::DenseTensor>& boxes_num,
                        const phi::DenseTensor& out_grad,
                        int pooled_height,
                        int pooled_width,
                        float spatial_scale,
                        int sampling_ratio,
                        bool aligned,
                        phi::DenseTensor* dx) {
  VLOG(4) << "Call SDAA RoiAlignGradKernel.";
  PADDLE_ENFORCE_LT(
      sampling_ratio,
      1024,
      phi::errors::InvalidArgument(
          "sampling_ratio must be less than 1024, but provided is %d",
          sampling_ratio));

  if (!dx) {
    return;
  }
  dev_ctx.template Alloc<T>(dx);

  int boxes_batch_size;
  int batch_size = x.dims()[0];
  std::vector<T> box_batch_id_data((boxes.dims()[0]));

  if (boxes_num) {
    boxes_batch_size = boxes_num->numel();
    std::vector<int> boxes_num_data;
    TensorToVector(dev_ctx, *boxes_num, dev_ctx, &boxes_num_data);
    int start = 0;
    // transfrom boxes_num to box_batch_id_data
    for (int n = 0; n < boxes_batch_size; ++n) {
      for (int i = start; i < start + boxes_num_data[n]; ++i) {
        box_batch_id_data[i] = static_cast<float>(n);
      }
      start += boxes_num_data[n];
    }
  } else {
    auto boxes_lod = boxes.lod().back();
    boxes_batch_size = boxes_lod.size() - 1;
    for (int n = 0; n < boxes_batch_size; ++n) {
      for (std::size_t i = boxes_lod[n]; i < boxes_lod[n + 1]; ++i) {
        box_batch_id_data[i] = n;
      }
    }
  }

  phi::DenseTensor boxes_num_t;
  TensorFromVector<T>(dev_ctx, box_batch_id_data, dev_ctx, &boxes_num_t);
  boxes_num_t.Resize({boxes.dims()[0], 1});

  std::vector<const phi::DenseTensor*> boxes_list;
  boxes_list.emplace_back(&boxes_num_t);
  boxes_list.emplace_back(&boxes);
  phi::DenseTensor boxes_N5;
  boxes_N5.Resize({boxes.dims()[0], 5});
  dev_ctx.template Alloc<T>(&boxes_N5);

  // concat boxes and rois location to [N,5] tensor
  int axis = 1;
  sdaa_ops::doConcatTensor(dev_ctx, boxes_list, axis, &boxes_N5);

  // tecodnnRoiAlignBackward only support input and output of NHWC format
  phi::DDim dout_t_dims =
      sdaa_ops::doDimPermute(out_grad, Convert_TF::NCHW2NHWC);
  phi::DDim dx_t_dims = sdaa_ops::doDimPermute(*dx, Convert_TF::NCHW2NHWC);
  phi::DenseTensor dout_t, dx_t;
  phi::DenseTensorMeta dout_t_meta = {out_grad.dtype(),
                                      dout_t_dims,
                                      phi::DataLayout::NHWC},
                       dx_t_meta = {
                           dx->dtype(), dx_t_dims, phi::DataLayout::NHWC};
  dout_t.set_meta(dout_t_meta);
  dx_t.set_meta(dx_t_meta);
  dev_ctx.template Alloc<T>(&dout_t);
  dev_ctx.template Alloc<T>(&dx_t);
  sdaa_ops::doTransformTensor(
      dev_ctx, out_grad, Convert_TF::NCHW2NHWC, &dout_t);

  tecodnnHandle_t tecodnnHanndle = GetHandleFromCTX(dev_ctx);
  tecodnnTensorDescriptor_t boxes_N5_Desc =
      sdaa_ops::GetTecodnnTensorDesc(phi::vectorize<int>(boxes_N5.dims()),
                                     boxes_N5.dtype(),
                                     TensorFormat::Undefined);
  tecodnnTensorDescriptor_t dout_Desc = sdaa_ops::GetTecodnnTensorDesc(
      phi::vectorize<int>(dout_t.dims()), dout_t.dtype(), TensorFormat::NHWC);
  tecodnnTensorDescriptor_t dx_Desc = sdaa_ops::GetTecodnnTensorDesc(
      phi::vectorize<int>(dx_t.dims()), dx_t.dtype(), TensorFormat::NHWC);

  TECODNN_CHECK(tecodnnRoiAlignBackward(tecodnnHanndle,
                                        static_cast<int>(aligned),
                                        spatial_scale,
                                        sampling_ratio,
                                        boxes_N5_Desc,
                                        boxes_N5.data(),
                                        dout_Desc,
                                        dout_t.data(),
                                        dx_Desc,
                                        dx_t.data()));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(boxes_N5_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(dout_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(dx_Desc));

  sdaa_ops::doTransformTensor(dev_ctx, dx_t, Convert_TF::NHWC2NCHW, dx);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(roi_align,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::RoiAlignKernel,
                          phi::dtype::float16,
                          float) {}

PD_REGISTER_PLUGIN_KERNEL(roi_align_grad,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::RoiAlignGradKernel,
                          phi::dtype::float16,
                          float) {}
