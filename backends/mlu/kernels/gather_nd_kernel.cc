/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "kernels/funcs/mlu_baseop.h"
#include "kernels/funcs/mlu_funcs.h"
#include "kernels/funcs/reduce_op.h"

namespace custom_kernel {

template <typename T, typename Context>
void GatherNdKernel(const Context &dev_ctx,
                    const phi::DenseTensor &x,
                    const phi::DenseTensor &index,
                    phi::DenseTensor *out) {
  dev_ctx.template Alloc<T>(out);

  if (x.numel() == 0) return;
  if (index.numel() == 0) {  // empty index, do broadcast and return
    int diff = out->dims().size() - x.dims().size();
    if (diff == 0) {
      TensorCopy(dev_ctx, x, false, out);
    } else {
      std::vector<int64_t> new_dims(diff, 1);
      for (size_t i = 0; i < x.dims().size(); ++i) {
        new_dims.emplace_back(x.dims()[i]);
      }

      phi::DenseTensor x_tmp(x);
      x_tmp.Resize(phi::make_ddim(new_dims));
      MLUCnnlTensorDesc x_tmp_desc(x_tmp);
      MLUCnnlTensorDesc out_desc(*out);

      MLUCnnl::BroadcastTo(dev_ctx,
                           x_tmp_desc.get(),
                           GetBasePtr(&x_tmp),
                           out_desc.get(),
                           GetBasePtr(out));
    }
    return;
  }

  const auto &index_type = index.dtype();
  bool index_type_match =
      index_type == phi::DataType::INT32 || index_type == phi::DataType::INT64;
  PADDLE_ENFORCE_EQ(
      index_type_match,
      true,
      phi::errors::InvalidArgument("Index holds the wrong type, it holds [%s],"
                                   "but desires to be [%s] or [%s]",
                                   index_type,
                                   phi::DataType::INT32,
                                   phi::DataType::INT64));

  MLUCnnlTensorDesc x_desc(x);
  MLUCnnlTensorDesc index_desc(index);
  MLUCnnlTensorDesc out_desc(*out);
  MLUCnnl::GatherNd(dev_ctx,
                    x_desc.get(),
                    GetBasePtr(&x),
                    index_desc.get(),
                    GetBasePtr(&index),
                    out_desc.get(),
                    GetBasePtr(out));
}

template <typename T, typename Context>
void GatherNdGradKernel(const Context &dev_ctx,
                        const phi::DenseTensor &x,
                        const phi::DenseTensor &index,
                        const phi::DenseTensor &dout,
                        phi::DenseTensor *dx) {
  auto x_dims = dx->dims();
  dev_ctx.template Alloc<T>(dx);

  if (dx->numel() == 0) return;
  if (index.numel() == 0) {
    int diff = dout.dims().size() - x_dims.size();
    if (diff == 0) {
      TensorCopy(dev_ctx, dout, false, dx);
    } else {
      std::vector<int64_t> axes;
      for (size_t i = 0; i < diff; ++i) {
        axes.push_back(i);
      }

      MLUReduceOp<T>(dev_ctx, dout, axes, false, false, "reduce_sum", dx);
    }
    return;
  }

  const phi::DenseTensor *p_index = &index;
  const phi::DenseTensor *p_dout = &dout;
  phi::DenseTensor tmp_tensor(index);
  phi::DenseTensor tmp_tensor2(dout);
  const auto index_dims = index.dims();
  if (index_dims.size() == 1) {
    std::vector<int64_t> new_dim = {1, index_dims[0]};
    tmp_tensor.Resize(phi::make_ddim(new_dim));
    p_index = &tmp_tensor;

    std::vector<int64_t> new_dim2{1};
    for (int i = index.numel(); i < x.dims().size(); i++) {
      new_dim2.push_back(x.dims()[i]);
    }
    tmp_tensor2.Resize(phi::make_ddim(new_dim2));
    p_dout = &tmp_tensor2;
  }

  MLUCnnlTensorDesc dx_desc(*dx);
  auto value = static_cast<T>(0);
  MLUCnnl::Fill(
      dev_ctx, CNNL_POINTER_MODE_HOST, &value, dx_desc.get(), GetBasePtr(dx));
  MLUCnnlTensorDesc index_desc(*p_index);
  MLUCnnlTensorDesc dout_desc(*p_dout);
  const cnnlScatterNdMode_t mode = CNNL_SCATTERND_ADD;
  MLUCnnl::ScatterNd(dev_ctx,
                     mode,
                     index_desc.get(),
                     GetBasePtr(p_index),
                     dout_desc.get(),
                     GetBasePtr(p_dout),
                     dx_desc.get(),
                     GetBasePtr(dx),
                     dx_desc.get(),
                     GetBasePtr(dx));
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(gather_nd,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::GatherNdKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(gather_nd_grad,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::GatherNdGradKernel,
                          float,
                          phi::dtype::float16) {}
