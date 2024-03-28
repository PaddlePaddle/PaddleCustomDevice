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

#include "kernels/funcs/elementwise_utils.h"
#include "kernels/funcs/logic_op.h"
#include "kernels/funcs/mlu_baseop.h"
#include "kernels/funcs/mlu_funcs.h"

namespace custom_kernel {

template <typename T, typename Context>
void CEmbeddingKernel(const Context& dev_ctx,
                      const phi::DenseTensor& w,
                      const phi::DenseTensor& ids,
                      int64_t start_index,
                      int64_t vocab_size,
                      phi::DenseTensor* out) {
  const auto& index_type = ids.dtype();

  if (index_type == phi::DataType::INT32 ||
      index_type == phi::DataType::INT64) {
    auto out_dims = out->dims();
    auto K = ids.numel();
    auto N = w.dims()[0];
    auto D = w.dims()[1];
    VLOG(5) << "K: " << K << ", N: " << N << ", D: " << D << std::endl;
    VLOG(5) << "start_index: " << start_index << ", vocab_size: " << vocab_size
            << std::endl;

    dev_ctx.template Alloc<T>(out);
    Tensor x_tmp, w_tmp;
    x_tmp = ids;
    x_tmp.Resize({K});
    w_tmp = w;
    w_tmp.Resize({N, D});

    Tensor x_tensor, w_tensor;
    x_tensor = x_tmp;
    w_tensor = w_tmp;
    Tensor start_index_tensor, end_index_tensor, ids_mask_tensor;
    start_index_tensor.Resize({x_tensor.dims()});
    auto start_index_value = static_cast<int32_t>(start_index);
    dev_ctx.template Alloc<int32_t>(&start_index_tensor);
    MLUCnnlTensorDesc start_index_tensor_desc(start_index_tensor);
    MLUCnnl::Fill(dev_ctx,
                  CNNL_POINTER_MODE_HOST,
                  &start_index_value,
                  start_index_tensor_desc.get(),
                  GetBasePtr(&start_index_tensor));

    end_index_tensor.Resize({x_tensor.dims()});
    dev_ctx.template Alloc<int32_t>(&end_index_tensor);
    int64_t end_index = start_index + N;
    auto end_index_value = static_cast<int32_t>(end_index);
    MLUCnnlTensorDesc end_index_tensor_desc(end_index_tensor);
    MLUCnnl::Fill(dev_ctx,
                  CNNL_POINTER_MODE_HOST,
                  &end_index_value,
                  end_index_tensor_desc.get(),
                  GetBasePtr(&end_index_tensor));

    MLUCnnlTensorDesc x_tensor_desc(x_tensor);

    Tensor x_tensor_int32;
    x_tensor_int32.Resize(x_tensor.dims());
    dev_ctx.template Alloc<int32_t>(&x_tensor_int32);
    MLUCnnlTensorDesc x_tensor_int32_desc(x_tensor_int32);

    if (index_type == DataType::INT32) {
      x_tensor_int32 = x_tensor;
    } else {
      cnnlCastDataType_t cast_type =
          GetCastDataType(DataType::INT64, DataType::INT32);
      MLUCnnl::Cast(dev_ctx,
                    cast_type,
                    x_tensor_desc.get(),
                    GetBasePtr(&x_tensor),
                    x_tensor_int32_desc.get(),
                    GetBasePtr(&x_tensor_int32));
    }

    Tensor out_greater_equal;
    out_greater_equal.Resize(x_tensor.dims());
    MLUCnnlTensorDesc out_greater_equal_desc(out_greater_equal);
    MLULogicOp(dev_ctx,
               x_tensor_int32,
               start_index_tensor,
               "greater_equal",
               &out_greater_equal);

    Tensor out_less_than;
    out_less_than.Resize(x_tensor.dims());
    MLUCnnlTensorDesc out_less_than_desc(out_less_than);
    MLULogicOp(
        dev_ctx, x_tensor_int32, end_index_tensor, "less_than", &out_less_than);

    ids_mask_tensor.Resize(x_tensor.dims());
    dev_ctx.template Alloc<bool>(&ids_mask_tensor);
    MLUCnnlTensorDesc ids_mask_tensor_desc(ids_mask_tensor);
    MLULogicOp(
        dev_ctx, out_greater_equal, out_less_than, "and", &ids_mask_tensor);

    Tensor out_sub;
    out_sub.Resize(x_tensor.dims());
    dev_ctx.template Alloc<int32_t>(&out_sub);
    MLUOpTensorKernel<int32_t>(dev_ctx,
                               x_tensor_int32,
                               start_index_tensor,
                               -1,
                               CNNL_OP_TENSOR_SUB,
                               &out_sub);

    Tensor ids_tensor;
    ids_tensor.Resize(x_tensor.dims());
    dev_ctx.template Alloc<int32_t>(&ids_tensor);
    MLUCnnlTensorDesc ids_tensor_desc(ids_tensor);

    Tensor ids_mask_tensor_int32;
    ids_mask_tensor_int32.Resize(x_tensor.dims());
    dev_ctx.template Alloc<int32_t>(&ids_mask_tensor_int32);
    MLUCnnlTensorDesc ids_mask_tensor_int32_desc(ids_mask_tensor_int32);
    cnnlCastDataType_t bool_to_int32 =
        GetCastDataType(DataType::BOOL, DataType::INT32);
    MLUCnnl::Cast(dev_ctx,
                  bool_to_int32,
                  ids_mask_tensor_desc.get(),
                  GetBasePtr(&ids_mask_tensor),
                  ids_mask_tensor_int32_desc.get(),
                  GetBasePtr(&ids_mask_tensor_int32));
    MLUOpTensorKernel<int32_t>(dev_ctx,
                               out_sub,
                               ids_mask_tensor_int32,
                               -1,
                               CNNL_OP_TENSOR_MUL,
                               &ids_tensor);

    ids_mask_tensor_int32.Resize({K, 1});
    MLUCnnlTensorDesc output_desc(*out);
    MLUCnnlTensorDesc w_tensor_desc(w_tensor);

    Tensor embedding_out;
    embedding_out.Resize({K, D});
    dev_ctx.template Alloc<T>(&embedding_out);
    MLUCnnlTensorDesc embedding_out_desc(embedding_out);

    MLUCnnl::EmbeddingForward(dev_ctx,
                              -1,
                              w_tensor_desc.get(),
                              GetBasePtr(&w_tensor),
                              ids_tensor_desc.get(),
                              static_cast<const int*>(GetBasePtr(&ids_tensor)),
                              embedding_out_desc.get(),
                              GetBasePtr(&embedding_out));

    Tensor ids_mask_tensor_new;
    ids_mask_tensor_new.Resize({K, 1});
    dev_ctx.template Alloc<T>(&ids_mask_tensor_new);
    MLUCnnlTensorDesc ids_mask_tensor_new_desc(ids_mask_tensor_new);
    cnnlCastDataType_t int32_to_float32 =
        GetCastDataType(DataType::INT32, w.dtype());
    MLUCnnl::Cast(dev_ctx,
                  int32_to_float32,
                  ids_mask_tensor_int32_desc.get(),
                  GetBasePtr(&ids_mask_tensor_int32),
                  ids_mask_tensor_new_desc.get(),
                  GetBasePtr(&ids_mask_tensor_new));

    out->Resize({K, D});
    MLUOpTensorKernel<T>(dev_ctx,
                         ids_mask_tensor_new,
                         embedding_out,
                         -1,
                         CNNL_OP_TENSOR_MUL,
                         out);
    out->Resize(out_dims);

  } else {
    PADDLE_THROW(phi::errors::Unavailable(
        "Custom MLU Device c_embedding ids only support int32 or int64."));
  }
}

template <typename T, typename Context>
void CEmbeddingGradKernel(const Context& dev_ctx,
                          const phi::DenseTensor& w,
                          const phi::DenseTensor& ids,
                          const phi::DenseTensor& out_grad,
                          int64_t start_index,
                          phi::DenseTensor* w_grad) {
  w_grad->Resize(w.dims());
  dev_ctx.template Alloc(w_grad, w.dtype());
  const auto& index_type = ids.dtype();
  if (index_type == phi::DataType::INT32 ||
      index_type == phi::DataType::INT64) {
    auto K = ids.numel();
    auto N = w.dims()[0];
    auto D = w.dims()[1];
    VLOG(5) << "K: " << K << ", N: " << N << ", D: " << D << std::endl;

    Tensor x_tmp;
    x_tmp = ids;
    x_tmp.Resize({K});
    Tensor w_tmp;
    w_tmp = w;
    dev_ctx.template Alloc<T>(&w_tmp);
    Tensor out_grad_tmp;
    out_grad_tmp = out_grad;
    out_grad_tmp.Resize({K, D});
    Tensor x_tensor, w_tensor, out_grad_tensor;
    x_tensor = x_tmp;
    w_tensor = w_tmp;
    out_grad_tensor = out_grad_tmp;

    Tensor start_index_tensor;
    start_index_tensor.Resize({x_tensor.dims()});
    dev_ctx.template Alloc<int32_t>(&start_index_tensor);
    MLUCnnlTensorDesc start_index_tensor_desc(start_index_tensor);
    auto start_index_value = static_cast<int32_t>(start_index);
    MLUCnnl::Fill(dev_ctx,
                  CNNL_POINTER_MODE_HOST,
                  &start_index_value,
                  start_index_tensor_desc.get(),
                  GetBasePtr(&start_index_tensor));

    Tensor end_index_tensor;
    end_index_tensor.Resize({x_tensor.dims()});
    dev_ctx.template Alloc<int32_t>(&end_index_tensor);
    MLUCnnlTensorDesc end_index_tensor_desc(end_index_tensor);
    int64_t end_index = start_index + N;
    auto end_index_value = static_cast<int32_t>(end_index);
    MLUCnnl::Fill(dev_ctx,
                  CNNL_POINTER_MODE_HOST,
                  &end_index_value,
                  end_index_tensor_desc.get(),
                  GetBasePtr(&end_index_tensor));
    MLUCnnlTensorDesc x_tensor_desc(x_tensor);
    Tensor x_tensor_int32;
    x_tensor_int32.Resize(x_tensor.dims());
    dev_ctx.template Alloc<int32_t>(&x_tensor_int32);
    MLUCnnlTensorDesc x_tensor_int32_desc(x_tensor_int32);
    if (index_type == DataType::INT32) {
      x_tensor_int32 = x_tensor;
    } else {
      cnnlCastDataType_t cast_type =
          GetCastDataType(DataType::INT64, DataType::INT32);
      MLUCnnl::Cast(dev_ctx,
                    cast_type,
                    x_tensor_desc.get(),
                    GetBasePtr(&x_tensor),
                    x_tensor_int32_desc.get(),
                    GetBasePtr(&x_tensor_int32));
    }

    Tensor out_greater_equal;
    out_greater_equal.Resize(x_tensor.dims());
    MLUCnnlTensorDesc out_greater_equal_desc(out_greater_equal);
    MLULogicOp(dev_ctx,
               x_tensor_int32,
               start_index_tensor,
               "greater_equal",
               &out_greater_equal);

    Tensor out_less_than;
    out_less_than.Resize(x_tensor.dims());
    MLUCnnlTensorDesc out_less_than_desc(out_less_than);
    MLULogicOp(
        dev_ctx, x_tensor_int32, end_index_tensor, "less_than", &out_less_than);

    Tensor ids_mask_tensor;
    ids_mask_tensor.Resize(x_tensor.dims());
    dev_ctx.template Alloc<bool>(&ids_mask_tensor);
    MLUCnnlTensorDesc ids_mask_tensor_desc(ids_mask_tensor);
    MLULogicOp(
        dev_ctx, out_greater_equal, out_less_than, "and", &ids_mask_tensor);

    Tensor out_sub;
    out_sub.Resize(x_tensor.dims());
    dev_ctx.template Alloc<int32_t>(&out_sub);
    MLUOpTensorKernel<int32_t>(dev_ctx,
                               x_tensor_int32,
                               start_index_tensor,
                               -1,
                               CNNL_OP_TENSOR_SUB,
                               &out_sub);

    Tensor ids_mask_tensor_int32;
    ids_mask_tensor_int32.Resize(x_tensor.dims());
    dev_ctx.template Alloc<int32_t>(&ids_mask_tensor_int32);
    MLUCnnlTensorDesc ids_mask_tensor_int32_desc(ids_mask_tensor_int32);
    cnnlCastDataType_t bool_to_int32 =
        GetCastDataType(DataType::BOOL, DataType::INT32);
    MLUCnnl::Cast(dev_ctx,
                  bool_to_int32,
                  ids_mask_tensor_desc.get(),
                  GetBasePtr(&ids_mask_tensor),
                  ids_mask_tensor_int32_desc.get(),
                  GetBasePtr(&ids_mask_tensor_int32));
    Tensor real_ids_tensor;
    real_ids_tensor.Resize(x_tensor.dims());
    dev_ctx.template Alloc<int32_t>(&real_ids_tensor);
    MLUCnnlTensorDesc real_ids_tensor_desc(real_ids_tensor);
    MLUOpTensorKernel<int32_t>(dev_ctx,
                               out_sub,
                               ids_mask_tensor_int32,
                               -1,
                               CNNL_OP_TENSOR_MUL,
                               &real_ids_tensor);

    out_grad_tensor.Resize({K, D});
    Tensor ids_mask_tensor_new;
    ids_mask_tensor_new.Resize({K, 1});
    dev_ctx.template Alloc<T>(&ids_mask_tensor_new);
    MLUCnnlTensorDesc ids_mask_tensor_new_desc(ids_mask_tensor_new);
    cnnlCastDataType_t cast_dtype = GetCastDataType(DataType::INT32, w.dtype());
    MLUCnnl::Cast(dev_ctx,
                  cast_dtype,
                  ids_mask_tensor_int32_desc.get(),
                  GetBasePtr(&ids_mask_tensor_int32),
                  ids_mask_tensor_new_desc.get(),
                  GetBasePtr(&ids_mask_tensor_new));

    Tensor out_grad_tensor_mul_mask;
    out_grad_tensor_mul_mask.Resize({K, D});
    MLUCnnlTensorDesc out_grad_tensor_mul_mask_desc(out_grad_tensor_mul_mask);
    MLUOpTensorKernel<T>(dev_ctx,
                         out_grad_tensor,
                         ids_mask_tensor_new,
                         -1,
                         CNNL_OP_TENSOR_MUL,
                         &out_grad_tensor_mul_mask);

    MLUCnnlTensorDesc w_grad_desc(*w_grad);
    MLUCnnl::EmbeddingBackward(dev_ctx,
                               -1,
                               false,
                               real_ids_tensor_desc.get(),
                               GetBasePtr(&real_ids_tensor),
                               out_grad_tensor_mul_mask_desc.get(),
                               GetBasePtr(&out_grad_tensor_mul_mask),
                               w_grad_desc.get(),
                               GetBasePtr(w_grad));

  } else {
    PADDLE_THROW(phi::errors::Unavailable(
        "Custom MLU Device c_embedding ids only support int32 or int64."));
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(c_embedding,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::CEmbeddingKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(c_embedding_grad,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::CEmbeddingGradKernel,
                          float,
                          double,
                          phi::dtype::float16) {}
