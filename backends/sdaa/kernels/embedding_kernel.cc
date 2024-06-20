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

#include "kernels/funcs/sdaa_baseop.h"
#include "kernels/funcs/sdaa_funcs.h"
#include "paddle/phi/extension.h"

namespace custom_kernel {

template <typename T, typename Context>
void EmbeddingKernel(const Context &dev_ctx,
                     const phi::DenseTensor &inputx,
                     const phi::DenseTensor &weight,
                     int64_t padding_idx,
                     phi::DenseTensor *out) {
  VLOG(4) << "Call SDAA EmbeddingKernel";
  // basic settings
  dev_ctx.template Alloc<T>(out);
  auto mode = TECODNN_EMBEDDING_SCALE_GRAD_UNFRED;
  auto type = TECODNN_ARRAY_DENSE;
  tecodnnDataType_t indice_dataType =
      sdaa_ops::ToTecodnnDataType(inputx.dtype());
  tecodnnDataType_t weight_dataType =
      sdaa_ops::ToTecodnnDataType(phi::CppTypeToDataType<T>::Type());
  int indicesBatch = 1;
  int indicesSeqLength = 1;
  if (inputx.dims().size() == 1) {
    indicesSeqLength = inputx.dims()[0];
  } else if (inputx.dims().size() == 2) {
    indicesBatch = inputx.dims()[0];
    indicesSeqLength = inputx.dims()[1];
  }
  int weightSeqLength = weight.dims()[0];
  int weightVectSize = weight.dims()[1];

  // set descriptors
  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);
  tecodnnSeqDataDescriptor_t indices_Desc, weight_Desc, out_Desc;
  TECODNN_CHECK(tecodnnCreateSeqDataDescriptor(&indices_Desc));
  TECODNN_CHECK(tecodnnCreateSeqDataDescriptor(&weight_Desc));
  TECODNN_CHECK(tecodnnCreateSeqDataDescriptor(&out_Desc));

  // set dims of seqData
  int dimA[TECODNN_SEQDATA_DIM_COUNT];  // array to save dim-params
  tecodnnSeqDataAxis_t axes[TECODNN_SEQDATA_DIM_COUNT];
  axes[0] = TECODNN_SEQDATA_BATCH_DIM;
  axes[1] = TECODNN_SEQDATA_BEAM_DIM;
  axes[2] = TECODNN_SEQDATA_TIME_DIM;
  axes[3] = TECODNN_SEQDATA_VECT_DIM;

  // indices Param
  dimA[TECODNN_SEQDATA_BATCH_DIM] = indicesBatch;
  dimA[TECODNN_SEQDATA_BEAM_DIM] = 1;
  dimA[TECODNN_SEQDATA_TIME_DIM] = indicesSeqLength;
  dimA[TECODNN_SEQDATA_VECT_DIM] = 1;
  size_t indicesSLAS = indicesBatch * 1;
  int *indicesSLA = reinterpret_cast<int *>(malloc(indicesSLAS * sizeof(int)));
  for (int i = 0; i < indicesSLAS; i++) indicesSLA[i] = indicesSeqLength;
  TECODNN_CHECK(tecodnnSetSeqDataDescriptor(indices_Desc,
                                            indice_dataType,
                                            TECODNN_SEQDATA_DIM_COUNT,
                                            dimA,
                                            axes,
                                            indicesSLAS,
                                            indicesSLA,
                                            NULL));
  // weight Param
  dimA[TECODNN_SEQDATA_BATCH_DIM] = 1;
  dimA[TECODNN_SEQDATA_BEAM_DIM] = 1;
  dimA[TECODNN_SEQDATA_TIME_DIM] = weightSeqLength;
  dimA[TECODNN_SEQDATA_VECT_DIM] = weightVectSize;
  size_t weightSLAS = 1 * 1;
  int *weightSLA = reinterpret_cast<int *>(malloc(weightSLAS * sizeof(int)));
  for (int i = 0; i < weightSLAS; i++) weightSLA[i] = weightSeqLength;
  TECODNN_CHECK(tecodnnSetSeqDataDescriptor(weight_Desc,
                                            weight_dataType,
                                            TECODNN_SEQDATA_DIM_COUNT,
                                            dimA,
                                            axes,
                                            weightSLAS,
                                            weightSLA,
                                            NULL));

  // out Param
  dimA[TECODNN_SEQDATA_BATCH_DIM] = indicesBatch;
  dimA[TECODNN_SEQDATA_BEAM_DIM] = 1;
  dimA[TECODNN_SEQDATA_TIME_DIM] = indicesSeqLength;
  dimA[TECODNN_SEQDATA_VECT_DIM] = weightVectSize;
  size_t outSLAS = indicesBatch * 1;
  int *outSLA = reinterpret_cast<int *>(malloc(outSLAS * sizeof(int)));
  for (int i = 0; i < outSLAS; i++) outSLA[i] = indicesSeqLength;
  TECODNN_CHECK(tecodnnSetSeqDataDescriptor(out_Desc,
                                            weight_dataType,
                                            TECODNN_SEQDATA_DIM_COUNT,
                                            dimA,
                                            axes,
                                            outSLAS,
                                            outSLA,
                                            NULL));

  // excute embedding forward
  tecodnnEmbeddingDescriptor_t embeddingDesc;
  TECODNN_CHECK(tecodnnCreateEmbeddingDescriptor(&embeddingDesc));
  TECODNN_CHECK(
      tecodnnSetEmbeddingDescriptor(embeddingDesc, padding_idx, mode, type));
  float alpha = 1.0f, beta = 0.0f;
  TECODNN_CHECK(tecodnnEmbeddingForward(tecodnnHandle,
                                        embeddingDesc,
                                        &alpha,
                                        indices_Desc,
                                        inputx.data(),
                                        weight_Desc,
                                        weight.data(),
                                        &beta,
                                        out_Desc,
                                        out->data()));
  TECODNN_CHECK(tecodnnDestroyEmbeddingDescriptor(embeddingDesc));
  TECODNN_CHECK(tecodnnDestroySeqDataDescriptor(weight_Desc));
  TECODNN_CHECK(tecodnnDestroySeqDataDescriptor(indices_Desc));
  TECODNN_CHECK(tecodnnDestroySeqDataDescriptor(out_Desc));
  free(indicesSLA);
  free(weightSLA);
  free(outSLA);
}

template <typename T, typename Context>
void EmbeddingGradKernel(const Context &dev_ctx,
                         const phi::DenseTensor &input,
                         const phi::DenseTensor &weight,
                         const phi::DenseTensor &out_grad,
                         int64_t padding_idx,
                         phi::DenseTensor *weight_grad) {
  VLOG(4) << "Call SDAA EmbeddingGradKernel";
  // basic settings
  dev_ctx.template Alloc<T>(weight_grad);
  auto mode = TECODNN_EMBEDDING_SCALE_GRAD_UNFRED;
  auto type = TECODNN_ARRAY_DENSE;
  tecodnnDataType_t indice_dataType = TECODNN_DATA_INT32;
  tecodnnDataType_t weight_dataType =
      sdaa_ops::ToTecodnnDataType(phi::CppTypeToDataType<T>::Type());

  int dweightSeqLength = weight_grad->dims()[0];
  int dweightVectSize = weight_grad->dims()[1];
  int indicesBatch = 1;
  int indicesSeqLength = 1;
  if (input.dims().size() == 1) {
    indicesSeqLength = input.dims()[0];
  } else {
    indicesBatch = input.dims()[0];
    indicesSeqLength = input.dims()[1];
  }

  // switch input from int64 into int32
  phi::DenseTensor inputx_cast;
  if (input.dtype() == phi::DataType::INT64) {
    inputx_cast.Resize(input.dims());
    dev_ctx.template Alloc<int32_t>(&inputx_cast);
    sdaa_ops::doCastTensor(dev_ctx, input, &inputx_cast);
  } else {
    inputx_cast = input;
  }

  // set descriptors
  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);
  tecodnnSeqDataDescriptor_t indices_Desc, dweight_Desc, dout_Desc;
  TECODNN_CHECK(tecodnnCreateSeqDataDescriptor(&indices_Desc));
  TECODNN_CHECK(tecodnnCreateSeqDataDescriptor(&dweight_Desc));
  TECODNN_CHECK(tecodnnCreateSeqDataDescriptor(&dout_Desc));

  // set dims of seqData
  int dimA[TECODNN_SEQDATA_DIM_COUNT];
  tecodnnSeqDataAxis_t axes[TECODNN_SEQDATA_DIM_COUNT];
  axes[0] = TECODNN_SEQDATA_BATCH_DIM;
  axes[1] = TECODNN_SEQDATA_BEAM_DIM;
  axes[2] = TECODNN_SEQDATA_TIME_DIM;
  axes[3] = TECODNN_SEQDATA_VECT_DIM;

  // indices Param
  dimA[TECODNN_SEQDATA_BATCH_DIM] = indicesBatch;
  dimA[TECODNN_SEQDATA_BEAM_DIM] = 1;
  dimA[TECODNN_SEQDATA_TIME_DIM] = indicesSeqLength;
  dimA[TECODNN_SEQDATA_VECT_DIM] = 1;
  size_t indicesSLAS = indicesBatch * 1;
  int *indicesSLA = reinterpret_cast<int *>(malloc(indicesSLAS * sizeof(int)));
  for (int i = 0; i < indicesSLAS; i++) indicesSLA[i] = indicesSeqLength;
  TECODNN_CHECK(tecodnnSetSeqDataDescriptor(indices_Desc,
                                            indice_dataType,
                                            TECODNN_SEQDATA_DIM_COUNT,
                                            dimA,
                                            axes,
                                            indicesSLAS,
                                            indicesSLA,
                                            NULL));
  // weight Param
  dimA[TECODNN_SEQDATA_BATCH_DIM] = 1;
  dimA[TECODNN_SEQDATA_BEAM_DIM] = 1;
  dimA[TECODNN_SEQDATA_TIME_DIM] = dweightSeqLength;
  dimA[TECODNN_SEQDATA_VECT_DIM] = dweightVectSize;
  size_t dweightSLAS = 1 * 1;
  int *dweightSLA = reinterpret_cast<int *>(malloc(dweightSLAS * sizeof(int)));
  for (int i = 0; i < dweightSLAS; i++) dweightSLA[i] = dweightSeqLength;
  TECODNN_CHECK(tecodnnSetSeqDataDescriptor(dweight_Desc,
                                            weight_dataType,
                                            TECODNN_SEQDATA_DIM_COUNT,
                                            dimA,
                                            axes,
                                            dweightSLAS,
                                            dweightSLA,
                                            NULL));

  // out Param
  dimA[TECODNN_SEQDATA_BATCH_DIM] = indicesBatch;
  dimA[TECODNN_SEQDATA_BEAM_DIM] = 1;
  dimA[TECODNN_SEQDATA_TIME_DIM] = indicesSeqLength;
  dimA[TECODNN_SEQDATA_VECT_DIM] = dweightVectSize;
  size_t doutSLAS = indicesBatch * 1;
  int *doutSLA = reinterpret_cast<int *>(malloc(doutSLAS * sizeof(int)));
  for (int i = 0; i < doutSLAS; i++) doutSLA[i] = indicesSeqLength;
  TECODNN_CHECK(tecodnnSetSeqDataDescriptor(dout_Desc,
                                            weight_dataType,
                                            TECODNN_SEQDATA_DIM_COUNT,
                                            dimA,
                                            axes,
                                            doutSLAS,
                                            doutSLA,
                                            NULL));

  // excute embedding forward
  tecodnnEmbeddingDescriptor_t embeddingDesc;
  TECODNN_CHECK(tecodnnCreateEmbeddingDescriptor(&embeddingDesc));
  TECODNN_CHECK(
      tecodnnSetEmbeddingDescriptor(embeddingDesc, padding_idx, mode, type));
  float alpha = 1.0f, beta = 0.0f;
  TECODNN_CHECK(tecodnnEmbeddingBackward(tecodnnHandle,
                                         embeddingDesc,
                                         &alpha,
                                         dout_Desc,
                                         out_grad.data(),
                                         indices_Desc,
                                         inputx_cast.data(),
                                         &beta,
                                         dweight_Desc,
                                         weight_grad->data()));
  TECODNN_CHECK(tecodnnDestroyEmbeddingDescriptor(embeddingDesc));
  TECODNN_CHECK(tecodnnDestroySeqDataDescriptor(dweight_Desc));
  TECODNN_CHECK(tecodnnDestroySeqDataDescriptor(indices_Desc));
  TECODNN_CHECK(tecodnnDestroySeqDataDescriptor(dout_Desc));
  free(indicesSLA);
  free(dweightSLA);
  free(doutSLA);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(embedding,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::EmbeddingKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(embedding_grad,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::EmbeddingGradKernel,
                          float,
                          phi::dtype::float16) {}
