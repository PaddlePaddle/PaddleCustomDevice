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

#pragma once

#include <string>

static const char* const KDev = "dtu";

// unary ops
static const char* const kNeg = "neg";
static const char* const kAbs = "abs";
static const char* const kSqrt = "sqrt";
static const char* const kExp = "exp";
static const char* const kLog = "log";
static const char* const kReciprocal = "reciprocal";
static const char* const kRemainder = "rem";
static const char* const kConvert = "convert";
static const char* const kSigmoid = "sigmoid_forward";
static const char* const kLeakyRelu = "leaky_relu_forward";
static const char* const kThreshold = "threshold";
static const char* const kLogSoftmax = "log_softmax_forward";
static const char* const kSoftmax = "softmax_forward";
static const char* const kLog2 = "log2";
static const char* const kRsqrt = "rsqrt";
static const char* const kFloor = "floor";
static const char* const kSign = "sign";
static const char* const kBitwiseNot = "not";
static const char* const kCeil = "ceil";
static const char* const kSin = "sin";
static const char* const kCos = "cos";
static const char* const kTan = "tan";
static const char* const kAsin = "asin";
static const char* const kAcos = "acos";
static const char* const kAtan = "atan";
static const char* const kSinh = "sinh";
static const char* const kCosh = "cosh";
static const char* const kTanh = "tanh";
static const char* const kAsinh = "asinh";
static const char* const kAcosh = "acosh";
static const char* const kAtanh = "atanh";

// binary ops
static const char* const kAdd = "add";
static const char* const kSub = "sub";
static const char* const kMul = "mul";
static const char* const kDiv = "div";
static const char* const kMaximum = "max";
static const char* const kMinimum = "min";
static const char* const kPow = "pow";
static const char* const kEq = "eq";
static const char* const kNe = "ne";
static const char* const kGe = "ge";
static const char* const kGt = "gt";
static const char* const kLe = "le";
static const char* const kLt = "lt";
static const char* const kSigmoidBackward = "sigmoid_backward";
static const char* const kLeakyReluBackward = "leaky_relu_backward";
static const char* const kBitwiseAnd = "and";
static const char* const kBitwiseOr = "or";
static const char* const kThresholdBackward = "threshold_backward";
static const char* const kLogSoftmaxBackward = "log_softmax_backward";
static const char* const kSoftmaxBackward = "softmax_backward";

// compare ops
static const char* const kSelect = "select";
static const char* const kClamp = "clamp";
static const char* const kClampScalar = "clamp_scalar";
static const char* const kTopk = "topk";
static const char* const kSort = "sort_ex";
static const char* const kSortEx = "sort";
static const char* const kUnique2 = "unique2";
static const char* const kUnique2Dims = "unique2_get_output_shape";

// dma ops
static const char* const kGather = "gather";
static const char* const kScatter = "scatter";
static const char* const kTransPose = "transpose";
static const char* const KConcat = "concatenate";
static const char* const kRepeat = "repeat";
static const char* const kBroadcast = "broadcast";
static const char* const kBroadcastInDim = "broadcast_in_dim";
static const char* const kArange = "arange";
static const char* const kPad = "pad";
static const char* const kReverse = "reverse";
static const char* const kTriu = "triu";
static const char* const kTril = "tril";
static const char* const kDynamicUpdateSlice = "dynamic_update_slice";
static const char* const kIota = "iota";

// index op
static const char* const kIndex = "index";
static const char* const kIndexPut = "index_put";
static const char* const kMaskedFill = "masked_fill";
static const char* const kEmbeddingDenseBackward = "embedding_dense_backward";

// reduce op
static const char* const kSum = "reduce_sum";
static const char* const kMean = "reduce_avg";
static const char* const kMax = "reduce_max";
static const char* const kMin = "reduce_min";
static const char* const kArgMax = "reduce_argmax";
static const char* const kArgMin = "reduce_argmin";
static const char* const kReduceNorm = "reduce_norm";
static const char* const kReduceNorm0 = "reduce_norm0";
static const char* const kReduceNorm1 = "reduce_norm1";
static const char* const kReduceNorm2 = "reduce_norm2";
static const char* const kReduceNormp = "reduce_normp";
static const char* const kAbsReduceMax = "abs_reduce_max";  // norm inf
static const char* const kAbsReduceMin = "abs_reduce_min";  // norm -inf
static const char* const kReduceOr = "reduce_or";
static const char* const kReduceAnd = "reduce_and";

// convolution op
static const char* const kConv2d = "conv2d";
static const char* const kConv2dBias = "conv2d_bias_activation";
static const char* const kConv2dBiasBackward = "conv2d_bias_backward";
static const char* const kConv2dBpk = "conv2d_bpk";
static const char* const kConv2dBpi = "conv2d_bpi";

// norm op
static const char* const kNativeBatchNorm = "batch_norm";
static const char* const kNativeBatchNormTraining = "batch_norm_training";
static const char* const kNativeBatchNormInference = "batch_norm_inference";
static const char* const kNativeBatchNormBackward = "batch_norm_grad";
static const char* const kNativeLayerNorm = "layer_norm_training";
static const char* const kNativeLayerNormInference = "layer_norm_inference";
static const char* const kNativeLayerNormBackward = "layer_norm_grad";
static const char* const kGroupNorm = "group_norm_inference";
static const char* const kNativeGroupNorm = "group_norm_training";
static const char* const kNativeGroupNormBackward = "group_norm_grad";

// resize op
static const char* const kUpsampleNearest2d = "resize";
static const char* const kUpsampleNearest2dBackward = "resize_grad";
static const char* const kUpsample2d = "resize";
static const char* const kUpsample2dBackward = "resize_grad";

// loss op
static const char* const kBinaryCrossEntropyWithLogits =
    "binary_cross_entropy_with_logits";
static const char* const kNllLossBackward = "nll_loss_backward";
static const char* const kNllLossForward = "nll_loss_forward";

// view op
static const char* const kReshape = "reshape";
static const char* const kSlice = "slice";

// mat mul op
static const char* const kDot = "dot";
static const char* const kDotGeneral = "dot_general";
static const char* const kLinear = "linear";
static const char* const kLinearBpi = "linear_bpi";
static const char* const kLinearBpk = "linear_bpk";

// rng op
static const char* const kFusedDropout = "dropout";
static const char* const kFusedDropoutBackward = "droput_grad";
static const char* const kUniform = "rng_uniform";

// pointwise op
static const char* const kAddcmul = "addcmul";
static const char* const kAddcdiv = "addcdiv";

// activation op
static const char* const kGelu = "gelu";
static const char* const kGeluBackward = "gelu_grad";
static const char* const kSilu = "swish_forward";
static const char* const kSiluBackward = "swish_backward";
static const char* const kRelu = "relu_forward";
static const char* const kReluGrad = "relu_backward";
static const char* const kTanhBackward = "tanh_backward";

// pooling op
static const char* const kMaxPool2d = "reduce_window";
static const char* const kMaxPool2dWithIndices = "max_pool2d_with_indices";
static const char* const kMaxPool2dBackward = "select_and_scatter";

// vision op
static const char* const kRoiAlign = "roi_align";
static const char* const kRoiAlignBackward = "roi_align_grad";
static const char* const kNms = "nms";
static const char* const kNmsDims = "nms_get_output_shape";
