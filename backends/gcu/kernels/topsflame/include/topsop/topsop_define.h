// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.

 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 *     @defgroup define
 *     @{
 *
 */

/**
 * @file topsop_define.h
 * @brief topsflame definitions.
 */

#ifndef TOPSOP_DEFINE_H_  // NOLINT
#define TOPSOP_DEFINE_H_

#if defined(__linux__)
#define TOPSOP_EXPORT __attribute((visibility("default")))
#else
#define TOPSOP_EXPORT
#endif  // __linux

#if defined(__cplusplus)
extern "C" {
#endif

#include <stdint.h>

#define TOPSOP_DIM_MAX (8)

/** topsopDataType_t */
typedef enum {
  TOPSOP_DATA_NONE = -1, /**< TOPSOP_DATA_NONE -1  */
  TOPSOP_DATA_I8 = 0,    /**< TOPSOP_DATA_I8 0  */
  TOPSOP_DATA_U8,        /**< TOPSOP_DATA_U8 1  */
  TOPSOP_DATA_I16,       /**< TOPSOP_DATA_I16 2  */
  TOPSOP_DATA_U16,       /**< TOPSOP_DATA_U16 3  */
  TOPSOP_DATA_FP16,      /**< TOPSOP_DATA_FP16 4  */
  TOPSOP_DATA_BF16,      /**< TOPSOP_DATA_BF16 5  */
  TOPSOP_DATA_I32,       /**< TOPSOP_DATA_I32 6  */
  TOPSOP_DATA_U32,       /**< TOPSOP_DATA_U32 7  */
  TOPSOP_DATA_FP32,      /**< TOPSOP_DATA_FP32 8  */
  TOPSOP_DATA_EF32,      /**< TOPSOP_DATA_EF32 9  */
  TOPSOP_DATA_TF32,      /**< TOPSOP_DATA_TF32 10  */
  TOPSOP_DATA_I64,       /**< TOPSOP_DATA_I64 11  */
  TOPSOP_DATA_U64,       /**< TOPSOP_DATA_U64 12  */
  TOPSOP_DATA_F64,       /**< TOPSOP_DATA_F64 13  */
  TOPSOP_DATA_PRED,      /**< TOPSOP_DATA_PRED 14  */
  TOPSOP_DATA_I4,        /**< TOPSOP_DATA_I4 15  */
} topsopDataType_t;

/** topsopStatus_t */
typedef enum {
  TOPSOP_STATUS_SUCCESS = 0,    /**< TOPSOP_STATUS_SUCCESS 0  */
  TOPSOP_STATUS_ALLOC_FAILED,   /**< TOPSOP_STATUS_ALLOC_FAILED 1  */
  TOPSOP_STATUS_BAD_PARAM,      /**< TOPSOP_STATUS_BAD_PARAM 2  */
  TOPSOP_STATUS_NOT_SUPPORT,    /**< TOPSOP_STATUS_NOT_SUPPORT 3  */
  TOPSOP_STATUS_INTERNAL_ERROR, /**< TOPSOP_STATUS_INTERNAL_ERROR 4  */
  TOPSOP_STATUS_RUNTIME_ERROR,  /**< TOPSOP_STATUS_RUNTIME_ERROR 5  */
  TOPSOP_STATUS_EXECUTE_ERROR,  /**< TOPSOP_STATUS_EXECUTE_ERROR 6  */
} topsopStatus_t;

/** topsopTensorFormat_t */
typedef enum {
  TOPSOP_TENSOR_NCHW = 0, /**< TOPSOP_TENSOR_NCHW 0  */
  TOPSOP_TENSOR_NHWC = 1, /**< TOPSOP_TENSOR_NHWC 1  */
} topsopTensorFormat_t;

/** topsopMemoryFormat_t */
typedef enum {
  TOPSOP_MEMORY_NONE = -1,      /**< topsopMemoryFormat_t == null. */
  TOPSOP_MEMORY_CONTIGUOUS = 0, /**< strides in decreasing order. */
  /**< s[0] > s[2] > s[3] > s[1] == 1  aka NHWC order.  */
  TOPSOP_MEMORY_NHWC,
  /**< s[0] > s[2] > s[3] > s[4] > s[1] == 1  aka NDHWC order.  */
  TOPSOP_MEMORY_NDHWC,
  TOPSOP_MEMORY_PRESERVE, /**< preserve the format as input tensor */
} topsopMemoryFormat_t;

/** topsopDevice_t */
typedef enum {
  TOPSOP_DEVICE_CPU = 0, /** tensor on cpu */
  TOPSOP_DEVICE_GCU,     /** tensor on gcu */
} topsopDevice_t;

/** topsopActivationMode_t */
typedef enum {
  TOPSOP_ACTIVATION_NONE = 0,         /**< TOPSOP_ACTIVATION_NONE 0  */
  TOPSOP_ACTIVATION_RELU = 1,         /**< TOPSOP_ACTIVATION_RELU 1  */
  TOPSOP_ACTIVATION_SIGMOID = 2,      /**< TOPSOP_ACTIVATION_SIGMOID 2  */
  TOPSOP_ACTIVATION_CLIPPED_RELU = 3, /**< TOPSOP_ACTIVATION_CLIPPED_RELU 3  */
  TOPSOP_ACTIVATION_ELU = 4,          /**< TOPSOP_ACTIVATION_ELU 4  */
  TOPSOP_ACTIVATION_IDENTITY = 5,     /**< TOPSOP_ACTIVATION_IDENTITY 5  */
  TOPSOP_ACTIVATION_TANH = 6,         /**< TOPSOP_ACTIVATION_TANH 6  */
  TOPSOP_ACTIVATION_SWISH = 7,        /**< TOPSOP_ACTIVATION_SWISH 7  */
  TOPSOP_ACTIVATION_LEAKY_RELU = 8,   /**< TOPSOP_ACTIVATION_LEAKY_RELU 8  */
  TOPSOP_ACTIVATION_GELU = 9,         /**< TOPSOP_ACTIVATION_GELU 9  */
  TOPSOP_ACTIVATION_SWIGLU = 10,      /**< TOPSOP_ACTIVATION_SWIGLU 10  */
  TOPSOP_ACTIVATION_HARD_SWISH = 11,  /**< TOPSOP_ACTIVATION_HARD_SWISH 11 */
} topsopActivationMode_t;

/** topsopNanPropagation_t */
typedef enum {
  TOPSOP_NOT_PROPAGATE_NAN = 0, /**< TOPSOP_NOT_PROPAGATE_NAN 0  */
  TOPSOP_PROPAGATE_NAN = 1,     /**< TOPSOP_PROPAGATE_NAN 1  */
} topsopNanPropagation_t;

/** topsopElementwiseOpType_t */
typedef enum {
  TOPSOP_ELEMENTWISE_ABS = 0,  /**< TOPSOP_ELEMENTWISE_ABS       0  */
  TOPSOP_ELEMENTWISE_ADD,      /**< TOPSOP_ELEMENTWISE_ADD       1  */
  TOPSOP_ELEMENTWISE_AND,      /**< TOPSOP_ELEMENTWISE_ADD       2  */
  TOPSOP_ELEMENTWISE_ACOS,     /**< TOPSOP_ELEMENTWISE_ACOS      3  */
  TOPSOP_ELEMENTWISE_ACOSH,    /**< TOPSOP_ELEMENTWISE_ACOSH     4  */
  TOPSOP_ELEMENTWISE_ASIN,     /**< TOPSOP_ELEMENTWISE_ASIN      5  */
  TOPSOP_ELEMENTWISE_ASINH,    /**< TOPSOP_ELEMENTWISE_ASINH     6  */
  TOPSOP_ELEMENTWISE_ATAN,     /**< TOPSOP_ELEMENTWISE_ATAN      7  */
  TOPSOP_ELEMENTWISE_ATANH,    /**< TOPSOP_ELEMENTWISE_ATANH     8  */
  TOPSOP_ELEMENTWISE_CEIL,     /**< TOPSOP_ELEMENTWISE_CEIL      9  */
  TOPSOP_ELEMENTWISE_COS,      /**< TOPSOP_ELEMENTWISE_COS       10 */
  TOPSOP_ELEMENTWISE_COSH,     /**< TOPSOP_ELEMENTWISE_COSH      11 */
  TOPSOP_ELEMENTWISE_DIV,      /**< TOPSOP_ELEMENTWISE_DIV       12 */
  TOPSOP_ELEMENTWISE_EQ,       /**< TOPSOP_ELEMENTWISE_EQ        13 */
  TOPSOP_ELEMENTWISE_EXP,      /**< TOPSOP_ELEMENTWISE_EXP       14 */
  TOPSOP_ELEMENTWISE_FLOOR,    /**< TOPSOP_ELEMENTWISE_FLOOR     15 */
  TOPSOP_ELEMENTWISE_GT,       /**< TOPSOP_ELEMENTWISE_GT        16 */
  TOPSOP_ELEMENTWISE_GE,       /**< TOPSOP_ELEMENTWISE_GE        17 */
  TOPSOP_ELEMENTWISE_GELU,     /**< TOPSOP_ELEMENTWISE_GELU      18 */
  TOPSOP_ELEMENTWISE_GELUGRAD, /**< TOPSOP_ELEMENTWISE_GELUGRAD  19 */
  TOPSOP_ELEMENTWISE_LE,       /**< TOPSOP_ELEMENTWISE_LE        20 */
  TOPSOP_ELEMENTWISE_LOG,      /**< TOPSOP_ELEMENTWISE_LOG       21 */
  TOPSOP_ELEMENTWISE_LT,       /**< TOPSOP_ELEMENTWISE_LT        22 */
  TOPSOP_ELEMENTWISE_MAX,      /**< TOPSOP_ELEMENTWISE_MAX       23 */
  TOPSOP_ELEMENTWISE_MIN,      /**< TOPSOP_ELEMENTWISE_MIN       24 */
  TOPSOP_ELEMENTWISE_MOD,      /**< TOPSOP_ELEMENTWISE_MOD       25 */
  TOPSOP_ELEMENTWISE_MUL,      /**< TOPSOP_ELEMENTWISE_MUL       26 */
  TOPSOP_ELEMENTWISE_NE,       /**< TOPSOP_ELEMENTWISE_NE        27 */
  TOPSOP_ELEMENTWISE_NEG,      /**< TOPSOP_ELEMENTWISE_NEG       28 */
  TOPSOP_ELEMENTWISE_NOT,      /**< TOPSOP_ELEMENTWISE_NOT       29 */
  TOPSOP_ELEMENTWISE_OR,       /**< TOPSOP_ELEMENTWISE_OR        30 */
  TOPSOP_ELEMENTWISE_POWER,    /**< TOPSOP_ELEMENTWISE_POWER     31 */
  TOPSOP_ELEMENTWISE_REM,      /**< TOPSOP_ELEMENTWISE_REM       32 */
  TOPSOP_ELEMENTWISE_RSQRT,    /**< TOPSOP_ELEMENTWISE_RSQRT     33 */
  TOPSOP_ELEMENTWISE_SIGN,     /**< TOPSOP_ELEMENTWISE_SIGN      34 */
  TOPSOP_ELEMENTWISE_SIN,      /**< TOPSOP_ELEMENTWISE_SIN       35 */
  TOPSOP_ELEMENTWISE_SINH,     /**< TOPSOP_ELEMENTWISE_SINH      36 */
  TOPSOP_ELEMENTWISE_SQRT,     /**< TOPSOP_ELEMENTWISE_SQRT      37 */
  TOPSOP_ELEMENTWISE_SUB,      /**< TOPSOP_ELEMENTWISE_SUB       38 */
  TOPSOP_ELEMENTWISE_TAN,      /**< TOPSOP_ELEMENTWISE_TAN       39 */
  TOPSOP_ELEMENTWISE_TANH,     /**< TOPSOP_ELEMENTWISE_TANH      40 */
} topsopElementwiseOpType_t;

typedef enum {
  /**< TOPSOP_POOLING_MAX 0  */
  TOPSOP_POOLING_MAX = 0,
  /**< TOPSOP_POOLING_AVERAGE_COUNT_INCLUDE_PADDING 1  */
  TOPSOP_POOLING_AVERAGE_COUNT_INCLUDE_PADDING = 1,
  /**< TOPSOP_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING 2  */
  TOPSOP_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING = 2,
} topsopPoolingMode_t;

typedef enum {
  /**< TOPSOP_REDUCE_WINDOW_AUTO_PAD_NOTSET 0  */
  TOPSOP_REDUCE_WINDOW_AUTO_PAD_NOTSET = 0,
  /**< TOPSOP_REDUCE_WINDOW_AUTO_PAD_SAME_UPPER 1  */
  TOPSOP_REDUCE_WINDOW_AUTO_PAD_SAME_UPPER = 1,
  /**< TOPSOP_REDUCE_WINDOW_AUTO_PAD_SAME_LOWER 2  */
  TOPSOP_REDUCE_WINDOW_AUTO_PAD_SAME_LOWER = 2,
  /**< TOPSOP_REDUCE_WINDOW_AUTO_PAD_VALID 3  */
  TOPSOP_REDUCE_WINDOW_AUTO_PAD_VALID = 3,
} topsopReduceWindowAutoPad_t;

typedef enum {
  /**< TOPSOP_REDUCE_WINDOW_COMPUTATION_ADD 0  */
  TOPSOP_REDUCE_WINDOW_COMPUTATION_ADD = 0,
  /**< TOPSOP_REDUCE_WINDOW_COMPUTATION_MAX 1  */
  TOPSOP_REDUCE_WINDOW_COMPUTATION_MAX = 1,
  /**< TOPSOP_REDUCE_WINDOW_COMPUTATION_MIN 2  */
  TOPSOP_REDUCE_WINDOW_COMPUTATION_MIN = 2,
  /**< TOPSOP_REDUCE_WINDOW_COMPUTATION_MUL 3  */
  TOPSOP_REDUCE_WINDOW_COMPUTATION_MUL = 3,
} topsopReduceWindowComputation_t;

/** topsopResizeCoordTransMode_t */
typedef enum {
  TOPSOP_RESIZE_HALF_PIXEL = 0, /**< TOPSOP_RESIZE_HALF_PIXEL 0  */
  TOPSOP_RESIZE_ASYMMETRIC = 1, /**< TOPSOP_RESIZE_ASYMMETRIC 1  */
  TOPSOP_RESIZE_PYTORCH_HALF_PIXEL =
      2,                           /**< TOPSOP_RESIZE_PYTORCH_HALF_PIXEL 2  */
  TOPSOP_RESIZE_TF_HALF_PIXEL = 3, /**< TOPSOP_RESIZE_TF_HALF_PIXEL 3  */
  TOPSOP_RESIZE_ALIGN_CORNERS = 4, /**< TOPSOP_RESIZE_ALIGN_CORNERS 4  */
} topsopResizeCoordTransMode_t;

/** topsopResizeInterpolationMode_t */
typedef enum {
  TOPSOP_RESIZE_NEAREST = 0,  /**< TOPSOP_RESIZE_NEAREST  0  */
  TOPSOP_RESIZE_BILINEAR = 1, /**< TOPSOP_RESIZE_BILINEAR 1  */
} topsopResizeInterpolationMode_t;

/** topsopResizeNearestMode_t */
typedef enum {
  TOPSOP_RESIZE_ROUND_PREFER_FLOOR =
      1, /**< TOPSOP_RESIZE_ROUND_PREFER_FLOOR  1  */
  TOPSOP_RESIZE_ROUND_PREFER_CEIL =
      2,                   /**< TOPSOP_RESIZE_ROUND_PREFER_CEIL  2  */
  TOPSOP_RESIZE_FLOOR = 3, /**< TOPSOP_RESIZE_FLOOR  3  */
  TOPSOP_RESIZE_CEIL = 4,  /**< TOPSOP_RESIZE_CEIL  4  */
} topsopResizeNearestMode_t;

/** topsopTopkCmpMode_t */
typedef enum {
  TOPSOP_TOPK_TYPE_INVALID = 0, /**< TOPSOP_TOPK_TYPE_INVALID 0 */
  TOPSOP_TOPK_TYPE_MAX = 1,     /**< TOPSOP_TOPK_TYPE_MAX 1 */
  TOPSOP_TOPK_TYPE_MIN = 2,     /**< TOPSOP_TOPK_TYPE_MIN 2 */
} topsopTopkCmpMode_t;

/** topsopSortCmpMode_t */
typedef enum {
  TOPSOP_SORT_TYPE_INVALID = 0, /**< TOPSOP_SORT_TYPE_INVALID 0 */
  TOPSOP_SORT_TYPE_ASCEND = 1,  /**< TOPSOP_SORT_TYPE_ASCEND 1 */
  TOPSOP_SORT_TYPE_DESCEND = 2, /**< TOPSOP_SORT_TYPE_DESCEND 2 */
} topsopSortCmpMode_t;

/** topsopSortStableMode_t */
typedef enum {
  TOPSOP_SORT_STABLE_INVALID = 0, /**< TOPSOP_SORT_STABLE_INVALID 0 */
  TOPSOP_SORT_STABLE = 1,         /**< TOPSOP_SORT_STABLE   1*/
  TOPSOP_SORT_INSTABLE = 2,       /**< TOPSOP_SORT_INSTABLE 2*/
} topsopSortStableMode_t;

typedef enum {
  /* bnScale, bnBias tensor dims are 1xCxHxWx */
  TOPSOP_BATCHNORM_PER_ACTIVATION = 0,

  /* bnScale, bnBias tensor dims are 1xCx1x1 */
  TOPSOP_BATCHNORM_SPATIAL = 1,

  /*
   * bnScale, bnBias tensor dims are 1xCx1x1.
   * May be faster but imposes some limits on the range of values
   */
  TOPSOP_BATCHNORM_SPATIAL_PERSISTENT = 2,
} topsopBatchNormMode_t;

/** topsopSelectType_t */
typedef enum {
  TOPSOP_SELECT_GE = 0, /**< TOPSOP_SELECT_GE 0 */
  TOPSOP_SELECT_LE = 1, /**< TOPSOP_SELECT_LE 1 */
  TOPSOP_SELECT_GT = 2, /**< TOPSOP_SELECT_GT 2 */
  TOPSOP_SELECT_LT = 3, /**< TOPSOP_SELECT_LT 3 */
} topsopSelectType_t;

/** topsopScatterType_t */
typedef enum {
  TOPSOP_SCATTER_ADD = 0, /**< TOPSOP_SCATTER_ADD 0 */
  TOPSOP_SCATTER_SUB = 1, /**< TOPSOP_SCATTER_SUB 1 */
  TOPSOP_SCATTER_MUL = 2, /**< TOPSOP_SCATTER_MUL 2 */
  TOPSOP_SCATTER_MAX = 3, /**< TOPSOP_SCATTER_MAX 3 */
  TOPSOP_SCATTER_MIN = 4, /**< TOPSOP_SCATTER_MIN 4 */
} topsopScatterType_t;

/** topsopScatterComputationType_t */
typedef enum {
  TOPSOP_SCATTER_COMP_UPDATE = 0, /**< TOPSOP_SCATTER_COMP_UPDATE 0 */
  TOPSOP_SCATTER_COMP_ADD = 1,    /**< TOPSOP_SCATTER_COMP_ADD 1 */
  TOPSOP_SCATTER_COMP_SUB = 2,    /**< TOPSOP_SCATTER_COMP_SUB 2 */
  TOPSOP_SCATTER_COMP_NUM = 3,    /**< TOPSOP_SCATTER_COMP_NUM 3 */
} topsopScatterComputationType_t;

#define TOPSOP_LRN_MIN_N 1       /* minimum allowed lrnN */
#define TOPSOP_LRN_MAX_N 16      /* maximum allowed lrnN */
#define TOPSOP_LRN_MIN_K 1e-5    /* minimum allowed lrnK */
#define TOPSOP_LRN_MIN_BETA 0.01 /* minimum allowed lrnBeta */

/** topsopLRNMode_t */
typedef enum {
  TOPSOP_LRN_CROSS_CHANNEL_DIM1 = 0, /**< TOPSOP_LRN_CROSS_CHANNEL_DIM1 0 */
} topsopLRNMode_t;

typedef enum {
  TOPSOP_CONVOLUTION_FWD_ALGO_AUTO = 0,
  TOPSOP_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM = 1,
  TOPSOP_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM = 2,
  TOPSOP_CONVOLUTION_FWD_ALGO_GEMM = 3,
  TOPSOP_CONVOLUTION_FWD_ALGO_DIRECT = 4,
  TOPSOP_CONVOLUTION_FWD_ALGO_FFT = 5,
  TOPSOP_CONVOLUTION_FWD_ALGO_FFT_TILING = 6,
  TOPSOP_CONVOLUTION_FWD_ALGO_WINOGRAD = 7,
  TOPSOP_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED = 8,
  TOPSOP_CONVOLUTION_FWD_ALGO_COUNT = 9
} topsopConvolutionFwdAlgo_t;

typedef enum {
  TOPSOP_CONVOLUTION_BWD_FILTER_ALGO_0 = 0,
  TOPSOP_CONVOLUTION_BWD_FILTER_ALGO_1 = 1,
  TOPSOP_CONVOLUTION_BWD_FILTER_ALGO_FFT = 2,
  TOPSOP_CONVOLUTION_BWD_FILTER_ALGO_3 = 3,
  TOPSOP_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD = 4,
  TOPSOP_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED = 5,
  TOPSOP_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING = 6,
  TOPSOP_CONVOLUTION_BWD_FILTER_ALGO_COUNT = 7
} topsopConvolutionBwdFilterAlgo_t;

/** topsopRoiAlignOpType_t */
typedef enum {
  TOPSOP_ROIALIGN_AVG_FP32 = 0,
  TOPSOP_ROIALIGN_AVG_FP16 = 1,
  TOPSOP_ROIALIGN_MAX_FP32 = 2,
  TOPSOP_ROIALIGN_MAX_FP16 = 3,
} topsopRoiAlignOpType_t;

typedef enum {
  TOPSOP_CONVOLUTION_BWD_DATA_ALGO_0 = 0,
  TOPSOP_CONVOLUTION_BWD_DATA_ALGO_1 = 1,
  TOPSOP_CONVOLUTION_BWD_DATA_ALGO_FFT = 2,
  TOPSOP_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING = 3,
  TOPSOP_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD = 4,
  TOPSOP_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED = 5,
  TOPSOP_CONVOLUTION_BWD_DATA_ALGO_COUNT = 6
} topsopConvolutionBwdDataAlgo_t;

/** topsopRoiAlignMode_t */
typedef enum {
  TOPSOP_ROIALIGN_AVG = 0,
  TOPSOP_ROIALIGN_MAX = 1,
} topsopRoiAlignMode_t;

/** topsopPadMode_t */
typedef enum {
  TOPSOP_PAD_CONSTANT = 0,
  TOPSOP_PAD_REFLECT = 1,
  TOPSOP_PAD_EDGE = 2,
  TOPSOP_PAD_REPLICATE = 3,
  TOPSOP_PAD_CIRCULAR = 4,
  TOPSOP_PAD_MAXIMUM = 5,
  TOPSOP_PAD_LINEAR_RAMP = 6,
  TOPSOP_PAD_MEAN = 7,
  TOPSOP_PAD_MEDIAN = 8,
  TOPSOP_PAD_MINIMUM = 9,
  TOPSOP_PAD_SYMMETRIC = 10,
  TOPSOP_PAD_WRAP = 11,
  TOPSOP_PAD_EMPTY = 12
} topsopPadMode_t;

/** topsopRoiAlignCoordinateTransformationMode_t */
typedef enum {
  TOPSOP_ROIALIGN_HALF_PIXL_TRUE = 0,
  TOPSOP_ROIALIGN_HALF_PIXL_FALSE = 1,
} topsopRoiAlignCoordinateTransformationMode_t;

/** topsopColorConversionCodes_t */
typedef enum {
  TOPSOP_COLORCVT_YUV2RGB = 0,  /**< YUV444packed to RGB888packed */
  TOPSOP_COLORCVT_YUV2BGR = 1,  /**< YUV444packed to BGR888packed */
  TOPSOP_COLORCVT_RGB2GRAY = 2, /**< RGB888packed to GRAY */
  TOPSOP_COLORCVT_BGR2GRAY = 3, /**< BGR888packed to GRAY */
} topsopColorConversionCodes_t;

typedef enum {
  TOPSOP_BLAS_OP_N = 0,
  TOPSOP_BLAS_OP_T = 1,
  TOPSOP_BLAS_OP_C = 2,        /* >= 2 not supported in the current release */
  TOPSOP_BLAS_OP_HERMITAN = 2, /* synonym if TOPSBLAS_OP_C */
  TOPSOP_BLAS_OP_CONJG = 3     /* conjugate, placeholder*/
} topsopBlasOperation_t;

/** topsopLayoutType_t */
typedef enum {
  TOPSOP_LAYOUT_STRIDED = 0, /**< TOPSOP_LAYOUT_STRIDED 0 */
  TOPSOP_LAYOUT_SPARSE = 1,  /**< TOPSOP_LAYOUT_SPARSE 1 */
} topsopLayoutType_t;

typedef enum {
  TOPSOP_DEFAULT_ALGO = 0,
  TOPSOP_PHILOX4_32_ALGO,
} topsopAlgorithm_t;

/**
 * A struct to descript a data pointer and its length
 */
typedef struct {
  /*@{*/
  const int64_t* data; /**< pointer to the data  */
  int64_t len;         /**< length of the data  */
  /*@}*/

  // #if defined(__cplusplus)
  //   topsopSize_t_() : data(nullptr), len(0) {}
  //   topsopSize_t_(const int64_t* d, int64_t l) : data(d), len(l) {}
  // #endif
} topsopSize_t;

/**
 * A struct to descript a scalar
 */
typedef struct {
  /*@{*/
  topsopDataType_t dtype; /**< data type  */
                          /*@}*/
                          /**
                           * union of actual value
                           */
  /*@{*/
  union {
    double fval;  /**< double type value  */
    int64_t ival; /**< long long type value  */
  };
  /*@}*/
} topsopScalar_t;

/**
 * A struct to descript a random number generator
 */
typedef struct topsopGenerator_s {
  /*@{*/
  uint64_t seed;                                /**< seed for generator  */
  uint64_t offset;                              /**< offset for generator  */
  topsopAlgorithm_t algo = TOPSOP_DEFAULT_ALGO; /**< algo for generator */
  /*@}*/
} topsopGenerator_t;

/**
 * @brief The type of device memory handle.
 *
 */
typedef void* topsopDeviceMemHandle_t;

/**
 * @brief the type of tensor handle.
 *
 */
typedef struct topsopTensorStruct* topsopTensorHandle_t;

typedef char* topsopVersion_t;

/**
 * @brief get the version of the Device-Independent Operator Inetrface
 */
const topsopVersion_t TOPSOP_EXPORT topsopGetVersion();

/**
 * @brief This function initialize the topsOP library, this function must be
 * called prior to
 * making any other topsOP calls.
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopInit();

/**
 * @brief This function finalize the topOP library, this function is usually the
 * last call
 * made to topsOP.
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopFinalize();

/**
 * @brief This function creates a generic tensor object by allocating the memory
 *        needed to hold its opaque structure.
 *
 * @param dims The dims of the created tensor. By convention, the ordering of
 * dimensions in the array follows the format - [N, C, D, H, W], with W
 * occupying the smallest index in the array.
 * @param strides The strides of the created tensor.By convention, the ordering
 * of the strides in the array follows the format - [Nstride, Cstride, Dstride,
 * Hstride, Wstride], with Wstride occupying the smallest index in the array.
 * @param dtype The data type of the created tensor.
 * @param memhandle The pointer of the memory handle which storage the tensor
 * data.
 * @param tensor The pointer of the created tensor object.
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopCreateTensor(topsopSize_t* dims,
                   topsopSize_t* strides,
                   topsopDataType_t dtype,
                   topsopDeviceMemHandle_t memhandle,
                   topsopTensorHandle_t* tensor);

/**
 * @brief This function creates a generic tensor object by allocating the memory
 *        needed to hold its opaque structure, different from
 * topsopCreateTensor.
 *        It support more data format like filter RSCiCo, but it does not check
 *        whether data format is NHWC/NCHW.
 *
 * @param dims The dims of the created tensor. By convention, the ordering of
 * dimensions in the array follows the format - [N, C, D, H, W], with W
 * occupying the smallest index in the array.
 * @param strides The strides of the created tensor.By convention, the ordering
 * of the strides in the array follows the format - [Nstride, Cstride, Dstride,
 * Hstride, Wstride], with Wstride occupying the smallest index in the array.
 * @param dtype The data type of the created tensor.
 * @param memhandle The pointer of the memory handle which storage the tensor
 * data.
 * @param tensor The pointer of the created tensor object.
 * @return topsopStatus_t
 */
topsopStatus_t topsopCreateTensorEx(topsopSize_t* dims,
                                    topsopSize_t* strides,
                                    topsopDataType_t dtype,
                                    topsopDeviceMemHandle_t memhandle,
                                    topsopTensorHandle_t* tensor);

/**
 * @brief This function sets offset to a tensor
 *
 * @param tensor The pointer of the tensor.
 * @param offset The storage offset num_elements of the tensor
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopSetOffset(topsopTensorHandle_t tensor,
                                             int64_t offset);

/**
 * @brief This function gets offset to a tensor
 *
 * @param tensor The pointer of the tensor.
 * @param offset Pointer to the storage offset num_elements of the tensor
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopGetOffset(topsopTensorHandle_t tensor,
                                             int64_t* offset);

/**
 * @brief This function destroys a previously created tensor object.
 * When the input pointer is NULL, this function performs no destroy operation.
 *
 * @param tensor Pointer to the tensor object to be destroyed.
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopDestroyTensor(topsopTensorHandle_t tensor);

/**
 * @brief This function gets the memory handle which storage the tensor data
 *
 * @param tensor Pointer to the tensor object.
 * @param data Pointer to the memory handle of the tensor object.
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopGetTensorData(topsopTensorHandle_t tensor,
                                                 void** data);

/**
 * @brief This function gets the memory handle which storage the tensor data.
 *
 * @param tensor Pointer to the tensor object.
 * @param data Const pointer to the memory handle of the tensor object.
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopGetTensorDataConst(const topsopTensorHandle_t tensor, const void** data);

/**
 * @brief This function gets the dimentions size of the tensor.
 *
 * @param tensor Pointer to the tensor object.
 * @param size Pointer to the dimentions size of the tensor shape.
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopGetTensorShape(const topsopTensorHandle_t tensor, topsopSize_t* size);

/**
 * @brief This function gets the strides of the tensor.
 *
 * @param tensor Pointer to the tensor object.
 * @param stride Pointer to the strides of the tensor shape.
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopGetTensorStride(const topsopTensorHandle_t tensor, topsopSize_t* stride);

/**
 * @brief This function gets the data type of the tensor.
 *
 * @param tensor Pointer to the tensor object.
 * @param dtype Pointer to the data type of the tensor shape.
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopGetTensorDataType(
    const topsopTensorHandle_t tensor, topsopDataType_t* dtype);

/**
 * @brief This function gets total elements number of the tensor.
 *
 * @param tensor Pointer to the tensor object.
 * @param nums Pointer to the elements number.
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopGetTensorElementNums(const topsopTensorHandle_t tensor, int64_t* nums);

/**
 * @brief This function gets bytes per element of the tensor.
 *
 * @param tensor Pointer to the tensor object.
 * @param itemsize Pointer to bpe.
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopGetTensorBPE(const topsopTensorHandle_t tensor, int64_t* itemsize);

/**
 * @brief This function generate strides by tensor format, only support 4D and
 * 5D strides.
 *
 * @param format Tensor format, which is TOPSOP_TENSOR_NHWC or
 * TOPSOP_TENSOR_NCHW.
 * @param dims Pointer to the dimention array,the ordering of dimensions
 * in the array follows the format - [N, C, D, H, W].
 * @param strides Pointer to the stride array,the ordering of the strides in the
 * array follows the format - [Nstride, Cstride, Dstride, Hstride, Wstride].
 * @param rank The rank of the stride array.
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopGenerateStridesByFormat(const topsopTensorFormat_t format,
                              const int64_t rank,
                              const int64_t* dims,
                              int64_t* strides);

/**
 * @brief This function judges whether the tensor is contiguous or not
 *
 * @param tensor Pointer to the tensor object.
 * @param format Memory format,
 * value includes TOPSOP_MEMORY_CONTIGUOUS, TOPSOP_MEMORY_NHWC,
 *                TOPSOP_MEMORY_NDHWC or TOPSOP_MEMORY_PRESERVE
 * @param result The boolean result of this function
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopIsTensorContiguous(const topsopTensorHandle_t tensor,
                         const topsopMemoryFormat_t format,
                         bool* result);

topsopStatus_t TOPSOP_EXPORT topsopIsTensorNonOverlappingandDense(
    const topsopTensorHandle_t tensor, bool* result);

#if defined(__cplusplus)
}
#endif

#endif /* TOPSOP_DEFINE_H_ */  // NOLINT

// Doxygen end group topsop_define.h
/** @} */
