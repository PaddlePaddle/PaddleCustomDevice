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
 *     @defgroup ops
 *     @{
 *
 */

/**
 * @file topsop_ops.h
 * @brief topsflame common ops api definitions.
 */

#ifndef TOPSOP_OPS_H_  // NOLINT
#define TOPSOP_OPS_H_

#include "tops/tops_runtime.h"
#include "topsop/topsop_define.h"

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * @brief This function adds the scaled values of a bias tensor to another
 *        tensor.
 * @param out Result tensor
 * @param lhs Input lhs tensor
 * @param rhs Input rhs tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopAddDemo(topsopTensorHandle_t out,
                                           const topsopTensorHandle_t lhs,
                                           const topsopTensorHandle_t rhs,
                                           const topsopScalar_t alpha,
                                           const topsopScalar_t beta,
                                           topsStream_t stream);
/**
 * @brief check whether current addDemo operator support or not
 *
 * @param lhs Input lhs tensor
 * @param rhs Input rhs tensor
 * @return bool
 */
bool TOPSOP_EXPORT topsopAddDemoIsSupported(const topsopTensorHandle_t lhs,
                                            const topsopTensorHandle_t rhs);
/**
 * @brief get output dims of current tensor addDemo operator
 *
 * @param lhs Input lhs tensor
 * @param rhs Input rhs tensor
 * @param dims The dims of output tensor
 * @param rank The rank of output tensor
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopAddDemoGetOutputDim(const topsopTensorHandle_t lhs,
                          const topsopTensorHandle_t rhs,
                          int64_t *dims,
                          int64_t *rank);

/**
 * @brief add operator
 *
 * @param out The output tensor
 * @param lhs Input lhs tensor
 * @param rhs Input rhs tensor
 * @param alpha1 The multiplier for lhs tensor
 * @param alpha2 The multiplier for rhs tensor
 * @param beta The multiplier for output tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopAdd(topsopTensorHandle_t out,
                                       const topsopTensorHandle_t lhs,
                                       const topsopTensorHandle_t rhs,
                                       const topsopScalar_t alpha1,
                                       const topsopScalar_t alpha2,
                                       const topsopScalar_t beta,
                                       topsStream_t stream);
/**
 * @brief check whether current add operator support or not
 *
 * @param lhs Input lhs tensor
 * @param rhs Input rhs tensor
 * @return bool
 */
bool TOPSOP_EXPORT topsopAddIsSupported(const topsopTensorHandle_t lhs,
                                        const topsopTensorHandle_t rhs);
/**
 * @brief get output dims of current tensor add operator
 *
 * @param lhs Input lhs tensor
 * @param rhs Input rhs tensor
 * @param dims The dims of output tensor
 * @param rank The rank of output tensor
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopAddGetOutputDim(const topsopTensorHandle_t lhs,
                      const topsopTensorHandle_t rhs,
                      int64_t *dims,
                      int64_t *rank);

/**
 * @brief sub operator
 *
 * @param out The output tensor
 * @param lhs Input lhs tensor
 * @param rhs Input rhs tensor
 * @param alpha1 The multiplier for lhs tensor
 * @param alpha2 The multiplier for rhs tensor
 * @param beta The multiplier for output tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopSub(topsopTensorHandle_t out,
                                       const topsopTensorHandle_t lhs,
                                       const topsopTensorHandle_t rhs,
                                       const topsopScalar_t alpha1,
                                       const topsopScalar_t alpha2,
                                       const topsopScalar_t beta,
                                       topsStream_t stream);
/**
 * @brief check whether current sub operator support or not
 *
 * @param lhs Input lhs tensor
 * @param rhs Input rhs tensor
 * @return bool
 */
bool TOPSOP_EXPORT topsopSubIsSupported(const topsopTensorHandle_t lhs,
                                        const topsopTensorHandle_t rhs);
/**
 * @brief get output dims of current tensor sub operator
 *
 * @param lhs Input lhs tensor
 * @param rhs Input rhs tensor
 * @param dims The dims of output tensor
 * @param rank The rank of output tensor
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopSubGetOutputDim(const topsopTensorHandle_t lhs,
                      const topsopTensorHandle_t rhs,
                      int64_t *dims,
                      int64_t *rank);

/**
 * @brief mul operator
 *
 * @param out The output tensor
 * @param lhs Input lhs tensor
 * @param rhs Input rhs tensor
 * @param alpha1 The multiplier for lhs tensor
 * @param alpha2 The multiplier for rhs tensor
 * @param beta The multiplier for output tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopMul(topsopTensorHandle_t out,
                                       const topsopTensorHandle_t lhs,
                                       const topsopTensorHandle_t rhs,
                                       const topsopScalar_t alpha1,
                                       const topsopScalar_t alpha2,
                                       const topsopScalar_t beta,
                                       topsStream_t stream);
/**
 * @brief check whether current mul operator support or not
 *
 * @param lhs Input lhs tensor
 * @param rhs Input rhs tensor
 * @return bool
 */
bool TOPSOP_EXPORT topsopMulIsSupported(const topsopTensorHandle_t lhs,
                                        const topsopTensorHandle_t rhs);
/**
 * @brief get output dims of current tensor mul operator
 *
 * @param lhs Input lhs tensor
 * @param rhs Input rhs tensor
 * @param dims The dims of output tensor
 * @param rank The rank of output tensor
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopMulGetOutputDim(const topsopTensorHandle_t lhs,
                      const topsopTensorHandle_t rhs,
                      int64_t *dims,
                      int64_t *rank);

/**
 * @brief div operator
 *
 * @param out The output tensor
 * @param lhs Input lhs tensor
 * @param rhs Input rhs tensor
 * @param alpha1 The multiplier for lhs tensor
 * @param alpha2 The multiplier for rhs tensor
 * @param beta The multiplier for output tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopDiv(topsopTensorHandle_t out,
                                       const topsopTensorHandle_t lhs,
                                       const topsopTensorHandle_t rhs,
                                       const topsopScalar_t alpha1,
                                       const topsopScalar_t alpha2,
                                       const topsopScalar_t beta,
                                       topsStream_t stream);
/**
 * @brief check whether current div operator support or not
 *
 * @param lhs Input lhs tensor
 * @param rhs Input rhs tensor
 * @return bool
 */
bool TOPSOP_EXPORT topsopDivIsSupported(const topsopTensorHandle_t lhs,
                                        const topsopTensorHandle_t rhs);
/**
 * @brief get output dims of current tensor div operator
 *
 * @param lhs Input lhs tensor
 * @param rhs Input rhs tensor
 * @param dims The dims of output tensor
 * @param rank The rank of output tensor
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopDivGetOutputDim(const topsopTensorHandle_t lhs,
                      const topsopTensorHandle_t rhs,
                      int64_t *dims,
                      int64_t *rank);

/**
 * @brief power operator
 *
 * @param out The output tensor
 * @param lhs Input lhs tensor
 * @param rhs Input rhs tensor
 * @param alpha1 The multiplier for lhs tensor
 * @param alpha2 The multiplier for rhs tensor
 * @param beta The multiplier for output tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopPower(topsopTensorHandle_t out,
                                         const topsopTensorHandle_t lhs,
                                         const topsopTensorHandle_t rhs,
                                         const topsopScalar_t alpha1,
                                         const topsopScalar_t alpha2,
                                         const topsopScalar_t beta,
                                         topsStream_t stream);
/**
 * @brief check whether current power operator support or not
 *
 * @param lhs Input lhs tensor
 * @param rhs Input rhs tensor
 * @return bool
 */
bool TOPSOP_EXPORT topsopPowerIsSupported(const topsopTensorHandle_t lhs,
                                          const topsopTensorHandle_t rhs);
/**
 * @brief get output dims of current tensor power operator
 *
 * @param lhs Input lhs tensor
 * @param rhs Input rhs tensor
 * @param dims The dims of output tensor
 * @param rank The rank of output tensor
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopPowerGetOutputDim(const topsopTensorHandle_t lhs,
                        const topsopTensorHandle_t rhs,
                        int64_t *dims,
                        int64_t *rank);

/**
 * @brief max operator
 *
 * @param out The output tensor
 * @param lhs Input lhs tensor
 * @param rhs Input rhs tensor
 * @param nanOpt an enumerated type used to indicate if a given routine
 * should propagate Nan numbers
 * @param alpha1 The multiplier for lhs tensor
 * @param alpha2 The multiplier for rhs tensor
 * @param beta The multiplier for output tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopMax(topsopTensorHandle_t out,
                                       const topsopTensorHandle_t lhs,
                                       const topsopTensorHandle_t rhs,
                                       const topsopNanPropagation_t nanOpt,
                                       const topsopScalar_t alpha1,
                                       const topsopScalar_t alpha2,
                                       const topsopScalar_t beta,
                                       topsStream_t stream);
/**
 * @brief check whether current max operator support or not
 *
 * @param lhs Input lhs tensor
 * @param rhs Input rhs tensor
 * @return bool
 */
bool TOPSOP_EXPORT topsopMaxIsSupported(const topsopTensorHandle_t lhs,
                                        const topsopTensorHandle_t rhs);
/**
 * @brief get output dims of current tensor max operator
 *
 * @param lhs Input lhs tensor
 * @param rhs Input rhs tensor
 * @param dims The dims of output tensor
 * @param rank The rank of output tensor
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopMaxGetOutputDim(const topsopTensorHandle_t lhs,
                      const topsopTensorHandle_t rhs,
                      int64_t *dims,
                      int64_t *rank);

/**
 * @brief min operator
 *
 * @param out The output tensor
 * @param lhs Input lhs tensor
 * @param rhs Input rhs tensor
 * @param nanOpt an enumerated type used to indicate if a given routine
 * should propagate Nan numbers
 * @param alpha1 The multiplier for lhs tensor
 * @param alpha2 The multiplier for rhs tensor
 * @param beta The multiplier for output tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopMin(topsopTensorHandle_t out,
                                       const topsopTensorHandle_t lhs,
                                       const topsopTensorHandle_t rhs,
                                       const topsopNanPropagation_t nanOpt,
                                       const topsopScalar_t alpha1,
                                       const topsopScalar_t alpha2,
                                       const topsopScalar_t beta,
                                       topsStream_t stream);
/**
 * @brief check whether current min operator support or not
 *
 * @param lhs Input lhs tensor
 * @param rhs Input rhs tensor
 * @return bool
 */
bool TOPSOP_EXPORT topsopMinIsSupported(const topsopTensorHandle_t lhs,
                                        const topsopTensorHandle_t rhs);
/**
 * @brief get output dims of current tensor min operator
 *
 * @param lhs Input lhs tensor
 * @param rhs Input rhs tensor
 * @param dims The dims of output tensor
 * @param rank The rank of output tensor
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopMinGetOutputDim(const topsopTensorHandle_t lhs,
                      const topsopTensorHandle_t rhs,
                      int64_t *dims,
                      int64_t *rank);

/**
 * @brief remainder operator
 *
 * @param out The output tensor
 * @param lhs Input lhs tensor
 * @param rhs Input rhs tensor
 * @param alpha1 The multiplier for lhs tensor
 * @param alpha2 The multiplier for rhs tensor
 * @param beta The multiplier for output tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopRem(topsopTensorHandle_t out,
                                       const topsopTensorHandle_t lhs,
                                       const topsopTensorHandle_t rhs,
                                       const topsopScalar_t alpha1,
                                       const topsopScalar_t alpha2,
                                       const topsopScalar_t beta,
                                       topsStream_t stream);
/**
 * @brief check whether current remainder operator support or not
 *
 * @param lhs Input lhs tensor
 * @param rhs Input rhs tensor
 * @return bool
 */
bool TOPSOP_EXPORT topsopRemIsSupported(const topsopTensorHandle_t lhs,
                                        const topsopTensorHandle_t rhs);
/**
 * @brief get output dims of current tensor remainder operator
 *
 * @param lhs Input lhs tensor
 * @param rhs Input rhs tensor
 * @param dims The dims of output tensor
 * @param rank The rank of output tensor
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopRemGetOutputDim(const topsopTensorHandle_t lhs,
                      const topsopTensorHandle_t rhs,
                      int64_t *dims,
                      int64_t *rank);

/**
 * @brief and operator
 *
 * @param out The output tensor
 * @param lhs Input lhs tensor
 * @param rhs Input rhs tensor
 * @param alpha1 The multiplier for lhs tensor
 * @param alpha2 The multiplier for rhs tensor
 * @param beta The multiplier for output tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopAnd(topsopTensorHandle_t out,
                                       const topsopTensorHandle_t lhs,
                                       const topsopTensorHandle_t rhs,
                                       const topsopScalar_t alpha1,
                                       const topsopScalar_t alpha2,
                                       const topsopScalar_t beta,
                                       topsStream_t stream);
/**
 * @brief check whether current and operator support or not
 *
 * @param lhs Input lhs tensor
 * @param rhs Input rhs tensor
 * @return bool
 */
bool TOPSOP_EXPORT topsopAndIsSupported(const topsopTensorHandle_t lhs,
                                        const topsopTensorHandle_t rhs);
/**
 * @brief get output dims of current tensor and operator
 *
 * @param lhs Input lhs tensor
 * @param rhs Input rhs tensor
 * @param dims The dims of output tensor
 * @param rank The rank of output tensor
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopAndGetOutputDim(const topsopTensorHandle_t lhs,
                      const topsopTensorHandle_t rhs,
                      int64_t *dims,
                      int64_t *rank);

/**
 * @brief or operator
 *
 * @param out The output tensor
 * @param lhs Input lhs tensor
 * @param rhs Input rhs tensor
 * @param alpha1 The multiplier for lhs tensor
 * @param alpha2 The multiplier for rhs tensor
 * @param beta The multiplier for output tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopOr(topsopTensorHandle_t out,
                                      const topsopTensorHandle_t lhs,
                                      const topsopTensorHandle_t rhs,
                                      const topsopScalar_t alpha1,
                                      const topsopScalar_t alpha2,
                                      const topsopScalar_t beta,
                                      topsStream_t stream);
/**
 * @brief check whether current and operator support or not
 *
 * @param lhs Input lhs tensor
 * @param rhs Input rhs tensor
 * @return bool
 */
bool TOPSOP_EXPORT topsopOrIsSupported(const topsopTensorHandle_t lhs,
                                       const topsopTensorHandle_t rhs);
/**
 * @brief get output dims of current tensor and operator
 *
 * @param lhs Input lhs tensor
 * @param rhs Input rhs tensor
 * @param dims The dims of output tensor
 * @param rank The rank of output tensor
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopOrGetOutputDim(const topsopTensorHandle_t lhs,
                     const topsopTensorHandle_t rhs,
                     int64_t *dims,
                     int64_t *rank);

/**
 * @brief Computes elementwise equality
 *
 * @param out The output tensor
 * @param lhs The lhs tensor to compare
 * @param rhs The rhs tensor to compare
 * @param alpha1 The multiplier for lhs tensor
 * @param alpha2 The multiplier for rhs tensor
 * @param beta The multiplier for output tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopEq(topsopTensorHandle_t out,
                                      const topsopTensorHandle_t lhs,
                                      const topsopTensorHandle_t rhs,
                                      const topsopScalar_t alpha1,
                                      const topsopScalar_t alpha2,
                                      const topsopScalar_t beta,
                                      topsStream_t stream);
/**
 * @brief check whether current equal operator support or not
 *
 * @param lhs Input lhs tensor
 * @param rhs Input rhs tensor
 * @return bool
 */
bool TOPSOP_EXPORT topsopEqIsSupported(const topsopTensorHandle_t lhs,
                                       const topsopTensorHandle_t rhs);
/**
 * @brief get output dims and rank
 *
 * @param lhs Input lhs tensor
 * @param rhs Input rhs tensor
 * @param dims The dims of output tensor
 * @param rank The rank of output tensor
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopEqGetOutputDim(const topsopTensorHandle_t lhs,
                     const topsopTensorHandle_t rhs,
                     int64_t *dims,
                     int64_t *rank);

/**
 * @brief Computes elementwise greater
 *
 * @param out The output tensor
 * @param lhs The lhs tensor to compare
 * @param rhs The rhs tensor to compare
 * @param alpha1 The multiplier for lhs tensor
 * @param alpha2 The multiplier for rhs tensor
 * @param beta The multiplier for output tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopGt(topsopTensorHandle_t out,
                                      const topsopTensorHandle_t lhs,
                                      const topsopTensorHandle_t rhs,
                                      const topsopScalar_t alpha1,
                                      const topsopScalar_t alpha2,
                                      const topsopScalar_t beta,
                                      topsStream_t stream);
/**
 * @brief check whether current greater operator support or not
 *
 * @param lhs Input lhs tensor
 * @param rhs Input rhs tensor
 * @return bool
 */
bool TOPSOP_EXPORT topsopGtIsSupported(const topsopTensorHandle_t lhs,
                                       const topsopTensorHandle_t rhs);
/**
 * @brief get output dims and rank
 *
 * @param lhs Input lhs tensor
 * @param rhs Input rhs tensor
 * @param dims The dims of output tensor
 * @param rank The rank of output tensor
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopGtGetOutputDim(const topsopTensorHandle_t lhs,
                     const topsopTensorHandle_t rhs,
                     int64_t *dims,
                     int64_t *rank);

/**
 * @brief Computes elementwise greater_equal
 *
 * @param out The output tensor
 * @param lhs The lhs tensor to compare
 * @param rhs The rhs tensor to compare
 * @param alpha1 The multiplier for lhs tensor
 * @param alpha2 The multiplier for rhs tensor
 * @param beta The multiplier for output tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopGe(topsopTensorHandle_t out,
                                      const topsopTensorHandle_t lhs,
                                      const topsopTensorHandle_t rhs,
                                      const topsopScalar_t alpha1,
                                      const topsopScalar_t alpha2,
                                      const topsopScalar_t beta,
                                      topsStream_t stream);
/**
 * @brief check whether current greater_equal operator support or not
 *
 * @param lhs Input lhs tensor
 * @param rhs Input rhs tensor
 * @return bool
 */
bool TOPSOP_EXPORT topsopGeIsSupported(const topsopTensorHandle_t lhs,
                                       const topsopTensorHandle_t rhs);
/**
 * @brief get output dims and rank
 *
 * @param lhs Input lhs tensor
 * @param rhs Input rhs tensor
 * @param dims The dims of output tensor
 * @param rank The rank of output tensor
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopGeGetOutputDim(const topsopTensorHandle_t lhs,
                     const topsopTensorHandle_t rhs,
                     int64_t *dims,
                     int64_t *rank);

/**
 * @brief Computes elementwise less
 *
 * @param out The output tensor
 * @param lhs The lhs tensor to compare
 * @param rhs The rhs tensor to compare
 * @param alpha1 The multiplier for lhs tensor
 * @param alpha2 The multiplier for rhs tensor
 * @param beta The multiplier for output tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopLt(topsopTensorHandle_t out,
                                      const topsopTensorHandle_t lhs,
                                      const topsopTensorHandle_t rhs,
                                      const topsopScalar_t alpha1,
                                      const topsopScalar_t alpha2,
                                      const topsopScalar_t beta,
                                      topsStream_t stream);
/**
 * @brief check whether current less operator support or not
 *
 * @param lhs Input lhs tensor
 * @param rhs Input rhs tensor
 * @return bool
 */
bool TOPSOP_EXPORT topsopLtIsSupported(const topsopTensorHandle_t lhs,
                                       const topsopTensorHandle_t rhs);
/**
 * @brief get output dims and rank
 *
 * @param lhs Input lhs tensor
 * @param rhs Input rhs tensor
 * @param dims The dims of output tensor
 * @param rank The rank of output tensor
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopLtGetOutputDim(const topsopTensorHandle_t lhs,
                     const topsopTensorHandle_t rhs,
                     int64_t *dims,
                     int64_t *rank);

/**
 * @brief Computes elementwise less_equal
 *
 * @param out The output tensor
 * @param lhs The lhs tensor to compare
 * @param rhs The rhs tensor to compare
 * @param alpha1 The multiplier for lhs tensor
 * @param alpha2 The multiplier for rhs tensor
 * @param beta The multiplier for output tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopLe(topsopTensorHandle_t out,
                                      const topsopTensorHandle_t lhs,
                                      const topsopTensorHandle_t rhs,
                                      const topsopScalar_t alpha1,
                                      const topsopScalar_t alpha2,
                                      const topsopScalar_t beta,
                                      topsStream_t stream);
/**
 * @brief check whether current less_equal operator support or not
 *
 * @param lhs Input lhs tensor
 * @param rhs Input rhs tensor
 * @return bool
 */
bool TOPSOP_EXPORT topsopLeIsSupported(const topsopTensorHandle_t lhs,
                                       const topsopTensorHandle_t rhs);
/**
 * @brief get output dims and rank
 *
 * @param lhs Input lhs tensor
 * @param rhs Input rhs tensor
 * @param dims The dims of output tensor
 * @param rank The rank of output tensor
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopLeGetOutputDim(const topsopTensorHandle_t lhs,
                     const topsopTensorHandle_t rhs,
                     int64_t *dims,
                     int64_t *rank);

/**
 * @brief Computes elementwise not_equal
 *
 * @param out The output tensor
 * @param lhs The lhs tensor to compare
 * @param rhs The rhs tensor to compare
 * @param alpha1 The multiplier for lhs tensor
 * @param alpha2 The multiplier for rhs tensor
 * @param beta The multiplier for output tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopNe(topsopTensorHandle_t out,
                                      const topsopTensorHandle_t lhs,
                                      const topsopTensorHandle_t rhs,
                                      const topsopScalar_t alpha1,
                                      const topsopScalar_t alpha2,
                                      const topsopScalar_t beta,
                                      topsStream_t stream);
/**
 * @brief check whether current not_equal operator support or not
 *
 * @param lhs Input lhs tensor
 * @param rhs Input rhs tensor
 * @return bool
 */
bool TOPSOP_EXPORT topsopNeIsSupported(const topsopTensorHandle_t lhs,
                                       const topsopTensorHandle_t rhs);
/**
 * @brief get output dims and rank
 *
 * @param lhs Input lhs tensor
 * @param rhs Input rhs tensor
 * @param dims The dims of output tensor
 * @param rank The rank of output tensor
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopNeGetOutputDim(const topsopTensorHandle_t lhs,
                     const topsopTensorHandle_t rhs,
                     int64_t *dims,
                     int64_t *rank);

/**
 * @brief Computes GeluGrad
 *
 * @param out The output tensor
 * @param x The input tensor x
 * @param dy The intput tensor dy
 * @param alpha1 The multiplier for x tensor
 * @param alpha2 The multiplier for dy tensor
 * @param beta The multiplier for output tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopGeluGrad(topsopTensorHandle_t out,
                                            const topsopTensorHandle_t x,
                                            const topsopTensorHandle_t dy,
                                            const topsopScalar_t alpha1,
                                            const topsopScalar_t alpha2,
                                            const topsopScalar_t beta,
                                            topsStream_t stream);
/**
 * @brief check whether current GeluGrad operator support or not
 *
 * @param x The input tensor x
 * @param dy The intput tensor dy
 * @return bool
 */
bool TOPSOP_EXPORT topsopGeluGradIsSupported(const topsopTensorHandle_t x,
                                             const topsopTensorHandle_t dy);
/**
 * @brief get output dims and rank
 *
 * @param x The input tensor x
 * @param dy The intput tensor dy
 * @param dims The dims of output tensor
 * @param rank The rank of output tensor
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopGeluGradGetOutputDim(const topsopTensorHandle_t x,
                           const topsopTensorHandle_t dy,
                           int64_t *dims,
                           int64_t *rank);

/**
 * @brief abs operator
 *
 * @param out The ouput tensor
 * @param in The input tensor
 * @param alpha The multiplier for input tensor
 * @param beta The multiplier for output tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopAbs(topsopTensorHandle_t out,
                                       const topsopTensorHandle_t in,
                                       const topsopScalar_t alpha,
                                       const topsopScalar_t beta,
                                       topsStream_t stream);
/**
 * @brief check whether current abs operator support or not
 *
 * @param in The input tensor
 * @return bool
 */
bool TOPSOP_EXPORT topsopAbsIsSupported(const topsopTensorHandle_t in);
/**
 * @brief get output dims of current tensor abs operator
 *
 * @param in The input tensor
 * @param dims The dims of output tensor
 * @param rank The rank of output tensor
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopAbsGetOutputDim(
    const topsopTensorHandle_t in, int64_t *dims, int64_t *rank);

/**
 * @brief ceil operator
 *
 * @param out The ouput tensor
 * @param in The input tensor
 * @param alpha The multiplier for input tensor
 * @param beta The multiplier for output tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopCeil(topsopTensorHandle_t out,
                                        const topsopTensorHandle_t in,
                                        const topsopScalar_t alpha,
                                        const topsopScalar_t beta,
                                        topsStream_t stream);
/**
 * @brief check whether current ceil operator support or not
 *
 * @param in The input tensor
 * @return bool
 */
bool TOPSOP_EXPORT topsopCeilIsSupported(const topsopTensorHandle_t in);
/**
 * @brief get output dims of current tensor ceil operator
 *
 * @param in The input tensor
 * @param dims The dims of output tensor
 * @param rank The rank of output tensor
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopCeilGetOutputDim(
    const topsopTensorHandle_t in, int64_t *dims, int64_t *rank);

/**
 * @brief exp operator
 *
 * @param out The ouput tensor
 * @param in The input tensor
 * @param alpha The multiplier for input tensor
 * @param beta The multiplier for output tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopExp(topsopTensorHandle_t out,
                                       const topsopTensorHandle_t in,
                                       const topsopScalar_t alpha,
                                       const topsopScalar_t beta,
                                       topsStream_t stream);
/**
 * @brief check whether current exp operator support or not
 *
 * @param in The input tensor
 * @return bool
 */
bool TOPSOP_EXPORT topsopExpIsSupported(const topsopTensorHandle_t in);
/**
 * @brief get output dims of current tensor exp operator
 *
 * @param in The input tensor
 * @param dims The dims of output tensor
 * @param rank The rank of output tensor
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopExpGetOutputDim(
    const topsopTensorHandle_t in, int64_t *dims, int64_t *rank);

/**
 * @brief floor operator
 *
 * @param out The ouput tensor
 * @param in The input tensor
 * @param alpha The multiplier for input tensor
 * @param beta The multiplier for output tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopFloor(topsopTensorHandle_t out,
                                         const topsopTensorHandle_t in,
                                         const topsopScalar_t alpha,
                                         const topsopScalar_t beta,
                                         topsStream_t stream);
/**
 * @brief check whether current floor operator support or not
 *
 * @param in The input tensor
 * @return bool
 */
bool TOPSOP_EXPORT topsopFloorIsSupported(const topsopTensorHandle_t in);
/**
 * @brief get output dims of current tensor floor operator
 *
 * @param in The input tensor
 * @param dims The dims of output tensor
 * @param rank The rank of output tensor
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopFloorGetOutputDim(
    const topsopTensorHandle_t in, int64_t *dims, int64_t *rank);

/**
 * @brief log operator
 *
 * @param out The ouput tensor
 * @param in The input tensor
 * @param alpha The multiplier for input tensor
 * @param beta The multiplier for output tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopLog(topsopTensorHandle_t out,
                                       const topsopTensorHandle_t in,
                                       const topsopScalar_t alpha,
                                       const topsopScalar_t beta,
                                       topsStream_t stream);
/**
 * @brief check whether current log operator support or not
 *
 * @param in The input tensor
 * @return bool
 */
bool TOPSOP_EXPORT topsopLogIsSupported(const topsopTensorHandle_t in);
/**
 * @brief get output dims of current tensor log operator
 *
 * @param in The input tensor
 * @param dims The dims of output tensor
 * @param rank The rank of output tensor
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopLogGetOutputDim(
    const topsopTensorHandle_t in, int64_t *dims, int64_t *rank);

/**
 * @brief neg operator
 *
 * @param out The ouput tensor
 * @param in The input tensor
 * @param alpha The multiplier for input tensor
 * @param beta The multiplier for output tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopNeg(topsopTensorHandle_t out,
                                       const topsopTensorHandle_t in,
                                       const topsopScalar_t alpha,
                                       const topsopScalar_t beta,
                                       topsStream_t stream);
/**
 * @brief check whether current neg operator support or not
 *
 * @param in The input tensor
 * @return bool
 */
bool TOPSOP_EXPORT topsopNegIsSupported(const topsopTensorHandle_t in);
/**
 * @brief get output dims of current tensor neg operator
 *
 * @param in The input tensor
 * @param dims The dims of output tensor
 * @param rank The rank of output tensor
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopNegGetOutputDim(
    const topsopTensorHandle_t in, int64_t *dims, int64_t *rank);

/**
 * @brief rsqrt operator
 *
 * @param out The ouput tensor
 * @param in The input tensor
 * @param alpha The multiplier for input tensor
 * @param beta The multiplier for output tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopRsqrt(topsopTensorHandle_t out,
                                         const topsopTensorHandle_t in,
                                         const topsopScalar_t alpha,
                                         const topsopScalar_t beta,
                                         topsStream_t stream);
/**
 * @brief check whether current rsqrt operator support or not
 *
 * @param in The input tensor
 * @return bool
 */
bool TOPSOP_EXPORT topsopRsqrtIsSupported(const topsopTensorHandle_t in);
/**
 * @brief get output dims of current tensor rsqrt operator
 *
 * @param out The ouput tensor
 * @param in The input tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopRsqrtGetOutputDim(
    const topsopTensorHandle_t in, int64_t *dims, int64_t *rank);

/**
 * @brief sign operator
 *
 * @param out The ouput tensor
 * @param in The input tensor
 * @param alpha The multiplier for input tensor
 * @param beta The multiplier for output tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopSign(topsopTensorHandle_t out,
                                        const topsopTensorHandle_t in,
                                        const topsopScalar_t alpha,
                                        const topsopScalar_t beta,
                                        topsStream_t stream);
/**
 * @brief check whether current sign operator support or not
 *
 * @param in The input tensor
 * @return bool
 */
bool TOPSOP_EXPORT topsopSignIsSupported(const topsopTensorHandle_t in);
/**
 * @brief get output dims of current tensor sign operator
 *
 * @param out The ouput tensor
 * @param in The input tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopSignGetOutputDim(
    const topsopTensorHandle_t in, int64_t *dims, int64_t *rank);

/**
 * @brief sqrt operator
 *
 * @param out The ouput tensor
 * @param in The input tensor
 * @param alpha The multiplier for input tensor
 * @param beta The multiplier for output tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopSqrt(topsopTensorHandle_t out,
                                        const topsopTensorHandle_t in,
                                        const topsopScalar_t alpha,
                                        const topsopScalar_t beta,
                                        topsStream_t stream);
/**
 * @brief check whether current sqrt operator support or not
 *
 * @param in The input tensor
 * @return bool
 */
bool TOPSOP_EXPORT topsopSqrtIsSupported(const topsopTensorHandle_t in);
/**
 * @brief get output dims of current tensor sqrt operator
 *
 * @param out The ouput tensor
 * @param in The input tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopSqrtGetOutputDim(
    const topsopTensorHandle_t in, int64_t *dims, int64_t *rank);

/**
 * @brief tanh operator
 *
 * @param out The ouput tensor
 * @param in The input tensor
 * @param alpha The multiplier for input tensor
 * @param beta The multiplier for output tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */

topsopStatus_t TOPSOP_EXPORT topsopTanh(topsopTensorHandle_t out,
                                        const topsopTensorHandle_t in,
                                        const topsopScalar_t alpha,
                                        const topsopScalar_t beta,
                                        topsStream_t stream);
/**
 * @brief check whether current tanh operator support or not
 *
 * @param in The input tensor
 * @return bool
 */
bool TOPSOP_EXPORT topsopTanhIsSupported(const topsopTensorHandle_t in);
/**
 * @brief get output dims of current tensor tanh operator
 *
 * @param out The ouput tensor
 * @param in The input tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopTanhGetOutputDim(
    const topsopTensorHandle_t in, int64_t *dims, int64_t *rank);

/**
 * @brief atan operator
 *
 * @param out The ouput tensor
 * @param in The input tensor
 * @param alpha The multiplier for input tensor
 * @param beta The multiplier for output tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopAtan(topsopTensorHandle_t out,
                                        const topsopTensorHandle_t in,
                                        const topsopScalar_t alpha,
                                        const topsopScalar_t beta,
                                        topsStream_t stream);
/**
 * @brief check whether current atan operator support or not
 *
 * @param in The input tensor
 * @return bool
 */
bool TOPSOP_EXPORT topsopAtanIsSupported(const topsopTensorHandle_t in);
/**
 * @brief get output dims of current tensor atan operator
 *
 * @param out The ouput tensor
 * @param in The input tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopAtanGetOutputDim(
    const topsopTensorHandle_t in, int64_t *dims, int64_t *rank);

/**
 * @brief bitwise_not operator
 *
 * @param out The ouput tensor
 * @param in The input tensor
 * @param alpha The multiplier for input tensor
 * @param beta The multiplier for output tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopNot(topsopTensorHandle_t out,
                                       const topsopTensorHandle_t in,
                                       const topsopScalar_t alpha,
                                       const topsopScalar_t beta,
                                       topsStream_t stream);
/**
 * @brief check whether current bitwise_not operator support or not
 *
 * @param in The input tensor
 * @return bool
 */
bool TOPSOP_EXPORT topsopNotIsSupported(const topsopTensorHandle_t in);
/**
 * @brief get output dims of current tensor bitwise_not operator
 *
 * @param out The ouput tensor
 * @param in The input tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopNotGetOutputDim(
    const topsopTensorHandle_t in, int64_t *dims, int64_t *rank);

/**
 * @brief gelu operator
 *
 * @param out The ouput tensor
 * @param in The input tensor
 * @param alpha The multiplier for input tensor
 * @param beta The multiplier for output tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopGelu(topsopTensorHandle_t out,
                                        const topsopTensorHandle_t in,
                                        const topsopScalar_t alpha,
                                        const topsopScalar_t beta,
                                        topsStream_t stream);
/**
 * @brief check whether current gelu operator support or not
 *
 * @param in The input tensor
 * @return bool
 */
bool TOPSOP_EXPORT topsopGeluIsSupported(const topsopTensorHandle_t in);
/**
 * @brief get output dims of current tensor gelu operator
 *
 * @param out The ouput tensor
 * @param in The input tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopGeluGetOutputDim(
    const topsopTensorHandle_t in, int64_t *dims, int64_t *rank);

/**
 * @brief acos operator
 *
 * @param out The ouput tensor
 * @param in The input tensor
 * @param alpha The multiplier for input tensor
 * @param beta The multiplier for output tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopAcos(topsopTensorHandle_t out,
                                        const topsopTensorHandle_t in,
                                        const topsopScalar_t alpha,
                                        const topsopScalar_t beta,
                                        topsStream_t stream);
/**
 * @brief check whether current acos operator support or not
 *
 * @param in The input tensor
 * @return bool
 */
bool TOPSOP_EXPORT topsopAcosIsSupported(const topsopTensorHandle_t in);
/**
 * @brief get output dims of current tensor acos operator
 *
 * @param out The ouput tensor
 * @param in The input tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopAcosGetOutputDim(
    const topsopTensorHandle_t in, int64_t *dims, int64_t *rank);

/**
 * @brief acosh operator
 *
 * @param out The ouput tensor
 * @param in The input tensor
 * @param alpha The multiplier for input tensor
 * @param beta The multiplier for output tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopAcosh(topsopTensorHandle_t out,
                                         const topsopTensorHandle_t in,
                                         const topsopScalar_t alpha,
                                         const topsopScalar_t beta,
                                         topsStream_t stream);
/**
 * @brief check whether current acosh operator support or not
 *
 * @param in The input tensor
 * @return bool
 */
bool TOPSOP_EXPORT topsopAcoshIsSupported(const topsopTensorHandle_t in);
/**
 * @brief get output dims of current tensor Acosh operator
 *
 * @param out The ouput tensor
 * @param in The input tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopAcoshGetOutputDim(
    const topsopTensorHandle_t in, int64_t *dims, int64_t *rank);

/**
 * @brief asin operator
 *
 * @param out The ouput tensor
 * @param in The input tensor
 * @param alpha The multiplier for input tensor
 * @param beta The multiplier for output tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopAsin(topsopTensorHandle_t out,
                                        const topsopTensorHandle_t in,
                                        const topsopScalar_t alpha,
                                        const topsopScalar_t beta,
                                        topsStream_t stream);
/**
 * @brief check whether current asin operator support or not
 *
 * @param in The input tensor
 * @return bool
 */
bool TOPSOP_EXPORT topsopAsinIsSupported(const topsopTensorHandle_t in);
/**
 * @brief get output dims of current tensor asin operator
 *
 * @param out The ouput tensor
 * @param in The input tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopAsinGetOutputDim(
    const topsopTensorHandle_t in, int64_t *dims, int64_t *rank);

/**
 * @brief asinh operator
 *
 * @param out The ouput tensor
 * @param in The input tensor
 * @param alpha The multiplier for input tensor
 * @param beta The multiplier for output tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopAsinh(topsopTensorHandle_t out,
                                         const topsopTensorHandle_t in,
                                         const topsopScalar_t alpha,
                                         const topsopScalar_t beta,
                                         topsStream_t stream);
/**
 * @brief check whether current asinh operator support or not
 *
 * @param in The input tensor
 * @return bool
 */
bool TOPSOP_EXPORT topsopAsinhIsSupported(const topsopTensorHandle_t in);
/**
 * @brief get output dims of current tensor asinh operator
 *
 * @param out The ouput tensor
 * @param in The input tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopAsinhGetOutputDim(
    const topsopTensorHandle_t in, int64_t *dims, int64_t *rank);

/**
 * @brief atanh operator
 *
 * @param out The ouput tensor
 * @param in The input tensor
 * @param alpha The multiplier for input tensor
 * @param beta The multiplier for output tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopAtanh(topsopTensorHandle_t out,
                                         const topsopTensorHandle_t in,
                                         const topsopScalar_t alpha,
                                         const topsopScalar_t beta,
                                         topsStream_t stream);
/**
 * @brief check whether current atanh operator support or not
 *
 * @param in The input tensor
 * @return bool
 */
bool TOPSOP_EXPORT topsopAtanhIsSupported(const topsopTensorHandle_t in);
/**
 * @brief get output dims of current tensor atanh operator
 *
 * @param out The ouput tensor
 * @param in The input tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopAtanhGetOutputDim(
    const topsopTensorHandle_t in, int64_t *dims, int64_t *rank);

/**
 * @brief cos operator
 *
 * @param out The ouput tensor
 * @param in The input tensor
 * @param alpha The multiplier for input tensor
 * @param beta The multiplier for output tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopCos(topsopTensorHandle_t out,
                                       const topsopTensorHandle_t in,
                                       const topsopScalar_t alpha,
                                       const topsopScalar_t beta,
                                       topsStream_t stream);
/**
 * @brief check whether current cos operator support or not
 *
 * @param in The input tensor
 * @return bool
 */
bool TOPSOP_EXPORT topsopCosIsSupported(const topsopTensorHandle_t in);
/**
 * @brief get output dims of current tensor cos operator
 *
 * @param out The ouput tensor
 * @param in The input tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopCosGetOutputDim(
    const topsopTensorHandle_t in, int64_t *dims, int64_t *rank);

/**
 * @brief cosh operator
 *
 * @param out The ouput tensor
 * @param in The input tensor
 * @param alpha The multiplier for input tensor
 * @param beta The multiplier for output tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopCosh(topsopTensorHandle_t out,
                                        const topsopTensorHandle_t in,
                                        const topsopScalar_t alpha,
                                        const topsopScalar_t beta,
                                        topsStream_t stream);
/**
 * @brief check whether current cosh operator support or not
 *
 * @param in The input tensor
 * @return bool
 */
bool TOPSOP_EXPORT topsopCoshIsSupported(const topsopTensorHandle_t in);
/**
 * @brief get output dims of current tensor cosh operator
 *
 * @param out The ouput tensor
 * @param in The input tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopCoshGetOutputDim(
    const topsopTensorHandle_t in, int64_t *dims, int64_t *rank);

/**
 * @brief sin operator
 *
 * @param out The ouput tensor
 * @param in The input tensor
 * @param alpha The multiplier for input tensor
 * @param beta The multiplier for output tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopSin(topsopTensorHandle_t out,
                                       const topsopTensorHandle_t in,
                                       const topsopScalar_t alpha,
                                       const topsopScalar_t beta,
                                       topsStream_t stream);
/**
 * @brief check whether current sin operator support or not
 *
 * @param in The input tensor
 * @return bool
 */
bool TOPSOP_EXPORT topsopSinIsSupported(const topsopTensorHandle_t in);
/**
 * @brief get output dims of current tensor sin operator
 *
 * @param out The ouput tensor
 * @param in The input tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopSinGetOutputDim(
    const topsopTensorHandle_t in, int64_t *dims, int64_t *rank);

/**
 * @brief sinh operator
 *
 * @param out The ouput tensor
 * @param in The input tensor
 * @param alpha The multiplier for input tensor
 * @param beta The multiplier for output tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopSinh(topsopTensorHandle_t out,
                                        const topsopTensorHandle_t in,
                                        const topsopScalar_t alpha,
                                        const topsopScalar_t beta,
                                        topsStream_t stream);
/**
 * @brief check whether current sinh operator support or not
 *
 * @param in The input tensor
 * @return bool
 */
bool TOPSOP_EXPORT topsopSinhIsSupported(const topsopTensorHandle_t in);
/**
 * @brief get output dims of current tensor sinh operator
 *
 * @param out The ouput tensor
 * @param in The input tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopSinhGetOutputDim(
    const topsopTensorHandle_t in, int64_t *dims, int64_t *rank);

/**
 * @brief tan operator
 *
 * @param out The ouput tensor
 * @param in The input tensor
 * @param alpha The multiplier for input tensor
 * @param beta The multiplier for output tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopTan(topsopTensorHandle_t out,
                                       const topsopTensorHandle_t in,
                                       const topsopScalar_t alpha,
                                       const topsopScalar_t beta,
                                       topsStream_t stream);
/**
 * @brief check whether current tan operator support or not
 *
 * @param in The input tensor
 * @return bool
 */
bool TOPSOP_EXPORT topsopTanIsSupported(const topsopTensorHandle_t in);
/**
 * @brief get output dims of current tensor tan operator
 *
 * @param out The ouput tensor
 * @param in The input tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopTanGetOutputDim(
    const topsopTensorHandle_t in, int64_t *dims, int64_t *rank);

/**
 * @brief convert operator
 *
 * @param out The ouput tensor
 * @param in The input tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopConvert(topsopTensorHandle_t out,
                                           const topsopTensorHandle_t in,
                                           topsStream_t stream);
/**
 * @brief check whether current convert operator support or not
 *
 * @param out The ouput tensor
 * @param in The input tensor
 * @return bool
 */
bool TOPSOP_EXPORT topsopConvertIsSupported(const topsopTensorHandle_t out,
                                            const topsopTensorHandle_t in);
/**
 * @brief get output dims of current tensor convert operator
 *
 * @param out The ouput tensor
 * @param in The input tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopConvertGetOutputDim(
    const topsopTensorHandle_t in, int64_t *dims, int64_t *rank);

/**
 * @brief clamp operator
 *
 * @param out The ouput tensor
 * @param in The input tensor
 * @param min The lower-bound of the range to be clamped to
 * @param max The upper-bound of the range to be clamped to
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopClamp(topsopTensorHandle_t out,
                                         const topsopTensorHandle_t in,
                                         const topsopTensorHandle_t min,
                                         const topsopTensorHandle_t max,
                                         topsStream_t stream);
/**
 * @brief check whether current clamp operator support or not
 *
 * @param in The input tensor
 * @param min The lower-bound of the range to be clamped to
 * @param max The upper-bound of the range to be clamped to
 * @return bool
 */
bool TOPSOP_EXPORT topsopClampIsSupported(const topsopTensorHandle_t in,
                                          const topsopTensorHandle_t min,
                                          const topsopTensorHandle_t max);
/**
 * @brief get output dims of current tensor clamp operator
 *
 * @param out The ouput tensor
 * @param in The input tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopClampGetOutputDim(
    const topsopTensorHandle_t in, int64_t *dims, int64_t *rank);

/**
 * @brief clamp operator, the max value and min value are scalars
 *
 * @param out The ouput tensor
 * @param in The input tensor
 * @param min The lower-bound of the range to be clamped to
 * @param max The upper-bound of the range to be clamped to
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopClampScalar(topsopTensorHandle_t out,
                                               const topsopTensorHandle_t in,
                                               const topsopScalar_t min,
                                               const topsopScalar_t max,
                                               topsStream_t stream);
/**
 * @brief check whether current clamp operator support or not
 *
 * @param in The input tensor
 * @param min The lower-bound of the range to be clamped to
 * @param max The upper-bound of the range to be clamped to
 * @return bool
 */
bool TOPSOP_EXPORT topsopClampScalarIsSupported(const topsopTensorHandle_t in,
                                                const topsopScalar_t min,
                                                const topsopScalar_t max);
/**
 * @brief get output dims of current tensor clamp operator
 *
 * @param out The ouput tensor
 * @param in The input tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopClampScalarGetOutputDim(
    const topsopTensorHandle_t in, int64_t *dims, int64_t *rank);

/**
 * @brief This function performs the forward LRN layer computation.
 *
 * @param output Result tensor
 * @param input Input tensor
 * @param lrnMode LRN layer mode of operation. Currently only
 *                TOPSOP_LRN_CROSS_CHANNEL_DIM1 is implemented
 * @param lrnN Normalization window width in elements
 * @param lrnAlpha Value of the alpha variance scaling parameter in
 *                 the normalization formula
 * @param lrnBeta Value of the beta power parameter in the
 *                normalization formula
 * @param lrnK Value of the k parameter in the normalization formula
 * @param alpha scaling factors (in host memory) used to blend the
 *              layer output value with prior value in the destination tensor
 * @param beta scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopLRNCrossChannelForward(topsopTensorHandle_t output,
                             const topsopTensorHandle_t input,
                             topsopLRNMode_t lrnMode,
                             int64_t lrnN,
                             float lrnAlpha,
                             float lrnBeta,
                             float lrnK,
                             topsopScalar_t alpha,
                             topsopScalar_t beta,
                             topsStream_t stream);

/**
 * @brief check whether current lrn forward operator support or not
 *
 * @param input Input tensor
 * @param lrnMode LRN layer mode of operation. Currently only
 *                TOPSOP_LRN_CROSS_CHANNEL_DIM1 is implemented
 * @param lrnN Normalization window width in elements
 * @param lrnAlpha Value of the alpha variance scaling parameter in
 *                 the normalization formula
 * @param lrnBeta Value of the beta power parameter in the
 *                normalization formula
 * @param lrnK Value of the k parameter in the normalization formula
 * @param alpha scaling factors (in host memory) used to blend the
 *              layer output value with prior value in the destination tensor
 * @param beta scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor
 * @return bool
 */
bool TOPSOP_EXPORT
topsopLRNCrossChannelForwardIsSupported(const topsopTensorHandle_t input,
                                        topsopLRNMode_t lrnMode,
                                        int64_t lrnN,
                                        float lrnAlpha,
                                        float lrnBeta,
                                        float lrnK,
                                        topsopScalar_t alpha,
                                        topsopScalar_t beta);

/**
 * @brief get output dims of current tensor lrn forward operator
 *
 * @param input Input tensor
 * @param dims The dims of output tensor
 * @param rank The rank of output tensor
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopLRNCrossChannelForwardGetOutputDim(
    const topsopTensorHandle_t input, int64_t *dims, int64_t *rank);

/**
 * @brief This function performs the backward LRN layer computation.
 *
 * @param grad_input Output differential tensor
 * @param input Input tensor
 * @param output Output tensor
 * @param grad_output Input differential tensor
 * @param lrnMode LRN layer mode of operation. Currently only
 *                TOPSOP_LRN_CROSS_CHANNEL_DIM1 is implemented
 * @param lrnN Normalization window width in elements
 * @param lrnAlpha Value of the alpha variance scaling parameter in
 *                 the normalization formula
 * @param lrnBeta Value of the beta power parameter in the
 *                normalization formula
 * @param lrnK Value of the k parameter in the normalization formula
 * @param alpha scaling factors (in host memory) used to blend the
 *              layer output value with prior value in the destination tensor
 * @param beta scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopLRNCrossChannelBackward(topsopTensorHandle_t grad_input,
                              const topsopTensorHandle_t input,
                              const topsopTensorHandle_t output,
                              const topsopTensorHandle_t grad_output,
                              topsopLRNMode_t lrnMode,
                              int64_t lrnN,
                              float lrnAlpha,
                              float lrnBeta,
                              float lrnK,
                              topsopScalar_t alpha,
                              topsopScalar_t beta,
                              topsStream_t stream);

/**
 * @brief check whether current lrn backward operator support or not
 *
 * @param input Input tensor
 * @param output Output tensor
 * @param grad_output Input differential tensor
 * @param lrnMode LRN layer mode of operation. Currently only
 *                TOPSOP_LRN_CROSS_CHANNEL_DIM1 is implemented
 * @param lrnN Normalization window width in elements
 * @param lrnAlpha Value of the alpha variance scaling parameter in
 *                 the normalization formula
 * @param lrnBeta Value of the beta power parameter in the
 *                normalization formula
 * @param lrnK Value of the k parameter in the normalization formula
 * @param alpha scaling factors (in host memory) used to blend the
 *              layer output value with prior value in the destination tensor
 * @param beta scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor
 * @return bool
 */
bool TOPSOP_EXPORT
topsopLRNCrossChannelBackwardIsSupported(const topsopTensorHandle_t input,
                                         const topsopTensorHandle_t output,
                                         const topsopTensorHandle_t grad_output,
                                         topsopLRNMode_t lrnMode,
                                         int64_t lrnN,
                                         float lrnAlpha,
                                         float lrnBeta,
                                         float lrnK,
                                         topsopScalar_t alpha,
                                         topsopScalar_t beta);

/**
 * @brief get output dims of current tensor lrn backward operator
 *
 * @param input Input tensor
 * @param output Output tensor
 * @param grad_output Input differential tensor
 * @param dims The dims of output tensor
 * @param rank The rank of output tensor
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopLRNCrossChannelBackwardGetOutputDim(
    const topsopTensorHandle_t input,
    const topsopTensorHandle_t output,
    const topsopTensorHandle_t grad_output,
    int64_t *dims,
    int64_t *rank);

/**
 * @brief This function performs the fast Softmax forward computation.
 *
 * @param output the output tensor
 * @param input the input tensor
 * @param axis a dimension along which Softmax will be computed
 * @param alpha scaling factor used to blend the computation result
 * @param beta scaling factor
 * @param stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopFastSoftmaxForward(topsopTensorHandle_t output,
                         const topsopTensorHandle_t input,
                         int32_t axis,
                         topsopScalar_t alpha,
                         topsopScalar_t beta,
                         topsStream_t stream);

/**
 * @brief check whether the current fast Softmax forward operation is
 * supported or not
 *
 * @param input input tensor
 * @param axis a dimension along which Softmax will be computed
 *
 * @return bool
 */
bool TOPSOP_EXPORT topsopFastSoftmaxForwardIsSupported(
    const topsopTensorHandle_t input, int32_t axis);

/**
 * @brief get output dims of the current fast Softmax forward operator
 *
 * @param input input tensor
 * @param axis a dimension along which Softmax will be computed
 * @param dims the dimensions of output tensor
 * @param rank the rank of output tensor
 *
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopFastSoftmaxForwardGetOutputDim(const topsopTensorHandle_t input,
                                     int32_t axis,
                                     int64_t *dims,
                                     int64_t *rank);

/**
 * @brief This function performs the Softmax forward computation.
 *
 * @param output the output tensor
 * @param input the input tensor
 * @param axis a dimension along which Softmax will be computed
 * @param alpha scaling factor used to blend the computation result
 * @param beta scaling factor
 * @param stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopSoftmaxForward(topsopTensorHandle_t output,
                     const topsopTensorHandle_t input,
                     int32_t axis,
                     topsopScalar_t alpha,
                     topsopScalar_t beta,
                     topsStream_t stream);

/**
 * @brief check whether the current Softmax forward operation is
 * supported or not
 *
 * @param input input tensor
 * @param axis a dimension along which Softmax will be computed
 *
 * @return bool
 */
bool TOPSOP_EXPORT
topsopSoftmaxForwardIsSupported(const topsopTensorHandle_t input, int32_t axis);

/**
 * @brief get output dims of the current Softmax forward operator
 *
 * @param input input tensor
 * @param axis a dimension along which Softmax will be computed
 * @param dims the dimensions of output tensor
 * @param rank the rank of output tensor
 *
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopSoftmaxForwardGetOutputDim(const topsopTensorHandle_t input,
                                 int32_t axis,
                                 int64_t *dims,
                                 int64_t *rank);

/**
 * @brief This function performs the forward LogSoftmax computation.
 *
 * @param output the output tensor
 * @param input the input tensor
 * @param axis a dimension along which LogSoftmax will be computed
 * @param alpha scaling factor used to blend the computation result
 * @param beta scaling factor
 * @param stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopLogSoftmaxForward(topsopTensorHandle_t output,
                        const topsopTensorHandle_t input,
                        int32_t axis,
                        topsopScalar_t alpha,
                        topsopScalar_t beta,
                        topsStream_t stream);

/**
 * @brief check whether the current LogSoftmax forward operation is
 * supported or not
 *
 * @param input input tensor
 * @param axis a dimension along which LogSoftmax will be computed
 *
 * @return bool
 */
bool TOPSOP_EXPORT topsopLogSoftmaxForwardIsSupported(
    const topsopTensorHandle_t input, int32_t axis);

/**
 * @brief get output dims of the current LogSoftmax forward operator
 *
 * @param input input tensor
 * @param axis a dimension along which LogSoftmax will be computed
 * @param dims the dimensions of output tensor
 * @param rank the rank of output tensor
 *
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopLogSoftmaxForwardGetOutputDim(const topsopTensorHandle_t input,
                                    int32_t axis,
                                    int64_t *dims,
                                    int64_t *rank);

/**
 * @brief This function performs the backward Softmax computation.
 *
 * @param output the output tensor
 * @param input_y the input tensor
 * @param input_dy the input tensor
 * @param axis a dimension along which Softmax will be computed
 * @param alpha scaling factor used to blend the computation result
 * @param beta scaling factor
 * @param stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopSoftmaxBackward(topsopTensorHandle_t output,
                      const topsopTensorHandle_t input_y,
                      const topsopTensorHandle_t input_dy,
                      int32_t axis,
                      topsopScalar_t alpha,
                      topsopScalar_t beta,
                      topsStream_t stream);

/**
 * @brief check whether the current LogSoftmax backward operation is
 * supported or not
 *
 * @param input_y input tensor
 * @param input_dy input tensor
 * @param axis a dimension along which Softmax will be computed
 *
 * @return bool
 */
bool TOPSOP_EXPORT
topsopSoftmaxBackwardIsSupported(const topsopTensorHandle_t input_y,
                                 const topsopTensorHandle_t input_dy,
                                 int32_t axis);

/**
 * @brief get output dims of the current Softmax forward operator
 *
 * @param input_y input tensor
 * @param input_dy input tensor
 * @param axis a dimension along which Softmax will be computed
 * @param dims the dimensions of output tensor
 * @param rank the rank of output tensor
 *
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopSoftmaxBackwardGetOutputDim(const topsopTensorHandle_t input,
                                  const topsopTensorHandle_t input_dy,
                                  int32_t axis,
                                  int64_t *dims,
                                  int64_t *rank);

/**
 * @brief This function performs the backward LogSoftmax computation.
 *
 * @param output the output tensor
 * @param input_y the input tensor
 * @param input_dy the input tensor
 * @param axis a dimension along which LogSoftmax will be computed
 * @param alpha scaling factor used to blend the computation result
 * @param beta scaling factor
 * @param stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopLogSoftmaxBackward(topsopTensorHandle_t output,
                         const topsopTensorHandle_t input_y,
                         const topsopTensorHandle_t input_dy,
                         int32_t axis,
                         topsopScalar_t alpha,
                         topsopScalar_t beta,
                         topsStream_t stream);

/**
 * @brief check whether the current LogSoftmax backward operation is
 * supported or not
 *
 * @param input_y input tensor
 * @param input_dy input tensor
 * @param axis a dimension along which LogSoftmax will be computed
 *
 * @return bool
 */
bool TOPSOP_EXPORT
topsopLogSoftmaxBackwardIsSupported(const topsopTensorHandle_t input_y,
                                    const topsopTensorHandle_t input_dy,
                                    int32_t axis);

/**
 * @brief get output dims of the current LogSoftmax forward operator
 *
 * @param input_y input tensor
 * @param input_dy input tensor
 * @param axis a dimension along which LogSoftmax will be computed
 * @param dims the dimensions of output tensor
 * @param rank the rank of output tensor
 *
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopLogSoftmaxBackwardGetOutputDim(const topsopTensorHandle_t input,
                                     const topsopTensorHandle_t input_dy,
                                     int32_t axis,
                                     int64_t *dims,
                                     int64_t *rank);

/**
 * @brief This function performs a addition computation.
 *
 * @param output the output tensor
 * @param input the input tensor
 * @param reduceNanOpt an enumerated type used to indicate if a given routine
 * should propagate Nan numbers
 * @param alpha scaling factor used to blend the computation result
 * @param beta scaling factor
 * @param stream tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopReduceSum(topsopTensorHandle_t output,
                const topsopTensorHandle_t input,
                const topsopSize_t *dimensions,
                bool keepdims,
                topsopNanPropagation_t reduceNanOpt,
                topsopScalar_t alpha,
                topsopScalar_t beta,
                topsStream_t stream);

/**
 * @brief This function checks whether a addition is supported or not
 * with currest input tensor.
 *
 * @param input the input tensor
 * @param dimensions the dimensions to reduce
 * @return bool
 */
bool TOPSOP_EXPORT topsopReduceSumIsSupported(const topsopTensorHandle_t input,
                                              const topsopSize_t *dimensions);

/**
 * @brief This function gets the dimensions and rank of output tensor.
 *
 * @param input the input tensor
 * @param dimensions the dimensions to reduce
 * @param keepdims whether the output tensor has dim retained or not
 * @param dims the dimensions of output tensor
 * @param rank the rank of output tensor
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopReduceSumGetOutputDim(const topsopTensorHandle_t input,
                            const topsopSize_t *dimensions,
                            bool keepdims,
                            int64_t *dims,
                            int64_t *rank);

/**
 * @brief This function performs a multiplication computation.
 *
 * @param output the output tensor
 * @param input the input tensor
 * @param reduceNanOpt an enumerated type used to indicate if a given routine
 * should propagate Nan numbers
 * @param alpha scaling factor used to blend the computation result
 * @param beta scaling factor
 * @param stream tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopReduceProd(topsopTensorHandle_t output,
                 const topsopTensorHandle_t input,
                 const topsopSize_t *dimensions,
                 bool keepdims,
                 topsopNanPropagation_t reduceNanOpt,
                 topsopScalar_t alpha,
                 topsopScalar_t beta,
                 topsStream_t stream);

/**
 * @brief This function checks whether a multiplication is supported or not
 * with currest input tensor.
 *
 * @param input the input tensor
 * @param dimensions the dimensions to reduce
 * @return bool
 */
bool TOPSOP_EXPORT topsopReduceProdIsSupported(const topsopTensorHandle_t input,
                                               const topsopSize_t *dimensions);

/**
 * @brief This function gets the dimensions and rank of output tensor.
 *
 * @param input the input tensor
 * @param dimensions the dimensions to reduce
 * @param keepdims whether the output tensor has dim retained or not
 * @param dims the dimensions of output tensor
 * @param rank the rank of output tensor
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopReduceProdGetOutputDim(const topsopTensorHandle_t input,
                             const topsopSize_t *dimensions,
                             bool keepdims,
                             int64_t *dims,
                             int64_t *rank);

/**
 * @brief This function performs a maximum comparison.
 *
 * @param output the output tensor
 * @param input the input tensor
 * @param reduceNanOpt an enumerated type used to indicate if a given routine
 * should propagate Nan numbers
 * @param alpha scaling factor used to blend the computation result
 * @param beta scaling factor
 * @param stream tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopReduceMax(topsopTensorHandle_t output,
                const topsopTensorHandle_t input,
                const topsopSize_t *dimensions,
                bool keepdims,
                topsopNanPropagation_t reduceNanOpt,
                topsopScalar_t alpha,
                topsopScalar_t beta,
                topsStream_t stream);

/**
 * @brief This function checks whether a maximum comparison is supported or not
 * with currest input tensor.
 *
 * @param input the input tensor
 * @param dimensions the dimensions to reduce
 * @return bool
 */
bool TOPSOP_EXPORT topsopReduceMaxIsSupported(const topsopTensorHandle_t input,
                                              const topsopSize_t *dimensions);

/**
 * @brief This function gets the dimensions and rank of output tensor.
 *
 * @param input the input tensor
 * @param dimensions the dimensions to reduce
 * @param keepdims whether the output tensor has dim retained or not
 * @param dims the dimensions of output tensor
 * @param rank the rank of output tensor
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopReduceMaxGetOutputDim(const topsopTensorHandle_t input,
                            const topsopSize_t *dimensions,
                            bool keepdims,
                            int64_t *dims,
                            int64_t *rank);

/**
 * @brief This function performs a minimum comparison.
 *
 * @param output the output tensor
 * @param input the input tensor
 * @param reduceNanOpt an enumerated type used to indicate if a given routine
 * should propagate Nan numbers
 * @param alpha scaling factor used to blend the computation result
 * @param beta scaling factor
 * @param stream tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopReduceMin(topsopTensorHandle_t output,
                const topsopTensorHandle_t input,
                const topsopSize_t *dimensions,
                bool keepdims,
                topsopNanPropagation_t reduceNanOpt,
                topsopScalar_t alpha,
                topsopScalar_t beta,
                topsStream_t stream);

/**
 * @brief This function checks whether a minimum comparison is supported or not
 * with currest input tensor.
 *
 * @param input the input tensor
 * @param dimensions the dimensions to reduce
 * @return bool
 */
bool TOPSOP_EXPORT topsopReduceMinIsSupported(const topsopTensorHandle_t input,
                                              const topsopSize_t *dimensions);

/**
 * @brief This function gets the dimensions and rank of output tensor.
 *
 * @param input the input tensor
 * @param dimensions the dimensions to reduce
 * @param keepdims whether the output tensor has dim retained or not
 * @param dims the dimensions of output tensor
 * @param rank the rank of output tensor
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopReduceMinGetOutputDim(const topsopTensorHandle_t input,
                            const topsopSize_t *dimensions,
                            bool keepdims,
                            int64_t *dims,
                            int64_t *rank);

/**
 * @brief This function performs a maximum comparison of absolute values.
 *
 * @param output the output tensor
 * @param input the input tensor
 * @param reduceNanOpt an enumerated type used to indicate if a given routine
 * should propagate Nan numbers
 * @param alpha scaling factor used to blend the computation result
 * @param beta scaling factor
 * @param stream tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopReduceAmax(topsopTensorHandle_t output,
                 const topsopTensorHandle_t input,
                 const topsopSize_t *dimensions,
                 bool keepdims,
                 topsopNanPropagation_t reduceNanOpt,
                 topsopScalar_t alpha,
                 topsopScalar_t beta,
                 topsStream_t stream);

/**
 * @brief This function checks whether a maximum comparison of absolute values
 * is supported or not with current input tensor.
 *
 * @param input the input tensor
 * @param dimensions the dimensions to reduce
 * @return bool
 */
bool TOPSOP_EXPORT topsopReduceAmaxIsSupported(const topsopTensorHandle_t input,
                                               const topsopSize_t *dimensions);

/**
 * @brief This function gets the dimensions and rank of output tensor.
 *
 * @param input the input tensor
 * @param dimensions the dimensions to reduce
 * @param keepdims whether the output tensor has dim retained or not
 * @param dims the dimensions of output tensor
 * @param rank the rank of output tensor
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopReduceAmaxGetOutputDim(const topsopTensorHandle_t input,
                             const topsopSize_t *dimensions,
                             bool keepdims,
                             int64_t *dims,
                             int64_t *rank);

/**
 * @brief This function performs a reduce average computation.
 *
 * @param output the output tensor
 * @param input the input tensor
 * @param reduceNanOpt an enumerated type used to indicate if a given routine
 * should propagate Nan numbers
 * @param alpha scaling factor used to blend the computation result
 * @param beta scaling factor
 * @param stream tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopReduceAvg(topsopTensorHandle_t output,
                const topsopTensorHandle_t input,
                const topsopSize_t *dimensions,
                bool keepdims,
                topsopNanPropagation_t reduceNanOpt,
                topsopScalar_t alpha,
                topsopScalar_t beta,
                topsStream_t stream);

/**
 * @brief This function checks whether a averaging is supported or not
 * with current input tensor.
 *
 * @param input the input tensor
 * @param dimensions the dimensions to reduce
 * @return bool
 */
bool TOPSOP_EXPORT topsopReduceAvgIsSupported(const topsopTensorHandle_t input,
                                              const topsopSize_t *dimensions);

/**
 * @brief This function gets the dimensions and rank of output tensor.
 *
 * @param input the input tensor
 * @param dimensions the dimensions to reduce
 * @param keepdims whether the output tensor has dim retained or not
 * @param dims the dimensions of output tensor
 * @param rank the rank of output tensor
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopReduceAvgGetOutputDim(const topsopTensorHandle_t input,
                            const topsopSize_t *dimensions,
                            bool keepdims,
                            int64_t *dims,
                            int64_t *rank);

/**
 * @brief This function performs a addition of absolute values computation.
 *
 * @param output the output tensor
 * @param input the input tensor
 * @param reduceNanOpt an enumerated type used to indicate if a given routine
 * should propagate Nan numbers
 * @param alpha scaling factor used to blend the computation result
 * @param beta scaling factor
 * @param stream tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopReduceNorm1(topsopTensorHandle_t output,
                  const topsopTensorHandle_t input,
                  const topsopSize_t *dimensions,
                  bool keepdims,
                  topsopNanPropagation_t reduceNanOpt,
                  topsopScalar_t alpha,
                  topsopScalar_t beta,
                  topsStream_t stream);

/**
 * @brief This function checks whether a addition of absolute values is
 * supported or not with current input tensor.
 *
 * @param input the input tensor
 * @param dimensions the dimensions to reduce
 * @return bool
 */
bool TOPSOP_EXPORT topsopReduceNorm1IsSupported(
    const topsopTensorHandle_t input, const topsopSize_t *dimensions);

/**
 * @brief This function gets the dimensions and rank of output tensor.
 *
 * @param input the input tensor
 * @param dimensions the dimensions to reduce
 * @param keepdims whether the output tensor has dim retained or not
 * @param dims the dimensions of output tensor
 * @param rank the rank of output tensor
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopReduceNorm1GetOutputDim(const topsopTensorHandle_t input,
                              const topsopSize_t *dimensions,
                              bool keepdims,
                              int64_t *dims,
                              int64_t *rank);

/**
 * @brief This function performs a square root of the sum of squares
 * computation.
 *
 * @param output the output tensor
 * @param input the input tensor
 * @param reduceNanOpt an enumerated type used to indicate if a given routine
 * should propagate Nan numbers
 * @param alpha scaling factor used to blend the computation result
 * @param beta scaling factor
 * @param stream tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopReduceNorm2(topsopTensorHandle_t output,
                  const topsopTensorHandle_t input,
                  const topsopSize_t *dimensions,
                  bool keepdims,
                  topsopNanPropagation_t reduceNanOpt,
                  topsopScalar_t alpha,
                  topsopScalar_t beta,
                  topsStream_t stream);

/**
 * @brief This function checks whether a square root of the sum of squares is
 * supported or not with current input tensor.
 *
 * @param input the input tensor
 * @param dimensions the dimensions to reduce
 * @return bool
 */
bool TOPSOP_EXPORT topsopReduceNorm2IsSupported(
    const topsopTensorHandle_t input, const topsopSize_t *dimensions);

/**
 * @brief This function gets the dimensions and rank of output tensor.
 *
 * @param input the input tensor
 * @param dimensions the dimensions to reduce
 * @param keepdims whether the output tensor has dim retained or not
 * @param dims the dimensions of output tensor
 * @param rank the rank of output tensor
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopReduceNorm2GetOutputDim(const topsopTensorHandle_t input,
                              const topsopSize_t *dimensions,
                              bool keepdims,
                              int64_t *dims,
                              int64_t *rank);

/**
 * @brief This function performs a multiplication, not including elements of
 * value zero.
 *
 * @param output the output tensor
 * @param input the input tensor
 * @param reduceNanOpt an enumerated type used to indicate if a given routine
 * should propagate Nan numbers
 * @param alpha scaling factor used to blend the computation result
 * @param beta scaling factor
 * @param stream tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopReduceMulNoZeros(topsopTensorHandle_t output,
                       const topsopTensorHandle_t input,
                       const topsopSize_t *dimensions,
                       bool keepdims,
                       topsopNanPropagation_t reduceNanOpt,
                       topsopScalar_t alpha,
                       topsopScalar_t beta,
                       topsStream_t stream);

/**
 * @brief This function checks whether a multiplication, not including elements
 * of value zeor is supported or not with current input tensor.
 *
 * @param input the input tensor
 * @param dimensions the dimensions to reduce
 * @return bool
 */
bool TOPSOP_EXPORT topsopReduceMulNoZerosIsSupported(
    const topsopTensorHandle_t input, const topsopSize_t *dimensions);

/**
 * @brief This function gets the dimensions and rank of output tensor.
 *
 * @param input the input tensor
 * @param dimensions the dimensions to reduce
 * @param keepdims whether the output tensor has dim retained or not
 * @param dims the dimensions of output tensor
 * @param rank the rank of output tensor
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopReduceMulNoZerosGetOutputDim(const topsopTensorHandle_t input,
                                   const topsopSize_t *dimensions,
                                   bool keepdims,
                                   int64_t *dims,
                                   int64_t *rank);

/**
 * @brief This function performs a reduce or computation.
 *
 * @param output the output tensor
 * @param input the input tensor
 * @param dimensions the dimensions to reduce
 * @param keepdims whether the output tensor has dim retained or not
 * @param reduceNanOpt an enumerated type used to indicate if a given routine
 * should propagate Nan numbers
 * @param stream tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopReduceOr(topsopTensorHandle_t output,
                                            const topsopTensorHandle_t input,
                                            const topsopSize_t *dimensions,
                                            bool keepdims,
                                            topsopNanPropagation_t reduceNanOpt,
                                            topsStream_t stream);

/**
 * @brief This function checks whether an or operation is supported or not.
 *
 * @param input the input tensor
 * @param dimensions the dimensions to reduce
 * @return bool
 */
bool TOPSOP_EXPORT topsopReduceOrIsSupported(const topsopTensorHandle_t input,
                                             const topsopSize_t *dimensions);

/**
 * @brief This function gets the dimensions and rank of output tensor.
 *
 * @param input the input tensor
 * @param dimensions the dimensions to reduce
 * @param keepdims whether the output tensor has dim retained or not
 * @param dims the dimensions of output tensor
 * @param rank the rank of output tensor
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopReduceOrGetOutputDim(const topsopTensorHandle_t input,
                           const topsopSize_t *dimensions,
                           bool keepdims,
                           int64_t *dims,
                           int64_t *rank);

/**
 * @brief This function performs the reduce and computation.
 *
 * @param output the output tensor
 * @param input the input tensor
 * @param dimensions the dimensions to reduce
 * @param keepdims whether the output tensor has dim retained or not
 * @param reduceNanOpt an enumerated type used to indicate if a given routine
 * should propagate Nan numbers
 * @param stream tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopReduceAnd(topsopTensorHandle_t output,
                const topsopTensorHandle_t input,
                const topsopSize_t *dimensions,
                bool keepdims,
                topsopNanPropagation_t reduceNanOpt,
                topsStream_t stream);

/**
 * @brief This function checks whether an and operation is supported or not
 * with current input tensor.
 *
 * @param input the input tensor
 * @param dimensions the dimensions to reduce
 * @return bool
 */
bool TOPSOP_EXPORT topsopReduceAndIsSupported(const topsopTensorHandle_t input,
                                              const topsopSize_t *dimensions);

/**
 * @brief This function gets the dimensions and rank of output tensor.
 *
 * @param input the input tensor
 * @param dimensions the dimensions to reduce
 * @param keepdims whether the output tensor has dim retained or not
 * @param dims the dimensions of output tensor
 * @param rank the rank of output tensor
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopReduceAndGetOutputDim(const topsopTensorHandle_t input,
                            const topsopSize_t *dimensions,
                            bool keepdims,
                            int64_t *dims,
                            int64_t *rank);

/**
 * @brief This function performs the reduce xor computation.
 *
 * @param output the output tensor
 * @param input the input tensor
 * @param dimensions the dimensions to reduce
 * @param keepdims whether the output tensor has dim retained or not
 * @param reduceNanOpt an enumerated type used to indicate if a given routine
 * should propagate Nan numbers
 * @param stream tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopReduceXor(topsopTensorHandle_t output,
                const topsopTensorHandle_t input,
                const topsopSize_t *dimensions,
                bool keepdims,
                topsopNanPropagation_t reduceNanOpt,
                topsStream_t stream);

/**
 * @brief This function checks whether a xor operation is supported or not
 * with current input tensor.
 *
 * @param input the input tensor
 * @param dimensions the dimensions to reduce
 * @return bool
 */
bool TOPSOP_EXPORT topsopReduceXorIsSupported(const topsopTensorHandle_t input,
                                              const topsopSize_t *dimensions);

/**
 * @brief This function gets the dimensions and rank of output tensor.
 *
 * @param input the input tensor
 * @param dimensions the dimensions to reduce
 * @param keepdims whether the output tensor has dim retained or not
 * @param dims the dimensions of output tensor
 * @param rank the rank of output tensor
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopReduceXorGetOutputDim(const topsopTensorHandle_t input,
                            const topsopSize_t *dimensions,
                            bool keepdims,
                            int64_t *dims,
                            int64_t *rank);

/**
 * @brief This function gets the relative indices of maximum value for
 * the dimensions being reduced.
 *
 * @param output_val the output tensor of value
 * @param output_idx the output tensor of indices
 * @param input the input tensor
 * @param dimensions the dimensions to reduce
 * @param keepdims whether the output tensor has dim retained or not
 * @param select_last if false, returns the smallest index in case of identity
 * @param reduceNanOpt an enumerated type used to indicate if a given routine
 * should propagate Nan numbers
 * @param alpha scaling factor used to blend the computation result
 * @param beta scaling factor
 * @param stream tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopReduceArgmax(topsopTensorHandle_t output_val,
                   topsopTensorHandle_t output_idx,
                   const topsopTensorHandle_t input,
                   const topsopSize_t *dimensions,
                   bool keepdims,
                   bool select_last,
                   topsopNanPropagation_t reduceNanOpt,
                   topsopScalar_t alpha,
                   topsopScalar_t beta,
                   topsStream_t stream);

/**
 * @brief This function checks whether the argmax is supported or not
 * with current input and output tensor.
 *
 * @param input the input tensor
 * @param indice_type the data type of the output tensor indice
 * @param dimensions the dimensions to reduce
 * @return bool
 */
bool TOPSOP_EXPORT
topsopReduceArgmaxIsSupported(const topsopTensorHandle_t input,
                              topsopDataType_t indice_type,
                              const topsopSize_t *dimensions);

/**
 * @brief This function gets the dimensions and rank of output tensor.
 *
 * @param input the input tensor
 * @param dimensions the dimensions to reduce
 * @param keepdims whether the output tensor has dim retained or not
 * @param dims the dimensions of output tensor
 * @param rank the rank of output tensor
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopReduceArgmaxGetOutputDim(const topsopTensorHandle_t input,
                               const topsopSize_t *dimensions,
                               bool keepdims,
                               int64_t *dims,
                               int64_t *rank);

/**
 * @brief This function gets the relative indices of minimum value for
 * the dimensions being reduced.
 *
 * @param output_val the output tensor of value
 * @param output_idx the output tensor of indices
 * @param dimensions the dimensions to reduce
 * @param keepdims whether the output tensor has dim retained or not
 * @param select_last if false, returns the smallest index in case of identity
 * @param reduceNanOpt an enumerated type used to indicate if a given routine
 * should propagate Nan numbers
 * @param alpha scaling factor used to blend the computation result
 * @param beta scaling factor
 * @param stream tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopReduceArgmin(topsopTensorHandle_t output_val,
                   topsopTensorHandle_t output_idx,
                   const topsopTensorHandle_t input,
                   const topsopSize_t *dimensions,
                   bool keepdims,
                   bool select_last,
                   topsopNanPropagation_t reduceNanOpt,
                   topsopScalar_t alpha,
                   topsopScalar_t beta,
                   topsStream_t stream);

/**
 * @brief This function checks whether the argmin is supported or not
 * with current input and output tensor.
 *
 * @param input the input tensor
 * @param indice_type the data type of the output tensor indice
 * @param dimensions the dimensions to reduce
 * @return bool
 */
bool TOPSOP_EXPORT
topsopReduceArgminIsSupported(const topsopTensorHandle_t input,
                              topsopDataType_t indice_type,
                              const topsopSize_t *dimensions);

/**
 * @brief This function gets the dimensions and rank of output tensor.
 *
 * @param input the input tensor
 * @param dimensions the dimensions to reduce
 * @param keepdims whether the output tensor has dim retained or not
 * @param dims the dimensions of output tensor
 * @param rank the rank of output tensor
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopReduceArgminGetOutputDim(const topsopTensorHandle_t input,
                               const topsopSize_t *dimensions,
                               bool keepdims,
                               int64_t *dims,
                               int64_t *rank);

/**
 * @brief This function performs the select(XLA semantics) computation.
 * For each element P of pred, the corresponding element of
 * the output array is taken from on_true if the value of P is true,
 * and from on_false if the value of P is false.
 * @param out output tensor
 * @param pred pred tensor
 * @param lhs on_true tensor
 * @param rhs on_false tensor
 * @param alpha support any
 * @param beta only support 0.0
 * @param stream tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopSelect(topsopTensorHandle_t out,
                                          const topsopTensorHandle_t pred,
                                          const topsopTensorHandle_t lhs,
                                          const topsopTensorHandle_t rhs,
                                          const topsopScalar_t alpha,
                                          const topsopScalar_t beta,
                                          topsStream_t stream);

/**
 * @brief check whether current add operator support or not
 *
 * @param pred pred tensor
 * @param lhs on_true tensor
 * @param rhs on_false tensor
 * @param alpha support any
 * @param beta only support 0.0
 * @return bool
 */
bool TOPSOP_EXPORT topsopSelectIsSupported(const topsopTensorHandle_t pred,
                                           const topsopTensorHandle_t lhs,
                                           const topsopTensorHandle_t rhs,
                                           const topsopScalar_t alpha,
                                           const topsopScalar_t beta);

/**
 * @brief get output dims of current tensor select operator
 *
 * @param pred pred tensor
 * @param lhs on_true tensor
 * @param rhs on_false tensor
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopSelectGetOutputDim(const topsopTensorHandle_t pred,
                         const topsopTensorHandle_t lhs,
                         const topsopTensorHandle_t rhs,
                         int64_t *dims,
                         int64_t *rank);

/**
 * @brief This function implements the equation.
 *        y = (x[index] == 1 ? lhs[index] : rhs[0]) * alpha * beta.
 * @param out output tensor
 * @param pred pred tensor
 * @param lhs on_true tensor
 * @param rhs on_false tensor, the shape is [1]
 * @param alpha support any
 * @param beta only support 0.0
 * @param stream tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopMask(topsopTensorHandle_t out,
                                        const topsopTensorHandle_t pred,
                                        const topsopTensorHandle_t lhs,
                                        const topsopTensorHandle_t rhs,
                                        const topsopScalar_t alpha,
                                        const topsopScalar_t beta,
                                        topsStream_t stream);

/**
 * @brief check whether current add operator support or not
 *
 * @param pred pred tensor
 * @param lhs on_true tensor
 * @param rhs on_false tensor
 * @param alpha support any
 * @param beta only support 0.0
 * @return bool
 */
bool TOPSOP_EXPORT topsopMaskIsSupported(const topsopTensorHandle_t pred,
                                         const topsopTensorHandle_t lhs,
                                         const topsopTensorHandle_t rhs,
                                         const topsopScalar_t alpha,
                                         const topsopScalar_t beta);

/**
 * @brief get output dims of current tensor select operator
 *
 * @param pred pred tensor
 * @param lhs on_true tensor
 * @param rhs on_false tensor
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopMaskGetOutputDim(const topsopTensorHandle_t pred,
                       const topsopTensorHandle_t lhs,
                       const topsopTensorHandle_t rhs,
                       int64_t *dims,
                       int64_t *rank);

typedef struct topsopPoolingStruct *topsopPoolingDescriptor_t;
/**
 * @brief This function computes pooling of input values
 * (meaning, the maximum or average of several adjacent values)
 * to produce an output with smaller height and/or width.
 *
 * @param output The output tensor.
 * @param input The input tensor over which the windows slide.
 * @param mode The select of pooling computation(meaning, the maximum or
 *             average of several adjacent values).
 * @param windowHeight Height of the pooling window.
 * @param windowWidth Width of the pooling window.
 * @param verticalPadding Size of vertical padding.
 * @param horizontalPadding Size of horizontal padding.
 * @param verticalStride Pooling vertical stride.
 * @param horizontalStride Pooling horizontal stride.
 * @param alpha Scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor.
 * @param beta Scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor.
 * @param stream Tops stream.
 * @return topsopStatus_t
 */
topsopStatus_t topsopPooling2d(topsopTensorHandle_t output,
                               const topsopTensorHandle_t input,
                               const topsopPoolingMode_t mode,
                               int32_t windowHeight,
                               int32_t windowWidth,
                               int32_t verticalPadding,
                               int32_t horizontalPadding,
                               int32_t verticalStride,
                               int32_t horizontalStride,
                               topsopScalar_t alpha,
                               topsopScalar_t beta,
                               topsStream_t stream);
/**
 * @brief This function computes pooling of input values
 * (meaning, the maximum or average of several adjacent values)
 * to produce an output with smaller height and/or width.
 *
 * @param output The output tensor.
 * @param input The input tensor over which the windows slide.
 * @param mode The select of pooling computation(meaning, the maximum or
 *             average of several adjacent values).
 * @param inputRankA Dimension of the pooling operation. Must be greater than
 * zero.
 * @param windowDimA Array of dimension nbDims containing the window size for
 *                   each dimension. The value of array elements must
 *                   be greater than zero.
 * @param paddingA Array of dimension nbDims containing the padding size for
 *                 each dimension. Negative padding is allowed.
 * @param strideA Array of dimension nbDims containing the striding size for
 *                each dimension. The value of array elements must be greater
 *                than zero (meaning, negative striding size is not allowed).
 * @param alpha Scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor.
 * @param beta Scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor.
 * @param stream Tops stream.
 * @return topsopStatus_t
 */
topsopStatus_t topsopPoolingNd(topsopTensorHandle_t output,
                               const topsopTensorHandle_t input,
                               const topsopPoolingMode_t mode,
                               int64_t inputRankA,
                               topsopSize_t windowDimA,
                               topsopSize_t paddingA,
                               topsopSize_t strideA,
                               topsopScalar_t alpha,
                               topsopScalar_t beta,
                               topsStream_t stream);
/**
 * @brief Check whether current pooling operator support or not
 *
 * @param input The input tensor over which the windows slide.
 * @param mode The select of pooling computation(meaning, the maximum or
 *             average of several adjacent values).
 * @param windowHeight Height of the pooling window.
 * @param windowWidth Width of the pooling window.
 * @param verticalPadding Size of vertical padding.
 * @param horizontalPadding Size of horizontal padding.
 * @param verticalStride Pooling vertical stride.
 * @param horizontalStride Pooling horizontal stride.
 * @param alpha Scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor.
 * @param beta Scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor.
 * @return bool
 */
bool topsopPooling2dIsSupported(const topsopTensorHandle_t input,
                                const topsopPoolingMode_t mode,
                                const int32_t windowHeight,
                                const int32_t windowWidth,
                                const int32_t verticalPadding,
                                const int32_t horizontalPadding,
                                const int32_t verticalStride,
                                const int32_t horizontalStride,
                                const topsopScalar_t alpha,
                                const topsopScalar_t beta);
/**
 * @brief Get output dims of current tensor pooling operator
 *
 * @param input The input tensor over which the windows slide.
 * @param windowHeight Height of the pooling window.
 * @param windowWidth Width of the pooling window.
 * @param verticalPadding Size of vertical padding.
 * @param horizontalPadding Size of horizontal padding.
 * @param verticalStride Pooling vertical stride.
 * @param horizontalStride Pooling horizontal stride.
 * @param dims The dims of output tensor.
 * @param rank The rank of output tensor.
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopPooling2dGetOutputDim(const topsopTensorHandle_t input,
                            int32_t windowHeight,
                            int32_t windowWidth,
                            int32_t verticalPadding,
                            int32_t horizontalPadding,
                            int32_t verticalStride,
                            int32_t horizontalStride,
                            int64_t *dims,
                            int64_t *rank);

/**
 * @brief Check whether current pooling operator support or not
 *
 * @param input The input tensor over which the windows slide.
 * @param mode The select of pooling computation(meaning, the maximum or
 *             average of several adjacent values).
 * @param inputRankA Dimension of the pooling operation. Must be greater than
 * zero.
 * @param windowDimA Array of dimension nbDims containing the window size for
 *                   each dimension. The value of array elements must
 *                   be greater than zero.
 * @param paddingA Array of dimension nbDims containing the padding size for
 *                 each dimension. Negative padding is allowed.
 * @param strideA Array of dimension nbDims containing the striding size for
 *                each dimension. The value of array elements must be greater
 *                than zero (meaning, negative striding size is not allowed).
 * @param alpha Scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor.
 * @param beta Scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor.
 * @return bool
 */
bool topsopPoolingNdIsSupported(const topsopTensorHandle_t input,
                                const topsopPoolingMode_t mode,
                                int64_t inputRankA,
                                topsopSize_t windowDimA,
                                topsopSize_t paddingA,
                                topsopSize_t strideA,
                                topsopScalar_t alpha,
                                topsopScalar_t beta);

/**
 * @brief Get output dims of current tensor pooling operator
 *
 * @param input The input tensor over which the windows slide.
 * @param inputRankA Dimension of the pooling operation. Must be greater than
 * zero.
 * @param windowDimA Array of dimension nbDims containing the window size for
 *                   each dimension. The value of array elements must
 *                   be greater than zero.
 * @param paddingA Array of dimension nbDims containing the padding size for
 *                 each dimension. Negative padding is allowed.
 * @param strideA Array of dimension nbDims containing the striding size for
 *                each dimension. The value of array elements must be greater
 *                than zero (meaning, negative striding size is not allowed).
 * @param dims The dims of output tensor.
 * @param rank The rank of output tensor.
 * @return topsopStatus_t
 */
topsopStatus_t topsopPoolingNdGetOutputDim(const topsopTensorHandle_t input,
                                           int64_t inputRankA,
                                           topsopSize_t windowDimA,
                                           topsopSize_t paddingA,
                                           topsopSize_t strideA,
                                           int64_t *dims,
                                           int64_t *rank);

/**
 * @brief Applies a reduction function to all elements in each window of
 *        a sequence of N multi-dimensional arrays, producing a single or
 *        a tuple of N multi-dimensional arrays as output.
 *
 * @param output The output tensor.
 * @param input The input tensor over which the windows slide.
 * @param mode Reduction function of type T_0,..., T_{N-1}, T_0, ..., T_{N-1}
 *             -> Collate(T_0, ..., T_{N-1}), to apply to elements in
 *             each window of all the input operands.
 * @param windowDimensions Array of integers for window dimension values.
 * @param windowStrides Array of integers for window stride values.
 * @param baseDilations Array of integers for base dilation values.
 * @param windowDilations Array of integers for window dilation values.
 * @param paddings Operand padding values, indicates the padding value for each
 *                dimensions' high and low, size is the rank of operand
 *                multiplied by 2
 * @param autoPad Padding type for window.
 * @param ceilMode Whether to use ceil or floor (default) to
 *                 compute the output shape.
 * @param stream Tops stream.
 * @return topsopStatus_t
 */
topsopStatus_t topsopReduceWindow(topsopTensorHandle_t output,
                                  const topsopTensorHandle_t input,
                                  const topsopReduceWindowComputation_t mode,
                                  topsopSize_t windowDimensions,
                                  topsopSize_t windowStrides,
                                  topsopSize_t baseDilations,
                                  topsopSize_t windowDilations,
                                  topsopSize_t paddings,
                                  topsopReduceWindowAutoPad_t autoPad,
                                  bool ceilMode,
                                  topsStream_t stream);

/**
 * @brief Check whether current reducewindow operator support or not.
 * @param input The input tensor over which the windows slide.
 * @param mode Reduction function of type T_0,..., T_{N-1}, T_0, ..., T_{N-1}
 *             -> Collate(T_0, ..., T_{N-1}), to apply to elements in
 *             each window of all the input operands.
 * @param windowDimensions Array of integers for window dimension values.
 * @param windowStrides Array of integers for window stride values.
 * @param baseDilations Array of integers for base dilation values.
 * @param windowDilations Array of integers for window dilation values.
 * @param paddings Operand padding values, indicates the padding value for each
 *                dimensions' high and low, size is the rank of operand
 *                multiplied by 2
 * @param autoPad Padding type for window.
 * @param ceilMode Whether to use ceil or floor (default) to
 *                 compute the output shape.
 * @return bool
 */
bool topsopReduceWindowIsSupported(const topsopTensorHandle_t input,
                                   const topsopReduceWindowComputation_t mode,
                                   topsopSize_t windowDimensions,
                                   topsopSize_t windowStrides,
                                   topsopSize_t baseDilations,
                                   topsopSize_t windowDilations,
                                   topsopSize_t paddings,
                                   topsopReduceWindowAutoPad_t autoPad,
                                   bool ceilMode);

/**
 * @brief Get output dims of current tensor reducewindow operator.
 *
 * @param input The input tensor over which the windows slide.
 * @param windowDimensions Array of integers for window dimension values.
 * @param windowStrides Array of integers for window stride values.
 * @param baseDilations Array of integers for base dilation values.
 * @param windowDilations Array of integers for window dilation values.
 * @param paddings Array of integers for window padding values.
 * @param autoPad Operand padding values, indicates the padding value for each
 *                dimensions' high and low, size is the rank of operand
 *                multiplied by 2
 * @param ceilMode Whether to use ceil or floor (default) to
 *                 compute the output shape.
 * @param dims The dims of output tensor.
 * @param rank The rank of output tensor.
 * @return topsopStatus_t
 */
topsopStatus_t topsopReduceWindowGetOutputDim(
    const topsopTensorHandle_t input,
    topsopSize_t windowDimensions,
    topsopSize_t windowStrides,
    topsopSize_t baseDilations,
    topsopSize_t windowDilations,
    topsopSize_t paddings,
    topsopReduceWindowAutoPad_t autoPad,
    bool ceilMode,
    int64_t *dims,
    int64_t *rank);

/**
 * @brief This function computes concatenate of input values
 * (meaning, the maximum or average of several adjacent values)
 * to produce an output with smaller height and/or width.
 *
 * @param output The output tensor.
 * @param inputs List of input tensor
 * @param inputs_number Number of input tensor
 * @param dimension A value in the interval [0, N) that names the dimension to
 *                  be concatenated between the operands.
 * @param alpha Scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor.
 * @param beta Scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor.
 * @param stream Tops stream.
 * @return topsopStatus_t
 */
topsopStatus_t topsopConcatenate(topsopTensorHandle_t output,
                                 const topsopTensorHandle_t *inputs,
                                 int64_t inputs_number,
                                 int64_t dimension,
                                 topsopScalar_t alpha,
                                 topsopScalar_t beta,
                                 topsStream_t stream);

/**
 * @brief Check whether current concatenate operator support or not.
 *
 * @param inputs List of input tensor
 * @param inputs_number Number of input tensor
 * @param dimension A value in the interval [0, N) that names the dimension to
 *                  be concatenated between the operands.
 * @param alpha Scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor.
 * @param beta Scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor.
 * @return bool
 */
bool topsopConcatenateIsSupported(const topsopTensorHandle_t *inputs,
                                  int64_t inputs_number,
                                  int64_t dimension,
                                  topsopScalar_t alpha,
                                  topsopScalar_t beta);

/**
 * @brief Get output dims of current tensor concatenate operator..
 *
 * @param inputs List of input tensor
 * @param inputs_number Number of input tensor
 * @param dimension A value in the interval [0, N) that names the dimension to
 *                  be concatenated between the operands.
 * @param dims The dims of output tensor.
 * @param rank The rank of output tensor.
 * @return topsopStatus_t
 */
topsopStatus_t topsopConcatenateGetOutputDim(const topsopTensorHandle_t *inputs,
                                             int64_t inputs_number,
                                             int64_t dimension,
                                             int64_t *dims,
                                             int64_t *rank);

/**
 * @brief This function computes globalaveragepool of input values
 * (meaning, the maximum or average of several adjacent values)
 * to produce an output with smaller height and/or width.
 *
 * @param output The output tensor.
 * @param input The input tensor over which the windows slide.
 * @param alpha Scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor.
 * @param beta Scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor.
 * @param stream Tops stream.
 * @return topsopStatus_t
 * @return topsopStatus_t
 */
topsopStatus_t topsopGlobalAveragePool(topsopTensorHandle_t output,
                                       const topsopTensorHandle_t input,
                                       topsopScalar_t alpha,
                                       topsopScalar_t beta,
                                       topsStream_t stream);

/**
 * @brief Get output dims of current tensor globalaveragepool operator.
 *
 * @param inputs List of input tensor.
 * @param dims The dims of output tensor.
 * @param rank The rank of output tensor.
 * @return topsopStatus_t
 */
topsopStatus_t topsopGlobalAveragePoolGetOutputDim(
    const topsopTensorHandle_t input, int64_t *dims, int64_t *rank);

/**
 * @brief This function checks whether globalaveragepool is supported or not
 * with currest input tensor.
 *
 * @param input the input tensor.
 * @param alpha Scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor.
 * @param beta Scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor.
 * @return bool
 */
bool topsopGlobalAveragePoolIsSupported(const topsopTensorHandle_t input,
                                        topsopScalar_t alpha,
                                        topsopScalar_t beta);

/**
 * @brief Broadcast operator
 *
 * @param output The output tensor
 * @param input The input tensor
 * @param dim_map The index of input relative to output
 * @param alpha Pointers to scaling factors (in host memory) used to
 * blend the layer output value with prior value in the destination tensor,
 * temporarily must be 1.0
 * @param beta Pointers to scaling factors (in host memory) used to
 * blend the layer output value with prior value in the destination tensor
 * temporarily must be 0.0
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t topsopBroadcast(topsopTensorHandle_t output,
                               const topsopTensorHandle_t input,
                               topsopSize_t dim_map,
                               topsopScalar_t alpha,
                               topsopScalar_t beta,
                               topsStream_t stream);

/**
 * @brief check whether current broadcast operator support or not
 *
 * @param input The input tensor
 * @param dim_map The index of input relative to output
 * @param alpha Pointers to scaling factors (in host memory) used to
 * blend the layer output value with prior value in the destination tensor,
 * temporarily must be 1.0
 * @param beta Pointers to scaling factors (in host memory) used to
 * blend the layer output value with prior value in the destination tensor
 * temporarily must be 0.0
 * @return bool
 */
bool topsopBroadcastIsSupported(const topsopTensorHandle_t input,
                                topsopSize_t dim_map,
                                topsopScalar_t alpha,
                                topsopScalar_t beta);

/**
 * @brief get output dims of current tensor of Broadcast operator
 *
 * @param broadcasted_dims The output dimension after the broadcast
 * @param dims The dims of output tensor
 * @param rank The rank of output tensor
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopBroadcastGetOutputDim(
    topsopSize_t broadcasted_dims, int64_t *dims, int64_t *rank);

/**
 * @brief Resize operator
 *
 * @param output The output tensor
 * @param input The input tensor
 * @param sizes The output dimension after the resize,
 * output size = input size * scales
 * @param interpolationMode Configure for resize mode candidate:
 * TOPSOP_RESIZE_NEAREST, TOPSOP_RESIZE_BILINEAR
 * @param transMode Configure for resize mode candidate:
 * TOPSOP_RESIZE_HALF_PIXEL, TOPSOP_RESIZE_ASYMMETRIC,
 * TOPSOP_RESIZE_PYTORCH_HALF_PIXEL, TOPSOP_RESIZE_ALIGN_CORNERS
 * @param nearestMode Configure for resize mode candidate:
 * TOPSOP_RESIZE_ROUND_PREFER_FLOOR, TOPSOP_RESIZE_ROUND_PREFER_CEIL,
 * TOPSOP_RESIZE_FLOOR, TOPSOP_RESIZE_CEIL
 * @param alpha Pointers to scaling factors (in host memory) used to
 * blend the layer output value with prior value in the destination tensor,
 * temporarily must be 1.0
 * @param beta Pointers to scaling factors (in host memory) used to
 * blend the layer output value with prior value in the destination tensor
 * temporarily must be 0.0
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t topsopResize(topsopTensorHandle_t output,
                            const topsopTensorHandle_t input,
                            topsopSize_t sizes,
                            topsopResizeInterpolationMode_t interpolationMode,
                            topsopResizeCoordTransMode_t transMode,
                            topsopResizeNearestMode_t nearestMode,
                            topsopScalar_t alpha,
                            topsopScalar_t beta,
                            topsStream_t stream);

/**
 * @brief check whether current resize operator support or not
 *
 * @param input The input tensor
 * @param sizes The output dimension after the resize,
 * output size = input size * scales
 * @param interpolationMode Configure for resize mode candidate:
 * TOPSOP_RESIZE_NEAREST, TOPSOP_RESIZE_BILINEAR
 * @param transMode Configure for resize mode candidate:
 * TOPSOP_RESIZE_HALF_PIXEL, TOPSOP_RESIZE_ASYMMETRIC,
 * TOPSOP_RESIZE_PYTORCH_HALF_PIXEL, TOPSOP_RESIZE_ALIGN_CORNERS
 * @param nearestMode Configure for resize mode candidate:
 * TOPSOP_RESIZE_ROUND_PREFER_FLOOR, TOPSOP_RESIZE_ROUND_PREFER_CEIL,
 * TOPSOP_RESIZE_FLOOR, TOPSOP_RESIZE_CEIL
 * @param alpha Pointers to scaling factors (in host memory) used to
 * blend the layer output value with prior value in the destination tensor,
 * temporarily must be 1.0
 * @param beta Pointers to scaling factors (in host memory) used to
 * blend the layer output value with prior value in the destination tensor
 * temporarily must be 0.0
 * @return bool
 */
bool topsopResizeIsSupported(const topsopTensorHandle_t input,
                             topsopSize_t sizes,
                             topsopResizeInterpolationMode_t interpolationMode,
                             topsopResizeCoordTransMode_t transMode,
                             topsopResizeNearestMode_t nearestMode,
                             topsopScalar_t alpha,
                             topsopScalar_t beta);

/**
 * @brief get output dims of current tensor of
 * Resize operator
 * @param sizes The output dimension after the resize,
 * output size = input size * scales
 * @param dims The dims of output tensor
 * @param rank The rank of output tensor
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopResizeGetOutputDim(topsopSize_t sizes,
                                                      int64_t *dims,
                                                      int64_t *rank);

/**
 * @brief Resizegrad operator
 *
 * @param output The output tensor, it corresponds to input_grad
 * @param input The input tensor, it corresponds to output_grad
 * @param sizes The output dimension after the resize grad,
 * output_grad(input) size = input_grad(output) * scales
 * @param interpolationMode Configure for resizegrad mode candidate:
 * TOPSOP_RESIZE_NEAREST, TOPSOP_RESIZE_BILINEAR
 * @param transMode Configure for resizegrad mode candidate:
 * TOPSOP_RESIZE_HALF_PIXEL, TOPSOP_RESIZE_ASYMMETRIC,
 * TOPSOP_RESIZE_PYTORCH_HALF_PIXEL, TOPSOP_RESIZE_ALIGN_CORNERS
 * @param nearestMode Configure for resizegrad mode candidate:
 * TOPSOP_RESIZE_ROUND_PREFER_FLOOR, TOPSOP_RESIZE_ROUND_PREFER_CEIL,
 * TOPSOP_RESIZE_FLOOR, TOPSOP_RESIZE_CEIL
 * @param alpha Pointers to scaling factors (in host memory) used to
 * blend the layer output value with prior value in the destination tensor,
 * temporarily must be 1.0
 * @param beta Pointers to scaling factors (in host memory) used to
 * blend the layer output value with prior value in the destination tensor
 * temporarily must be 0.0
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t topsopResizeGrad(
    topsopTensorHandle_t output,
    const topsopTensorHandle_t input,
    topsopSize_t sizes,
    topsopResizeInterpolationMode_t interpolationMode,
    topsopResizeCoordTransMode_t transMode,
    topsopResizeNearestMode_t nearestMode,
    topsopScalar_t alpha,
    topsopScalar_t beta,
    topsStream_t stream);

/**
 * @brief check whether current resizegrad operator support or not
 *
 * @param input The input tensor, it corresponds to output_grad
 * @param sizes The output dimension after the resize grad,
 * output_grad(input) size = input_grad(output) * scales
 * @param interpolationMode Configure for resizegrad mode candidate:
 * TOPSOP_RESIZE_NEAREST, TOPSOP_RESIZE_BILINEAR
 * @param transMode Configure for resizegrad mode candidate:
 * TOPSOP_RESIZE_HALF_PIXEL, TOPSOP_RESIZE_ASYMMETRIC,
 * TOPSOP_RESIZE_PYTORCH_HALF_PIXEL, TOPSOP_RESIZE_ALIGN_CORNERS
 * @param nearestMode Configure for resizegrad mode candidate:
 * TOPSOP_RESIZE_ROUND_PREFER_FLOOR, TOPSOP_RESIZE_ROUND_PREFER_CEIL,
 * TOPSOP_RESIZE_FLOOR, TOPSOP_RESIZE_CEIL
 * @param alpha Pointers to scaling factors (in host memory) used to
 * blend the layer output value with prior value in the destination tensor,
 * temporarily must be 1.0
 * @param beta Pointers to scaling factors (in host memory) used to
 * blend the layer output value with prior value in the destination tensor
 * temporarily must be 0.0
 * @return bool
 */
bool topsopResizeGradIsSupported(
    const topsopTensorHandle_t input,
    topsopSize_t sizes,
    topsopResizeInterpolationMode_t interpolationMode,
    topsopResizeCoordTransMode_t transMode,
    topsopResizeNearestMode_t nearestMode,
    topsopScalar_t alpha,
    topsopScalar_t beta);

/**
 * @brief get output dims of current tensor of
 * Resizegrad operator
 * @param sizes The output dimension after the resize grad,
 * output_grad(input) size = input_grad(output) * scales
 * @param dims The dims of output tensor
 * @param rank The rank of output tensor
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopResizeGradGetOutputDim(topsopSize_t sizes,
                                                          int64_t *dims,
                                                          int64_t *rank);

/**
 * @brief dot operator
 *
 * @param out the output tensor
 * @param lhs the input tensor of lhs
 * @param rhs the input tensor of rhs
 * @param alpha scaling factor used to blend the computation result
 * @param beta scaling factor
 * @param stream tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopDot(topsopTensorHandle_t out,
                                       const topsopTensorHandle_t lhs,
                                       const topsopTensorHandle_t rhs,
                                       const topsopScalar_t alpha,
                                       const topsopScalar_t beta,
                                       topsStream_t stream);

/**
 * @brief general dot operator
 *
 * @param out the output tensor
 * @param lhs the input tensor of lhs
 * @param rhs the input tensor of rhs
 * @param lhsContractingDimension contracting dimensions info of lhs
 * @param rhsContractingDimension contracting dimensions info of rhs
 * @param lhsBatchDimension batch dimensions info of lhs
 * @param rhsBatchDimension batch dimensions info of rhs
 * @param alpha scaling factor used to blend the computation result
 * @param beta scaling factor
 * @param stream tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopDotGeneral(topsopTensorHandle_t out,
                 const topsopTensorHandle_t lhs,
                 const topsopTensorHandle_t rhs,
                 const topsopSize_t *lhsContractingDimension,
                 const topsopSize_t *rhsContractingDimension,
                 const topsopSize_t *lhsBatchDimension,
                 const topsopSize_t *rhsBatchDimension,
                 const topsopScalar_t alpha,
                 const topsopScalar_t beta,
                 topsStream_t stream);

/**
 * @brief general dot bias operator
 *
 * @param out the output tensor
 * @param lhs the input tensor of lhs
 * @param rhs the input tensor of rhs
 * @param bias the input tensor of bias
 * @param lhsContractingDimension contracting dimensions info of lhs
 * @param rhsContractingDimension contracting dimensions info of rhs
 * @param lhsBatchDimension batch dimensions info of lhs
 * @param rhsBatchDimension batch dimensions info of rhs
 * @param alpha scaling factor used to blend the computation result
 * @param beta scaling factor
 * @param stream tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopDotBias(topsopTensorHandle_t out,
              const topsopTensorHandle_t lhs,
              const topsopTensorHandle_t rhs,
              const topsopTensorHandle_t bias,
              const topsopSize_t *lhsContractingDimension,
              const topsopSize_t *rhsContractingDimension,
              const topsopSize_t *lhsBatchDimension,
              const topsopSize_t *rhsBatchDimension,
              const topsopScalar_t alpha,
              const topsopScalar_t beta,
              topsStream_t stream);

/**
 * @brief check whether dot operator is supported or not with
 * current input tensor
 *
 * @param lhs the input tensor of lhs
 * @param rhs the input tensor of rhs
 *
 * @return bool
 */
bool TOPSOP_EXPORT topsopDotIsSupported(const topsopTensorHandle_t lhs,
                                        const topsopTensorHandle_t rhs);

/**
 * @brief get the dimensions and rank of output tensor
 *
 * @param lhs the input tensor of lhs
 * @param rhs the input tensor of rhs
 *
 * @param dims the dimensions of output tensor
 * @param rank the rank of output tensor
 * @return bool
 */
topsopStatus_t TOPSOP_EXPORT
topsopDotGetOutputDim(const topsopTensorHandle_t lhs,
                      const topsopTensorHandle_t rhs,
                      int64_t *dims,
                      int64_t *rank);

/**
 * @brief check whether dot operator is supported or not with
 * current input tensor
 *
 * @param lhs the input tensor of lhs
 * @param rhs the input tensor of rhs
 * @param lhsContractingDim 'lhs' contracting dimension numbers
 * @param rhsContractingDim 'rhs' contracting dimension numbers
 * @param lhsBatchDim 'lhs' batch dimension numbers
 * @param rhsBatchDim 'rhs' batch dimension numbers
 *
 * @return bool
 */
bool TOPSOP_EXPORT
topsopDotGeneralIsSupported(const topsopTensorHandle_t lhs,
                            const topsopTensorHandle_t rhs,
                            const topsopSize_t *lhsContractingDim,
                            const topsopSize_t *rhsContractingDim,
                            const topsopSize_t *lhsBatchDim,
                            const topsopSize_t *rhsBatchDim);

/**
 * @brief get the dimensions and rank of output tensor
 *
 * @param lhs the input tensor of lhs
 * @param rhs the input tensor of rhs
 * @param lhsContractingDim 'lhs' contracting dimension numbers
 * @param rhsContractingDim 'rhs' contracting dimension numbers
 * @param lhsBatchDim 'lhs' batch dimension numbers
 * @param rhsBatchDim 'rhs' batch dimension numbers
 *
 * @param dims the dimensions of output tensor
 * @param rank the rank of output tensor
 * @return bool
 */
topsopStatus_t TOPSOP_EXPORT
topsopDotGeneralGetOutputDim(const topsopTensorHandle_t lhs,
                             const topsopTensorHandle_t rhs,
                             const topsopSize_t *lhsContractingDim,
                             const topsopSize_t *rhsContractingDim,
                             const topsopSize_t *lhsBatchDim,
                             const topsopSize_t *rhsBatchDim,
                             int64_t *dims,
                             int64_t *rank);

/**
 * @brief check whether dot bias operator is supported or not with
 * current input tensor
 *
 * @param lhs the input tensor of lhs
 * @param rhs the input tensor of rhs
 * @param bias the input tensor of bias
 * @param lhsContractingDim 'lhs' contracting dimension numbers
 * @param rhsContractingDim 'rhs' contracting dimension numbers
 * @param lhsBatchDim 'lhs' batch dimension numbers
 * @param rhsBatchDim 'rhs' batch dimension numbers
 *
 * @return bool
 */
bool TOPSOP_EXPORT
topsopDotBiasIsSupported(const topsopTensorHandle_t lhs,
                         const topsopTensorHandle_t rhs,
                         const topsopTensorHandle_t bias,
                         const topsopSize_t *lhsContractingDim,
                         const topsopSize_t *rhsContractingDim,
                         const topsopSize_t *lhsBatchDim,
                         const topsopSize_t *rhsBatchDim);

/**
 * @brief get the dimensions and rank of output tensor
 *
 * @param lhs the input tensor of lhs
 * @param rhs the input tensor of rhs
 * @param lhsContractingDim 'lhs' contracting dimension numbers
 * @param rhsContractingDim 'rhs' contracting dimension numbers
 * @param lhsBatchDim 'lhs' batch dimension numbers
 * @param rhsBatchDim 'rhs' batch dimension numbers
 *
 * @param dims the dimensions of output tensor
 * @param rank the rank of output tensor
 * @return bool
 */
topsopStatus_t TOPSOP_EXPORT
topsopDotBiasGetOutputDim(const topsopTensorHandle_t lhs,
                          const topsopTensorHandle_t rhs,
                          const topsopSize_t *lhsContractingDim,
                          const topsopSize_t *rhsContractingDim,
                          const topsopSize_t *lhsBatchDim,
                          const topsopSize_t *rhsBatchDim,
                          int64_t *dims,
                          int64_t *rank);

/**
 * @brief This function performs the rectified linear function.
 *
 * @param output Result tensor
 * @param input Input tensor
 * @param reluNanOpt Enumerant to specify the Nan propagation mode
 * @param alpha scaling factors (in host memory) used to blend the
 *              layer output value with prior value in the destination tensor
 * @param beta scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopReluForward(topsopTensorHandle_t output,
                  const topsopTensorHandle_t input,
                  topsopNanPropagation_t reluNanOpt,
                  topsopScalar_t alpha,
                  topsopScalar_t beta,
                  topsStream_t stream);

/**
 * @brief This function performs the sigmoid function.
 *
 * @param output Result tensor
 * @param input Input tensor
 * @param reluNanOpt Enumerant to specify the Nan propagation mode
 * @param alpha scaling factors (in host memory) used to blend the
 *              layer output value with prior value in the destination tensor
 * @param beta scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopSigmoidForward(topsopTensorHandle_t output,
                     const topsopTensorHandle_t input,
                     topsopNanPropagation_t reluNanOpt,
                     topsopScalar_t alpha,
                     topsopScalar_t beta,
                     topsStream_t stream);

/**
 * @brief This function performs the clipped rectified linear function.
 *
 * @param output Result tensor
 * @param input Input tensor
 * @param reluNanOpt Enumerant to specify the Nan propagation mode
 * @param coef float type now, f(x) = min(max(0, x), coef)
 * @param alpha scaling factors (in host memory) used to blend the
 *              layer output value with prior value in the destination tensor
 * @param beta scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopClippedReluForward(topsopTensorHandle_t output,
                         const topsopTensorHandle_t input,
                         topsopNanPropagation_t reluNanOpt,
                         topsopScalar_t coef,
                         topsopScalar_t alpha,
                         topsopScalar_t beta,
                         topsStream_t stream);

/**
 * @brief This function performs the exponential linear function.
 *
 * @param output Result tensor
 * @param input Input tensor
 * @param reluNanOpt Enumerant to specify the Nan propagation mode
 * @param coef  float type now, if x >= 0, f(x) = x
 *              if x < 0, f(x) = coef * (exp(x) - 1)
 * @param alpha scaling factors (in host memory) used to blend the
 *              layer output value with prior value in the destination tensor
 * @param beta scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopEluForward(topsopTensorHandle_t output,
                                              const topsopTensorHandle_t input,
                                              topsopNanPropagation_t reluNanOpt,
                                              topsopScalar_t coef,
                                              topsopScalar_t alpha,
                                              topsopScalar_t beta,
                                              topsStream_t stream);

/**
 * @brief This function performs the hyperbolic tangent function.
 *
 * @param output Result tensor
 * @param input Input tensor
 * @param reluNanOpt Enumerant to specify the Nan propagation mode
 * @param alpha scaling factors (in host memory) used to blend the
 *              layer output value with prior value in the destination tensor
 * @param beta scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopTanhForward(topsopTensorHandle_t output,
                  const topsopTensorHandle_t input,
                  topsopNanPropagation_t reluNanOpt,
                  topsopScalar_t alpha,
                  topsopScalar_t beta,
                  topsStream_t stream);

/**
 * @brief This function performs the swish function.
 *
 * @param output Result tensor
 * @param input Input tensor
 * @param reluNanOpt Enumerant to specify the Nan propagation mode
 * @param alpha scaling factors (in host memory) used to blend the
 *              layer output value with prior value in the destination tensor
 * @param beta scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopSwishForward(topsopTensorHandle_t output,
                   const topsopTensorHandle_t input,
                   topsopNanPropagation_t reluNanOpt,
                   topsopScalar_t alpha,
                   topsopScalar_t beta,
                   topsStream_t stream);

/**
 * @brief This function performs the leaky relu function.
 *
 * @param output Result tensor
 * @param input Input tensor
 * @param reluNanOpt Enumerant to specify the Nan propagation mode
 * @param coef float type now, if x >= 0, f(x) = x; if x < 0, f(x) = coef * x
 * @param alpha scaling factors (in host memory) used to blend the
 *              layer output value with prior value in the destination tensor
 * @param beta scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopLeakyReluForward(topsopTensorHandle_t output,
                       const topsopTensorHandle_t input,
                       topsopNanPropagation_t reluNanOpt,
                       topsopScalar_t coef,
                       topsopScalar_t alpha,
                       topsopScalar_t beta,
                       topsStream_t stream);

/**
 * @brief check whether current relu forward operator support or not
 *
 * @param input Input tensor
 * @param reluNanOpt Enumerant to specify the Nan propagation mode
 * @param alpha scaling factors (in host memory) used to blend the
 *              layer output value with prior value in the destination tensor
 * @param beta scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor
 * @return bool
 */
bool TOPSOP_EXPORT
topsopReluForwardIsSupported(const topsopTensorHandle_t input,
                             topsopNanPropagation_t reluNanOpt,
                             topsopScalar_t alpha,
                             topsopScalar_t beta);

/**
 * @brief get output dims of current tensor relu forward operator
 *
 * @param input Input tensor
 * @param dims The dims of output tensor
 * @param rank The rank of output tensor
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopReluForwardGetOutputDim(
    const topsopTensorHandle_t input, int64_t *dims, int64_t *rank);

/**
 * @brief check whether current sigmoid forward operator support or not
 *
 * @param input Input tensor
 * @param reluNanOpt Enumerant to specify the Nan propagation mode
 * @param alpha scaling factors (in host memory) used to blend the
 *              layer output value with prior value in the destination tensor
 * @param beta scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor
 * @return bool
 */
bool TOPSOP_EXPORT
topsopSigmoidForwardIsSupported(const topsopTensorHandle_t input,
                                topsopNanPropagation_t reluNanOpt,
                                topsopScalar_t alpha,
                                topsopScalar_t beta);

/**
 * @brief get output dims of current tensor sigmoid forward operator
 *
 * @param input Input tensor
 * @param dims The dims of output tensor
 * @param rank The rank of output tensor
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopSigmoidForwardGetOutputDim(
    const topsopTensorHandle_t input, int64_t *dims, int64_t *rank);

/**
 * @brief check whether current clipped relu forward operator support or not
 *
 * @param input Input tensor
 * @param reluNanOpt Enumerant to specify the Nan propagation mode
 * @param coef Floating point number, f(x) = min(max(0, x), coef)
 * @param alpha scaling factors (in host memory) used to blend the
 *              layer output value with prior value in the destination tensor
 * @param beta scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor
 * @return bool
 */
bool TOPSOP_EXPORT
topsopClippedReluForwardIsSupported(const topsopTensorHandle_t input,
                                    topsopNanPropagation_t reluNanOpt,
                                    topsopScalar_t coef,
                                    topsopScalar_t alpha,
                                    topsopScalar_t beta);

/**
 * @brief get output dims of current tensor clipped relu forward operator
 *
 * @param input Input tensor
 * @param dims The dims of output tensor
 * @param rank The rank of output tensor
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopClippedReluForwardGetOutputDim(
    const topsopTensorHandle_t input, int64_t *dims, int64_t *rank);

/**
 * @brief check whether current elu forward operator support or not
 *
 * @param input Input tensor
 * @param reluNanOpt Enumerant to specify the Nan propagation mode
 * @param coef Floating point number, if x >= 0, f(x) = x
 *             if x < 0, f(x) = coef * (exp(x) - 1)
 * @param alpha scaling factors (in host memory) used to blend the
 *              layer output value with prior value in the destination tensor
 * @param beta scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor
 * @return bool
 */
bool TOPSOP_EXPORT
topsopEluForwardIsSupported(const topsopTensorHandle_t input,
                            topsopNanPropagation_t reluNanOpt,
                            topsopScalar_t coef,
                            topsopScalar_t alpha,
                            topsopScalar_t beta);

/**
 * @brief get output dims of current tensor elu forward operator
 *
 * @param input Input tensor
 * @param dims The dims of output tensor
 * @param rank The rank of output tensor
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopEluForwardGetOutputDim(
    const topsopTensorHandle_t input, int64_t *dims, int64_t *rank);

/**
 * @brief check whether current tanh forward operator support or not
 *
 * @param input Input tensor
 * @param reluNanOpt Enumerant to specify the Nan propagation mode
 * @param alpha scaling factors (in host memory) used to blend the
 *              layer output value with prior value in the destination tensor
 * @param beta scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor
 * @return bool
 */
bool TOPSOP_EXPORT
topsopTanhForwardIsSupported(const topsopTensorHandle_t input,
                             topsopNanPropagation_t reluNanOpt,
                             topsopScalar_t alpha,
                             topsopScalar_t beta);

/**
 * @brief get output dims of current tensor tanh forward operator
 *
 * @param input Input tensor
 * @param dims The dims of output tensor
 * @param rank The rank of output tensor
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopTanhForwardGetOutputDim(
    const topsopTensorHandle_t input, int64_t *dims, int64_t *rank);

/**
 * @brief check whether current swish forward operator support or not
 *
 * @param input Input tensor
 * @param reluNanOpt Enumerant to specify the Nan propagation mode
 * @param alpha scaling factors (in host memory) used to blend the
 *              layer output value with prior value in the destination tensor
 * @param beta scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor
 * @return bool
 */
bool TOPSOP_EXPORT
topsopSwishForwardIsSupported(const topsopTensorHandle_t input,
                              topsopNanPropagation_t reluNanOpt,
                              topsopScalar_t alpha,
                              topsopScalar_t beta);

/**
 * @brief get output dims of current tensor swish forward operator
 *
 * @param input Input tensor
 * @param dims The dims of output tensor
 * @param rank The rank of output tensor
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopSwishForwardGetOutputDim(
    const topsopTensorHandle_t input, int64_t *dims, int64_t *rank);

/**
 * @brief check whether current leaky relu forward operator support or not
 *
 * @param input Input tensor
 * @param reluNanOpt Enumerant to specify the Nan propagation mode
 * @param coef Floating point number, if x >= 0, f(x) = x
 *             if x < 0, f(x) = coef * x
 * @param alpha scaling factors (in host memory) used to blend the
 *              layer output value with prior value in the destination tensor
 * @param beta scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor
 * @return bool
 */
bool TOPSOP_EXPORT
topsopLeakyReluForwardIsSupported(const topsopTensorHandle_t input,
                                  topsopNanPropagation_t reluNanOpt,
                                  topsopScalar_t coef,
                                  topsopScalar_t alpha,
                                  topsopScalar_t beta);

/**
 * @brief get output dims of current tensor leaky relu forward operator
 *
 * @param input Input tensor
 * @param dims The dims of output tensor
 * @param rank The rank of output tensor
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopLeakyReluForwardGetOutputDim(
    const topsopTensorHandle_t input, int64_t *dims, int64_t *rank);

/**
 * @brief This function performs the pad function.
 *
 * @param output output tensor
 * @param input input tensor
 * @param pads pads should be a 1D array of shape [2 * rank] or [3 * rank].
 *             pads format should be:
 *                [x1_begin, x2_begin, ..., x1_end, x2_end,...]
 *             or
 *                [x1_begin, ..., x1_end, ..., x1_mid,...]
 *             if mode chosen not is `constant`, x1_mid cannot take effect.
 * @param mode mode = 0 (constant pad), mode = 1(reflect pad), mode = 2(edge
 * pad)
 * @param value A scalar value to be used if the mode chosen is `constant`
 *              (by default it is 0).
 * @param alpha N/A (only support 1.0)
 * @param beta N/A (only support 0.0)
 */
topsopStatus_t TOPSOP_EXPORT topsopPad(topsopTensorHandle_t output,
                                       const topsopTensorHandle_t input,
                                       const topsopSize_t pads,
                                       const int64_t mode,
                                       const topsopScalar_t value,
                                       const topsopScalar_t alpha,
                                       const topsopScalar_t beta,
                                       topsStream_t stream);

/**
 * @brief check whether current pad operator support or not
 *
 * @param input input tensor
 * @param pads pads should be a 1D array of shape [2 * rank] or [3 * rank].
 *             pads format should be:
 *                [x1_begin, x2_begin, ..., x1_end, x2_end,...]
 *             or
 *                [x1_begin, ..., x1_end, ..., x1_mid,...]
 *             if mode chosen not is `constant`, x1_mid cannot take effect.
 * @param mode mode = 0 (constant pad), mode = 1(reflect pad), mode = 2(edge
 * pad)
 * @param value A scalar value to be used if the mode chosen is `constant`
 *              (by default it is 0).
 * @param alpha N/A (only support 1.0)
 * @param beta N/A (only support 0.0)
 * @return bool
 */
bool TOPSOP_EXPORT topsopPadIsSupported(const topsopTensorHandle_t input,
                                        const topsopSize_t pads,
                                        const int64_t mode,
                                        const topsopScalar_t alpha,
                                        const topsopScalar_t beta);

/**
 * @brief get output dims of current tensor of pad operator
 *
 * @param input input tensor
 * @param pads pads should be a 1D array of shape [2 * rank] or [3 * rank].
 *             pads format should be:
 *                [x1_begin, x2_begin, ..., x1_end, x2_end,...]
 *             or
 *                [x1_begin, ..., x1_end, ..., x1_mid,...]
 *             if mode chosen not is `constant`, x1_mid cannot take effect.
 * @param mode mode = 0 (constant pad), mode = 1(reflect pad), mode = 2(edge
 * pad)
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopPadGetOutputDim(const topsopTensorHandle_t input,
                      const topsopSize_t pads,
                      const int64_t mode,
                      int64_t *dims,
                      int64_t *rank);

/**
 * @brief This function performs the topk function.
 *
 * @param output_value output value tensor
 * @param output_index output index tensor
 * @param input input tensort
 * @param k The K value is used to take the largest first k value
 *          or the smallest first k value, k should be less than
 *          or equal to dim_input[axis].
 * @param axis Take the dimensions of the first k values.
 * @param is_sorted Whether the results are sorted or not.
 * @param cmp_mod  The select of topk computation(
 *                 The first k values are the largest:TOPSOP_TOPK_TYPE_MAX,
 *                 The first k of the smallest values:TOPSOP_TOPK_TYPE_MIN
 * @param stable_mod The select of stable sort option: whether it is a stable
 * sort (TOPSOP_SORT_STABLE TOPSOP_SORT_INSTABLE)
 * @param alpha Scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor.
 * @param beta Scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor.
 * @param stream Tops stream.
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopTopk(topsopTensorHandle_t output_value,
                                        topsopTensorHandle_t output_index,
                                        const topsopTensorHandle_t input,
                                        int64_t k,
                                        int64_t axis,
                                        bool is_sorted,
                                        topsopTopkCmpMode_t cmp_mod,
                                        topsopSortStableMode_t stable_mod,
                                        topsopScalar_t alpha,
                                        topsopScalar_t beta,
                                        topsStream_t stream);

/**
 * @brief check whether topsopTopk func
 * is support or not
 * @param input Tensor to be taked the first k values.
 * @param k The K value is used to take the largest first k value
 *          or the smallest first k value, k should be less than
 *          or equal to dim_input[axis].
 * @param axis Take the dimensions of the first k values, axis should be less
 *             than or equal to the rank of input.
 * @param is_sorted Whether the results are sorted or not
 * @param cmp_mod The select of topk computation(
 *                The first k values are the largest:TOPSOP_TOPK_TYPE_MAX,
 *                The first k of the smallest values:TOPSOP_TOPK_TYPE_MIN
 * @param stable_mod The select of stable sort option: whether it is a stable
 * sort (TOPSOP_SORT_STABLE TOPSOP_SORT_INSTABLE)
 * @param alpha Scaling factors (in host memory) used to blend the
 *              layer output value with prior value in the destination tensor.
 * @param beta Scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor.
 * @return bool
 */
bool TOPSOP_EXPORT
topsopTopkIsSupported(const topsopTensorHandle_t input,
                      const int64_t k,
                      const int64_t axis,
                      bool is_sorted,
                      const topsopTopkCmpMode_t cmp_mod,
                      const topsopSortStableMode_t stable_mod,
                      topsopScalar_t alpha,
                      topsopScalar_t beta);
/**
 * @brief get output dims of current tensor of topk operator
 * @param input Tensor to be taked the first k values.
 * @param k The K value is used to take the largest first k value
 *          or the smallest first k value, k should be less than
 *          or equal to dim_input[axis].
 * @param axis Take the dimensions of the first k values, axis should be
 *             less than or equal to the rank of input.
 * @param dims The dims of output tensor
 * @param rank The rank of output tensor
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopTopkGetOutputDim(const topsopTensorHandle_t input,
                       const int64_t k,
                       const int64_t axis,
                       int64_t *dims,
                       int64_t *rank);

/**
 * @brief This function performs the forward batch normalization
 * layer computation for the inference phase.
 *
 * @param out output tensor
 * @param input input tensor
 * @param bnScale fp32/fp64 pointers in device memory for the batch
 * normalization scale parameters
 * @param bnBias fp32/fp64 pointers in device memory for the batch normalization
 * bias parameters
 * @param eMean fp32/fp64 pointers in device memory of resultRunningMean in
 * backward training phase
 * @param eVar fp32/fp64 pointers in device memory of resultRunningVariance in
 * backward training phase
 * @param bn_mode configure for norm mode candidate:
 * TOPSOP_BATCHNORM_PER_ACTIVATION, TOPSOP_BATCHNORM_SPATIAL &
 * TOPSOP_BATCHNORM_SPATIAL_PERSISTENT
 * @param epsilon  Epsilon value used in the batch normalization formula.
 * Its value should be greater than 0.0f
 * @param alpha Pointers to scaling factors (in host memory) used to
 * blend the layer output value with prior value in the destination tensor
 * @param beta Pointers to scaling factors (in host memory) used to
 * blend the layer output value with prior value in the destination tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopBatchNormalizationInference(topsopTensorHandle_t out,
                                  const topsopTensorHandle_t input,
                                  const topsopTensorHandle_t bnScale,
                                  const topsopTensorHandle_t bnBias,
                                  const topsopTensorHandle_t eMean,
                                  const topsopTensorHandle_t eVar,
                                  const topsopBatchNormMode_t bn_mode,
                                  topsopScalar_t epsilon,
                                  topsopScalar_t alpha,
                                  topsopScalar_t beta,
                                  topsStream_t stream);
/**
 * @brief check whether topsopBatchNormalizationInference func
 * is support or not
 * @param input input tensor
 * @param bnScale fp32/fp64 pointers in device memory for the batch
 * normalization scale parameters
 * @param bnBias fp32/fp64 pointers in device memory for the batch normalization
 * bias parameters
 * @param eMean fp32/fp64 pointers in device memory of resultRunningMean in
 * backward training phase
 * @param eVar fp32/fp64 pointers in device memory of resultRunningVariance in
 * backward training phase
 * @param bn_mode configuration for norm mode, candidate:
 * TOPSOP_BATCHNORM_PER_ACTIVATION,
 * TOPSOP_BATCHNORM_SPATIAL & TOPSOP_BATCHNORM_SPATIAL_PERSISTENT
 * @param alpha Pointers to scaling factors (in host memory) used to
 * blend the layer output value with prior value in the destination tensor
 * @param beta Pointers to scaling factors (in host memory) used to
 * blend the layer output value with prior value in the destination tensor
 * @return bool
 */
bool TOPSOP_EXPORT topsopBatchNormalizationInferenceIsSupported(
    const topsopTensorHandle_t input,
    const topsopTensorHandle_t bnScale,
    const topsopTensorHandle_t bnBias,
    const topsopTensorHandle_t eMean,
    const topsopTensorHandle_t eVar,
    const topsopBatchNormMode_t bn_mode,
    topsopScalar_t epsilon,
    topsopScalar_t alpha,
    topsopScalar_t beta);
/**
 * @brief get output dims of current tensor of
 * BatchNormalizationForwardInference operator
 * @param input input tensor
 * @param dims output shape dims
 * @param rank output shape rank
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopBatchNormalizationInferenceGetOutputDim(
    const topsopTensorHandle_t input, int64_t *dims, int64_t *rank);
/**
 * @brief get parameter dims according to input handle and bnmode
 * @param input input tensor
 * @param bn_mode configuration for norm mode, candidate:
 * TOPSOP_BATCHNORM_PER_ACTIVATION,
 * TOPSOP_BATCHNORM_SPATIAL & TOPSOP_BATCHNORM_SPATIAL_PERSISTENT
 * @param dims outparameterput shape dims
 * @param rank parameter shape rank
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopBatchNormalizationGetParamDim(const topsopTensorHandle_t input,
                                    const topsopBatchNormMode_t bn_mode,
                                    int64_t *dims,
                                    int64_t *rank);

/**
 * @brief This function performs the forward batch normalization
 * layer computation for the training phase.
 * @param out output tensor
 * @param resultSaveMean fp32/fp64 output pointers in device memory of save mean
 * for local normalize
 * @param resultSaveInvVariance fp32/fp64 outputpointers in device memory of
 * save inv-variance for local normalize
 * @param input input tensor
 * @param bnScale fp32/fp64 input pointers in device memory for the batch
 * normalization scale parameters
 * @param bnBias fp32/fp64 input pointers in device memory for the batch
 * normalization bias parameters
 * @param resultRunningMean fp32/fp64 input/output pointers in device memory of
 * resultRunningMean for global inference
 * @param resultRunningVariance fp32/fp64 input/output pointers in device memory
 * of resultRunningVariance for global inference
 * @param bn_mode configure for norm mode candidate:
 * TOPSOP_BATCHNORM_PER_ACTIVATION, TOPSOP_BATCHNORM_SPATIAL &
 * TOPSOP_BATCHNORM_SPATIAL_PERSISTENT
 * @param exponentialAverageFactor weight for CMA calculation of
 * resultRunningMean and resultRunningVariance
 * @param epsilon  Epsilon value used in the batch normalization formula.
 * Its value should be greater than 0.0f
 * @param alpha Pointers to scaling factors (in host memory) used to
 * blend the layer output value with prior value in the destination tensor
 * @param beta Pointers to scaling factors (in host memory) used to
 * blend the layer output value with prior value in the destination tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */

topsopStatus_t TOPSOP_EXPORT
topsopBatchNormalizationTraining(topsopTensorHandle_t out,
                                 topsopTensorHandle_t resultSaveMean,
                                 topsopTensorHandle_t resultSaveInvVariance,
                                 const topsopTensorHandle_t input,
                                 const topsopTensorHandle_t bnScale,
                                 const topsopTensorHandle_t bnBias,
                                 topsopTensorHandle_t resultRunningMean,
                                 topsopTensorHandle_t resultRunningVariance,
                                 const topsopBatchNormMode_t bn_mode,
                                 topsopScalar_t exponentialAverageFactor,
                                 topsopScalar_t epsilon,
                                 topsopScalar_t alpha,
                                 topsopScalar_t beta,
                                 topsStream_t stream);
/**
 * @brief check whether topsopBatchNormalizationInference func
 * is support or not
 * @param input input tensor
 * @param bnScale fp32/fp64 input pointers in device memory for the batch
 * normalization scale parameters
 * @param bnBias fp32/fp64 input pointers in device memory for the batch
 * normalization bias parameters
 * @param resultRunningMean fp32/fp64 input/output pointers in device memory of
 * resultRunningMean for global inference
 * @param resultRunningVariance fp32/fp64 input/output pointers in device memory
 * of resultRunningVariance for global inference
 * @param bn_mode configure for norm mode candidate:
 * TOPSOP_BATCHNORM_PER_ACTIVATION, TOPSOP_BATCHNORM_SPATIAL &
 * TOPSOP_BATCHNORM_SPATIAL_PERSISTENT
 * @param exponentialAverageFactor weight for CMA calculation of
 * resultRunningMean and resultRunningVariance
 * @param epsilon  Epsilon value used in the batch normalization formula.
 * Its value should be greater than 0.0f
 * @param alpha Pointers to scaling factors (in host memory) used to
 * blend the layer output value with prior value in the destination tensor
 * @param beta Pointers to scaling factors (in host memory) used to
 * blend the layer output value with prior value in the destination tensor
 * @return bool
 */

bool TOPSOP_EXPORT topsopBatchNormalizationTrainingIsSupported(
    const topsopTensorHandle_t input,
    const topsopTensorHandle_t bnScale,
    const topsopTensorHandle_t bnBias,
    topsopTensorHandle_t resultRunningMean,
    topsopTensorHandle_t resultRunningVariance,
    const topsopBatchNormMode_t bn_mode,
    topsopScalar_t exponentialAverageFactor,
    topsopScalar_t epsilon,
    topsopScalar_t alpha,
    topsopScalar_t beta);
/**
 * @brief get output dims of current tensor of
 * topsopBatchNormalizationTraining operator
 * @param input input tensor
 * @param bnScale fp32/fp64 input pointers in device memory for the batch
 * normalization scale parameters
 * @param bn_mode configure for norm mode candidate:
 * TOPSOP_BATCHNORM_PER_ACTIVATION,
 * TOPSOP_BATCHNORM_SPATIAL & TOPSOP_BATCHNORM_SPATIAL_PERSISTENT
 * @param dimsOutput output shape dims
 * @param rankOutput output shape rank
 * @param dimsResultSaveMean ResultSaveMean shape dims
 * @param rankResultSaveMean ResultSaveMean shape rank
 * @param dimsresultSaveInvVariance SaveInvVariance shape dims
 * @param rankresultSaveInvVariance SaveInvVariance shape rank
 * @return topsopStatus_t
 */

topsopStatus_t TOPSOP_EXPORT topsopBatchNormalizationTrainingGetOutputDim(
    const topsopTensorHandle_t input,
    const topsopTensorHandle_t bnScale,
    const topsopBatchNormMode_t bn_mode,
    int64_t *dimsOutput,
    int64_t *rankOutput,
    int64_t *dimsResultSaveMean,
    int64_t *rankResultSaveMean,
    int64_t *dimsresultSaveInvVariance,
    int64_t *rankresultSaveInvVariance);

/**
 * @brief This function performs tthe forward batch normalization layer
 * computation for the training phase..
 * @param output_dx Output. Output tensor data pointer in device memory for the
 * resulting differential output with respect to input_x.
 * @param resultBnScaleDiff fp32/fp64 output, Pointers in device memory for the
 * resulting scale differentials computed by this routine
 * @param resultBnBiasDiff fp32/fp64 output, Pointers in device memory for the
 * resulting bias differentials computed by this routine
 * @param input_x Input. Input tensor data pointer in device memory.
 * @param input_dy Input. Input tensor data pointer in device memory for the
 * backpropagated differential dy input.
 * @param bnScale fp32/fp64 Input, Pointers in device memory for the batch
 * normalization scale  parameters.
 * @param SaveMean fp32/fp64 Input, Optional cache containing saved intermediate
 * results that were computed during the forward pass.
 * @param SaveInvVariance fp32/fp64 Input, Optional cache containing saved
 * intermediate results that were computed during the forward pass.
 * @param bn_mode configure for norm mode candidate:
 * TOPSOP_BATCHNORM_PER_ACTIVATION, TOPSOP_BATCHNORM_SPATIAL &
 * TOPSOP_BATCHNORM_SPATIAL_PERSISTENT
 * @param epsilon  Epsilon value used in the batch normalization formula.
 * Its value should be greater than 0.0f
 * @param alphaDataDiff Input. Pointers to scaling factors (in host memory) used
 * to blend the output_dx with prior value in the destination tensor
 * @param betaDataDiff Input. Pointers to scaling factors (in host memory) used
 * to blend the output_dx with prior value in the destination tensor
 * @param alphaParamDiff Input. Pointers to scaling factors (in host memory)
 * used to blend the resultBnScaleDiff/resultBnBiasDiff with prior value in the
 *              destination tensor
 * @param betaParamDiff Input. Pointers to scaling factors (in host memory) used
 * to blend the resultBnScaleDiff/resultBnBiasDiff with prior value in the
 *             destination tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */

topsopStatus_t TOPSOP_EXPORT
topsopBatchNormalizationBackward(topsopTensorHandle_t output_dx,
                                 topsopTensorHandle_t resultBnScaleDiff,
                                 topsopTensorHandle_t resultBnBiasDiff,
                                 const topsopTensorHandle_t input_x,
                                 const topsopTensorHandle_t input_dy,
                                 const topsopTensorHandle_t bnScale,
                                 const topsopTensorHandle_t SaveMean,
                                 const topsopTensorHandle_t SaveInvVariance,
                                 const topsopBatchNormMode_t bn_mode,
                                 topsopScalar_t epsilon,
                                 topsopScalar_t alphaDataDiff,
                                 topsopScalar_t betaDataDiff,
                                 topsopScalar_t alphaParamDiff,
                                 topsopScalar_t betaParamDiff,
                                 topsStream_t stream);
/**
 * @brief check whether topsopBatchNormalizationInference func
 * is support or not
 * @param input_x Input. Input tensor data pointer in device memory.
 * @param input_dy Input. Input tensor data pointer in device memory for the
 * backpropagated differential dy input.
 * @param bnScale fp32/fp64 Input, Pointers in device memory for the batch
 * normalization scale  parameters.
 * @param SaveMean fp32/fp64 Input, Optional cache containing saved intermediate
 * results that were computed during the forward pass.
 * @param SaveInvVariance fp32/fp64 Input, Optional cache containing saved
 * intermediate results that were computed during the forward pass.
 * @param bn_mode configure for norm mode candidate:
 * TOPSOP_BATCHNORM_PER_ACTIVATION, TOPSOP_BATCHNORM_SPATIAL &
 * TOPSOP_BATCHNORM_SPATIAL_PERSISTENT
 * @param epsilon  Epsilon value used in the batch normalization formula.
 * Its value should be greater than 0.0f
 * @param alphaDataDiff Input. Pointers to scaling factors (in host memory) used
 * to blend the output_dx with prior value in the destination tensor
 * @param betaDataDiff Input. Pointers to scaling factors (in host memory) used
 * to blend the output_dx with prior value in the destination tensor
 * @param alphaParamDiff Input. Pointers to scaling factors (in host memory)
 * used to blend the resultBnScaleDiff/resultBnBiasDiff with prior value in the
 *              destination tensor
 * @param betaParamDiff Input. Pointers to scaling factors (in host memory) used
 * to blend the resultBnScaleDiff/resultBnBiasDiff with prior value in the
 *             destination tensor
 * @return bool
 */

bool TOPSOP_EXPORT topsopBatchNormalizationBackwardIsSupported(
    const topsopTensorHandle_t input_x,
    const topsopTensorHandle_t input_dy,
    const topsopTensorHandle_t bnScale,
    const topsopTensorHandle_t SaveMean,
    const topsopTensorHandle_t SaveInvVariance,
    const topsopBatchNormMode_t bn_mode,
    topsopScalar_t epsilon,
    topsopScalar_t alphaDataDiff,
    topsopScalar_t betaDataDiff,
    topsopScalar_t alphaParamDiff,
    topsopScalar_t betaParamDiff);
/**
 * @brief get output dims of current tensor of
 * topsopBatchNormalizationBackward operator
 * @param input_x input tensor
 * @param bnScale fp32/fp64 Input, Pointers in device memory for the batch
 * @param bn_mode configure for norm mode candidate:
 * TOPSOP_BATCHNORM_PER_ACTIVATION,
 * TOPSOP_BATCHNORM_SPATIAL & TOPSOP_BATCHNORM_SPATIAL_PERSISTENT
 * @param dimsOutput output shape dims
 * @param rankOutput output shape rank
 * @param dimsResultBnScaleDiff resultBnScaleDiff shape dims
 * @param rankResultBnScaleDiff resultBnScaleDiff shape rank
 * @param dimsResultBnBiasDiff resultBnBiasDiff shape dims
 * @param rankResultBnBiasDiff resultBnBiasDiff shape rank
 * @return topsopStatus_t
 */

topsopStatus_t TOPSOP_EXPORT topsopBatchNormalizationBackwardGetOutputDim(
    const topsopTensorHandle_t input_x,
    const topsopTensorHandle_t bnScale,
    const topsopBatchNormMode_t bn_mode,
    int64_t *dimsOutput,
    int64_t *rankOutput,
    int64_t *dimsResultBnScaleDiff,
    int64_t *rankResultBnScaleDiff,
    int64_t *dimsResultBnBiasDiff,
    int64_t *rankResultBnBiasDiff);
/**
 * @brief select_and_scatter operator, the backward function for a max pooling
 *
 * @param output the output tensor
 * @param operand the input tensor over which the windows slide
 * @param source the 2nd input tensor with the values to scatter
 * @param initValue a scalar used to initialize the output tensor
 * @param select a binary comparison computation, used to select an element from
 *               each window by applying it across each window
 * @param scatter a binary computation, used to apply each scatter source
 *                element with its destination element
 * @param window window dimension values, size is the rank of operand
 * @param stride window stride values, size is the rank of operand
 * @param Padding operand padding values, indicates the padding value for each
 *                dimensions' high and low, size is the rank of operand
 *                multiplied by 2
 * @param alpha The multiplier for source tensor
 * @param beta The multiplier for output tensor
 * @param stream tops stream
 * @return topsopStatus_t
 */
topsopStatus_t topsopSelectAndScatter(topsopTensorHandle_t output,
                                      const topsopTensorHandle_t operand,
                                      const topsopTensorHandle_t source,
                                      const topsopScalar_t initValue,
                                      const topsopSelectType_t select,
                                      const topsopScatterType_t scatter,
                                      topsopSize_t window,
                                      topsopSize_t stride,
                                      topsopSize_t padding,
                                      topsopScalar_t alpha,
                                      topsopScalar_t beta,
                                      topsStream_t stream);
/**
 * @brief check whether current select_and_scatter operator support or not
 *
 * @param operand the input tensor over which the windows slide
 * @param source the 2nd input tensor with the values to scatter
 * @param initValue a scalar used to initialize the output tensor
 * @param select a binary comparison computation, used to select an element from
 *               each window by applying it across each window
 * @param scatter a binary computation, used to apply each scatter source
 *                element with its destination element
 * @param window window dimension values, size is the rank of operand
 * @param stride window stride values, size is the rank of operand
 * @param Padding operand padding values, indicates the padding value for each
 *                dimensions' high and low, size is the rank of operand
 *                multiplied by 2
 * @param alpha The multiplier for source tensor
 * @param beta The multiplier for output tensor
 * @return bool
 */
bool topsopSelectAndScatterIsSupported(const topsopTensorHandle_t operand,
                                       const topsopTensorHandle_t source,
                                       const topsopScalar_t initValue,
                                       const topsopSelectType_t select,
                                       const topsopScatterType_t scatter,
                                       topsopSize_t windowDimenssions,
                                       topsopSize_t windowStrides,
                                       topsopSize_t padding,
                                       topsopScalar_t alpha,
                                       topsopScalar_t beta);

/**
 * @brief get output dims of current tensor select_and_scatter operator
 *
 * @param operand Input operand tensor
 * @param dims The dims of output tensor
 * @param rank The rank of output tensor
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopSelectAndScatterGetOutputDim(
    const topsopTensorHandle_t operand, int64_t *dims, int64_t *rank);

/**
 * @brief RoiAlign operator.
 * @param output output operand tensor
 * @param input Input operand tensor
 * @param rois 2D input of shape (N, 4) specifying R RoIs
 * @param batch_indices batch indices corresponding to each set of ROI, only
 * support int32
 * @param output_height Pooled output output's height
 * @param output_width Pooled output output's width
 * @param sampling_ratio number of sampling points in the interpolation grid
 * @param mode avg or max mode
 * @param coordinate_transformation_mode half_pixel or non half_pixel mode
 * @param spatial_scale Spatial scale of the input feature map
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopRoiAlign(
    topsopTensorHandle_t output,
    const topsopTensorHandle_t input,
    const topsopTensorHandle_t rois,
    const topsopTensorHandle_t batch_indices,
    const int64_t output_height,
    const int64_t output_width,
    const int64_t sampling_ratio,
    const topsopRoiAlignMode_t mode,
    const topsopRoiAlignCoordinateTransformationMode_t transformation_mode,
    const float spatial_scale,
    topsStream_t stream);

/**
 * @brief RoiAlign operator.
 * @param input Input operand tensor
 * @param rois 2D input of shape (N, 4) specifying R RoIs
 * @param batch_indices batch indices corresponding to each set of ROI, only
 * support int32
 * @param sampling_ratio number of sampling points in the interpolation grid
 * @param mode avg or max mode
 * @param coordinate_transformation_mode half_pixel or non half_pixel mode
 * @return bool
 */
bool topsopRoiAlignIsSupported(
    const topsopTensorHandle_t input,
    const topsopTensorHandle_t rois,
    const topsopTensorHandle_t batch_indices,
    const int64_t sampling_ratio,
    const topsopRoiAlignMode_t mode,
    const topsopRoiAlignCoordinateTransformationMode_t transformation_mode);

/**
 * @brief RoiAlign operator.
 * @param input Input operand tensor
 * @param rois 2D input of shape (N, 4) specifying R RoIs
 * @param output_height Pooled output output's height
 * @param output_width Pooled output output's width
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopRoiAlignGetOutputDim(const topsopTensorHandle_t input,
                           const topsopTensorHandle_t rois,
                           int64_t output_height,
                           int64_t output_width,
                           int64_t *dims,
                           int64_t *rank);

/**
 * @brief RoiAlign operator.
 * @param output output operand tensor
 * @param input Input operand tensor
 * @param rois 2D input of shape (N, 4) specifying R RoIs
 * @param batch_indices batch indices corresponding to each set of ROI, only
 * support int32
 * @param pooled_height grad input h
 * @param pooled_width grad input w
 * @param batch_size output h
 * @param channels output c
 * @param height output w
 * @param width  output h
 * @param sampling_ratio number of sampling points in the interpolation grid
 * @param spatial_scale Spatial scale of the input feature map
 * @param aligned Pixel shift flag
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopRoiAlignBackward(topsopTensorHandle_t output,
                       const topsopTensorHandle_t input,
                       const topsopTensorHandle_t rois,
                       const topsopTensorHandle_t batch_indices,
                       const int64_t pooled_height,
                       const int64_t pooled_width,
                       const int64_t batch_size,
                       const int64_t channels,
                       const int64_t height,
                       const int64_t width,
                       const int64_t sampling_ratio,
                       const float spatial_scale,
                       const bool aligned,
                       topsStream_t stream);

/**
 * @brief RoiAlign operator.
 * @param input Input operand tensor
 * @param rois 2D input of shape (N, 4) specifying R RoIs
 * @param batch_indices batch indices corresponding to each set of ROI, only
 * support int32
 * @param pooled_height grad input h
 * @param pooled_width grad input w
 * @param batch_size output h
 * @param channels output c
 * @param height output w
 * @param width  output h
 * @param sampling_ratio number of sampling points in the interpolation grid
 * @param spatial_scale Spatial scale of the input feature map
 * @return bool
 */
bool topsopRoiAlignBackwardIsSupported(const topsopTensorHandle_t input,
                                       const topsopTensorHandle_t rois,
                                       const topsopTensorHandle_t batch_indices,
                                       const int64_t pooled_height,
                                       const int64_t pooled_width,
                                       const int64_t batch_size,
                                       const int64_t channels,
                                       const int64_t height,
                                       const int64_t width,
                                       const int64_t sampling_ratio,
                                       const float spatial_scale);

/**
 * @brief RoiAlign operator.
 * @param batch_size output h
 * @param channels output c
 * @param height output w
 * @param width  output h
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopRoiAlignBackwardGetOutputDim(const int64_t batch_size,
                                   const int64_t channels,
                                   const int64_t height,
                                   const int64_t width,
                                   int64_t *dims,
                                   int64_t *rank);

/**
 * @brief dynamicslice operator.
 *
 * @param output Result tensor
 * @param input Input tensor,
 *  the max dim value should be less than (0x7fffffff/bpe)
 * @param start_indices List of N scalar integers containing
 *  the starting indices of the slice for each dimension
 * @param size_indices List of N integers containing
 *  the slice size for each dimension
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopDynamicSlice(topsopTensorHandle_t output,
                   const topsopTensorHandle_t input,
                   const topsopTensorHandle_t *start_indices,
                   topsopSize_t size_indices,
                   topsStream_t stream);

/**
 * @brief check whether current dynamicslice operator support or not
 *
 * @param output Result tensor
 * @param input Input tensor
 * @param size_indices List of N integers containing
 * the slice size for each dimension
 * @return bool
 */
bool topsopDynamicSliceIsSupported(topsopTensorHandle_t output,
                                   const topsopTensorHandle_t input,
                                   topsopSize_t size_indices);

/**
 * @brief get output dims of current tensor dynamicslice operator
 *
 * @param size_indices List of N integers containing
 * the slice size for each dimension
 * @param dims The dims of output tensor
 * @param rank The rank of output tensor
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopDynamicSliceGetOutputDim(
    topsopSize_t size_indices, int64_t *dims, int64_t *rank);

/**
 * @brief dynamicupdateslice operator.
 *
 * @param output Result tensor
 * @param input Input tensor
 *  the max dim value should be less than (0x7fffffff/bpe)
 * @param update N dimensional array of type T containing the slice update
 * @param start_indices List of N scalar integers containing
 *  the starting indices of the slice for each dimension
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopDynamicUpdateSlice(topsopTensorHandle_t output,
                         const topsopTensorHandle_t input,
                         const topsopTensorHandle_t update,
                         const topsopTensorHandle_t *start_indices,
                         topsStream_t stream);

/**
 * @brief check whether current dynamicupdateslice operator support or not
 *
 * @param output Result tensor
 * @param input Input tensor
 * @param update N dimensional array of type T containing the slice update
 * @return bool
 */
bool topsopDynamicUpdateSliceIsSupported(topsopTensorHandle_t output,
                                         const topsopTensorHandle_t input,
                                         const topsopTensorHandle_t update);

/**
 * @brief get output dims of current tensor dynamicupdateslice operator
 *
 * @param input Input tensor
 * @param dims The dims of output tensor
 * @param rank The rank of output tensor
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopDynamicUpdateSliceGetOutputDim(
    const topsopTensorHandle_t input, int64_t *dims, int64_t *rank);

/**
 * @brief Returns a tensor filled with the scalar value 0.
 *
 * @param output Result tensor
 * @param size output shape.
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopZeros(topsopTensorHandle_t output,
                                         const topsopSize_t size,
                                         topsStream_t stream);

/**
 * @brief check whether current topsopZeros operator support or not
 *
 * @param output Result tensor
 * @param size output shape.
 * @return bool
 */
bool TOPSOP_EXPORT topsopZerosIsSupported(topsopTensorHandle_t output,
                                          topsopSize_t size);

/**
 * @brief get output dims of current tensor zeros operator
 *
 * @param size output shape.
 * @param dims The dims of output tensor
 * @param rank The rank of output tensor
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopZerosGetOutputDim(topsopSize_t size,
                                                     int64_t *dims,
                                                     int64_t *rank);

/**
 * @brief This function performs the rgb2gray function
 *
 * @param out output tensor
 * @param input input tensor
 * @param alpha N/A (only support 1.0)
 * @param beta N/A (only support 0.0)
 * @param stream tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopRgb2gray(topsopTensorHandle_t out,
                                            const topsopTensorHandle_t input,
                                            const topsopScalar_t alpha,
                                            const topsopScalar_t beta,
                                            topsStream_t stream);

/**
 * @brief check whether current rgb2gray operator support or not
 *
 * @param output output tensor
 * @param input input tensor
 * @param alpha N/A (only support 1.0)
 * @param beta N/A (only support 0.0)
 * @return true
 * @return false
 */
bool TOPSOP_EXPORT topsopRgb2grayIsSupported(const topsopTensorHandle_t output,
                                             const topsopTensorHandle_t input,
                                             const topsopScalar_t alpha,
                                             const topsopScalar_t beta);

/**
 * @brief get output dims of current tensor of rgb2gray operator
 *
 * @param input input tensor
 * @param dims shape
 * @param rank rank
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopRgb2grayGetOutputDim(
    const topsopTensorHandle_t input, int64_t *dims, int64_t *rank);

/**
 * @brief This function performs the slice(onnx semantics) function.
 * Slicing extracts a sub-array from the input array.
 * The sub-array is of the same rank as the input
 * and contains the values inside a bounding box within the input array
 * where the dimensions and indices of the bounding box are given as
 * arguments to the slice operation.
 * @param output output tensor
 * @param input input tensor
 * @param starts List of N integers containing the starting indices of the slice
 * for dimension.
 * @param ends List of N integers containing the ending indices (exclusive) for
 * the slice for dimension.
 * @param axes List of N integers containing the axes that `starts` and `ends`
 * apply to.
 * @param steps List of N integers that decides the input stride of the slice.
 * @param alpha N/A (only support 1.0)
 * @param beta N/A (only support 0.0)
 * @param stream  tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopSlice(topsopTensorHandle_t output,
                                         const topsopTensorHandle_t input,
                                         const topsopSize_t starts,
                                         const topsopSize_t ends,
                                         const topsopSize_t axes,
                                         const topsopSize_t steps,
                                         const topsopScalar_t alpha,
                                         const topsopScalar_t beta,
                                         topsStream_t stream);

/**
 * @brief check whether current slice operator support or not
 *
 * @param input input tensor
 * @param starts List of N integers containing the starting indices of the slice
 * for dimension.
 * @param ends List of N integers containing the ending indices (exclusive) for
 * the slice for dimension.
 * @param axes List of N integers containing the axes that `starts` and `ends`
 * apply to.
 * @param steps List of N integers that decides the input stride of the slice.
 * @param alpha N/A (only support 1.0)
 * @param beta N/A (only support 0.0)
 * @return bool
 */
bool TOPSOP_EXPORT topsopSliceIsSupported(const topsopTensorHandle_t input,
                                          const topsopSize_t starts,
                                          const topsopSize_t ends,
                                          const topsopSize_t axes,
                                          const topsopSize_t steps,
                                          const topsopScalar_t alpha,
                                          const topsopScalar_t beta);

/**
 * @brief get output dims of current tensor of slice operator
 *
 * @param input input tensor
 * @param starts List of N integers containing the starting indices of the slice
 * for dimension.
 * @param ends List of N integers containing the ending indices (exclusive) for
 * the slice for dimension.
 * @param axes List of N integers containing the axes that `starts` and `ends`
 * apply to.
 * @param steps List of N integers that decides the input stride of the slice.
 * @param alpha N/A (only support 1.0)
 * @param beta N/A (only support 0.0)
 * @param dims output tensor shape
 * @param rank output tensor rank
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopSliceGetOutputDim(const topsopTensorHandle_t input,
                        const topsopSize_t starts,
                        const topsopSize_t ends,
                        const topsopSize_t axes,
                        const topsopSize_t steps,
                        int64_t *dims,
                        int64_t *rank);

/**
 * @brief This function performs the transpose(onnx semantics) function.
 *  Transpose the input tensor similar to numpy.transpose.
 *  For example, when perm=(1, 0, 2), given an input tensor of shape (1, 2, 3),
 *  the output shape will be (2, 1, 3).
 * @param output output tensor
 * @param input input tensor
 * @param permutation A list of integers. By default, reverse the dimensions,
 * otherwise permute the axes according to the values given.
 * @param alpha N/A (only support 1.0)
 * @param beta N/A (only support 0.0)
 * @param stream tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopTranspose(topsopTensorHandle_t output,
                                             const topsopTensorHandle_t input,
                                             const topsopSize_t permutation,
                                             const topsopScalar_t alpha,
                                             const topsopScalar_t beta,
                                             topsStream_t stream);

/**
 * @brief check whether current slice operator support or not
 *
 * @param input input tensor
 * @param permutation A list of integers. By default, reverse the dimensions,
 * otherwise permute the axes according to the values given.
 * @param alpha N/A (only support 1.0)
 * @param beta N/A (only support 0.0)
 * @return bool
 */
bool TOPSOP_EXPORT topsopTransposeIsSupported(const topsopTensorHandle_t input,
                                              const topsopSize_t permutation,
                                              const topsopScalar_t alpha,
                                              const topsopScalar_t beta);

/**
 * @brief get output dims of current tensor of slice operator
 *
 * @param input input tensor
 * @param permutation A list of integers. By default, reverse the dimensions,
 * otherwise permute the axes according to the values given.
 * @param dims output tensor shape
 * @param rank output tensor rank
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopTransposeGetOutputDim(const topsopTensorHandle_t input,
                            const topsopSize_t permutation,
                            int64_t *dims,
                            int64_t *rank);

/**
 * @brief This function performs the sort function.
 *
 * @param output output value tensor array.
 * @param input input value tensor array.
 * @param num input tensor number.
 * @param axis Dimension to be sorted.
 * @param is_sorted Results sorted or not.
 * @param cmp_mod The select of sort computation(Ascending
 * sort:TOPSOP_SORT_TYPE_ASCEND, Descending sort: TOPSOP_SORT_TYPE_DESCEND)
 * @param stable_mod Whether it is a stable sort(stable mode: TOPSOP_SORT_STABLE
 *                   unstable sort: TOPSOP_SORT_INSTABLE)
 * @param alpha Scaling factors (in host memory) used to blend the
 *              layer output value with prior value in the destination tensor.
 * @param beta Scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor.
 * @param stream Tops stream.
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopSort(topsopTensorHandle_t *output,
                                        const topsopTensorHandle_t *input,
                                        int64_t num,
                                        int64_t axis,
                                        bool is_sorted,
                                        topsopSortCmpMode_t cmp_mod,
                                        topsopSortStableMode_t stable_mod,
                                        topsopScalar_t alpha,
                                        topsopScalar_t beta,
                                        topsStream_t stream);

/**
 * @brief This function performs the one input  sort function.
 *
 * @param output_value output value tensor.
 * @param output_index output index tensor.
 * @param input input value tensor.
 * @param axis Dimension to be sorted.
 * @param is_sorted Results sorted or not.
 * @param cmp_mod The select of sort computation(Ascending
 * sort:TOPSOP_SORT_TYPE_ASCEND, Descending sort: TOPSOP_SORT_TYPE_DESCEND)
 * @param stable_mod Whether it is a stable sort(stable mode: TOPSOP_SORT_STABLE
 *                   unstable sort: TOPSOP_SORT_INSTABLE)
 * @param alpha Scaling factors (in host memory) used to blend the
 *              layer output value with prior value in the destination tensor.
 * @param beta Scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor.
 * @param stream Tops stream.
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopSortEx(topsopTensorHandle_t output_value,
                                          topsopTensorHandle_t output_index,
                                          const topsopTensorHandle_t input,
                                          int64_t axis,
                                          bool is_sorted,
                                          topsopSortCmpMode_t cmp_mod,
                                          topsopSortStableMode_t stable_mod,
                                          topsopScalar_t alpha,
                                          topsopScalar_t beta,
                                          topsStream_t stream);

/**
 * @brief check whether topsopSort func
 * is support or not
 * @param input input value tensor array.
 * @param num input tensor number.
 * @param axis Dimension to be sorted.
 * @param is_sorted Results sorted or not.
 * @param cmp_mod The select of sort computation(Ascending
 * sort:TOPSOP_SORT_TYPE_ASCEND, Descending sort: TOPSOP_SORT_TYPE_DESCEND)
 * @param stable_mod Whether it is a stable sort(stable mode: TOPSOP_SORT_STABLE
 *                   unstable sort: TOPSOP_SORT_INSTABLE)
 * @param alpha Scaling factors (in host memory) used to blend the
 *              layer output value with prior value in the destination tensor.
 * @param beta Scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor.
 * @return bool
 */
bool TOPSOP_EXPORT
topsopSortIsSupported(topsopTensorHandle_t *input,
                      int64_t num,
                      const int64_t axis,
                      bool is_sorted,
                      const topsopSortCmpMode_t cmp_mod,
                      const topsopSortStableMode_t stable_mod,
                      const topsopScalar_t alpha,
                      const topsopScalar_t beta);
/**
 * @brief get output dims of current tensor of sort operator
 * @param input input value tensor.
 * @param dims The dims of output tensor
 * @param rank The rank of output tensor
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopSortGetOutputDim(
    const topsopTensorHandle_t input, int64_t *dims, int64_t *rank);

/**
 * @brief check whether topsopSort func
 * is support or not
 * @param input input value tensor array.
 * @param axis Dimension to be sorted.
 * @param is_sorted Results sorted or not.
 * @param cmp_mod The select of sort computation(Ascending
 * sort:TOPSOP_SORT_TYPE_ASCEND, Descending sort: TOPSOP_SORT_TYPE_DESCEND)
 * @param stable_mod Whether it is a stable sort(stable mode: TOPSOP_SORT_STABLE
 *                   unstable sort: TOPSOP_SORT_INSTABLE)
 * @param alpha Scaling factors (in host memory) used to blend the
 *              layer output value with prior value in the destination tensor.
 * @param beta Scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor.
 * @return bool
 */
bool TOPSOP_EXPORT
topsopSortExIsSupported(topsopTensorHandle_t *input,
                        const int64_t axis,
                        bool is_sorted,
                        const topsopSortCmpMode_t cmp_mod,
                        const topsopSortStableMode_t stable_mod,
                        const topsopScalar_t alpha,
                        const topsopScalar_t beta);
/**
 * @brief get output dims of current tensor of sort operator
 * @param input input value tensor.
 * @param dims The dims of output tensor
 * @param rank The rank of output tensor
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopSortExGetOutputDim(
    const topsopTensorHandle_t input, int64_t *dims, int64_t *rank);

/**
 * @brief This function performs the gather function
 *
 * @param output:Gather result
 * @param operand:The array we’re gathering from.
 * @param start_indices:Array containing the starting indices of the slices we
 gather.
 * @param offset_dims:The set of dimensions in the output shape that offset into
 an array sliced from operand.
 * @param slice_sizes:slice_sizes[i] is the bounds for the slice on dimension i.
 * @param collapsed_slice_dims:The set of dimensions in each slice that are
 collapsed away. These dimensions must have size 1.
 * @param start_index_map:A map that describes how to map indices in
 start_indices to legal indices into operand.
 * @param index_vector_dim:The dimension in start_indices that "contains" the
 starting indices.
 * @param indices_are_sorted:Whether the indices are guaranteed to be sorted by
 the caller.
 * @param unique_indices:Whether the indices are guaranteed to be unique by the
 caller.
 * @param stream:Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopGather(topsopTensorHandle_t output,
             const topsopTensorHandle_t operand,
             const topsopTensorHandle_t start_indices,
             topsopSize_t offset_dims,
             topsopSize_t slice_sizes,
             topsopSize_t collapsed_slice_dims,
             topsopSize_t start_index_map,
             int64_t index_vector_dim,
             bool indices_are_sorted,
             bool unique_indices,
             topsStream_t stream);

/**
 * @brief check whether topsopGather func is support or not
 *
 * @param operand:The array we’re gathering from.
 * @param start_indices:Array containing the starting indices of the slices we
 gather.
 * @param offset_dims:The set of dimensions in the output shape that offset into
 an array sliced from operand.
 * @param slice_sizes:slice_sizes[i] is the bounds for the slice on dimension i.
 * @param collapsed_slice_dims:The set of dimensions in each slice that are
 collapsed away. These dimensions must have size 1.
 * @param start_index_map:A map that describes how to map indices in
 start_indices to legal indices into operand.
 * @param index_vector_dim:The dimension in start_indices that "contains" the
 starting indices.
 * @param indices_are_sorted:Whether the indices are guaranteed to be sorted by
 the caller.
 * @param unique_indices:Whether the indices are guaranteed to be unique by the
 caller.
 * @return bool: bool flag to indicate whether support this case.
 */
bool TOPSOP_EXPORT
topsopGatherIsSupported(const topsopTensorHandle_t operand,
                        const topsopTensorHandle_t start_indices,
                        topsopSize_t offset_dims,
                        topsopSize_t slice_sizes,
                        topsopSize_t collapsed_slice_dims,
                        topsopSize_t start_index_map,
                        int64_t index_vector_dim,
                        bool indices_are_sorted,
                        bool unique_indices);
/**
 * @brief get output dims info for gather
 *
 * @param operand:The array we’re gathering from.
 * @param start_indices:Array containing the starting indices of the slices we
 gather.
 * @param offset_dims:The set of dimensions in the output shape that offset into
 an array sliced from operand.
 * @param slice_sizes:slice_sizes[i] is the bounds for the slice on dimension i.
 * @param collapsed_slice_dims:The set of dimensions in each slice that are
 collapsed away. These dimensions must have size 1.
 * @param start_index_map:A map that describes how to map indices in
 start_indices to legal indices into operand.
 * @param index_vector_dim:The dimension in start_indices that "contains" the
 starting indices.
 * @param indices_are_sorted:Whether the indices are guaranteed to be sorted by
 the caller.
 * @param unique_indices:Whether the indices are guaranteed to be unique by the
 caller.
 * @param dims:Pointer point to dimensions of output.
 * @param rank:Pointer point to rank of output.
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopGatherGetOutputDim(const topsopTensorHandle_t operand,
                         const topsopTensorHandle_t start_indices,
                         topsopSize_t offset_dims,
                         topsopSize_t slice_sizes,
                         topsopSize_t collapsed_slice_dims,
                         topsopSize_t start_index_map,
                         int64_t index_vector_dim,
                         bool indices_are_sorted,
                         bool unique_indices,
                         int64_t *dims,
                         int64_t *rank);

/**
 * @brief This function performs the scatter function
 *
 * @param output: Scatter result
 * @param operand: Array we’re scattering to.
 * @param indices: Array containing the starting indices of the slices that must
 * be scattered to.
 * @param updates: Array containing the values that must be used for scattering
 * operands.
 * @param computation: Computation to be used for combining the existing values
 *                     in the input array and the updates during scatter.
 * @param index_vector_dim: The dimension in indices that contains the starting
 * indices.
 * @param update_window_dims: The set of dimensions in updates shape that are
 * window dimensions.
 * @param inserted_window_dims: The set of window dimensions that must be
 * inserted into updates shape.
 * @param scatter_dims_to_operand_dims: A dimensions map from the scatter
 * indices to the operand index space.
 * @param indices_are_sorted: Whether the indices are guaranteed to be sorted by
 * the caller.
 * @param unique_indices: Whether the indices are guaranteed to be unique by the
 * caller.
 * @param stream:Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopScatter(topsopTensorHandle_t output,
              const topsopTensorHandle_t operand,
              const topsopTensorHandle_t indices,
              const topsopTensorHandle_t updates,
              const topsopScatterComputationType_t computation,
              const int64_t index_vector_dim,
              const topsopSize_t update_window_dims,
              const topsopSize_t inserted_window_dims,
              const topsopSize_t scatter_dims_to_operand_dims,
              const bool indices_are_sorted,
              const bool unique_indices,
              topsStream_t stream);

/**
 * @brief check whether topsopScatter func is supported or not
 *
 * @param operand: Array we’re scattering to.
 * @param indices: Array containing the starting indices of the slices that must
 * be scattered to.
 * @param updates: Array containing the values that must be used for scattering
 * operands.
 * @param computation: Computation to be used for combining the existing values
 *                     in the input array and the updates during scatter.
 * @param index_vector_dim: The dimension in indices that contains the starting
 * indices.
 * @param update_window_dims: The set of dimensions in updates shape that are
 * window dimensions.
 * @param inserted_window_dims: The set of window dimensions that must be
 * inserted into updates shape.
 * @param scatter_dims_to_operand_dims: A dimensions map from the scatter
 * indices to the operand index space.
 * @param indices_are_sorted: Whether the indices are guaranteed to be sorted by
 * the caller.
 * @param unique_indices: Whether the indices are guaranteed to be unique by the
 * caller.
 * @return bool
 */
bool TOPSOP_EXPORT
topsopScatterIsSupported(const topsopTensorHandle_t operand,
                         const topsopTensorHandle_t indices,
                         const topsopTensorHandle_t updates,
                         const topsopScatterComputationType_t computation,
                         const int64_t index_vector_dim,
                         const topsopSize_t update_window_dims,
                         const topsopSize_t inserted_window_dims,
                         const topsopSize_t scatter_dims_to_operand_dims,
                         const bool indices_are_sorted,
                         const bool unique_indices);

/**
 * @brief get output dims info for scatter
 *
 * @param operand:Array we’re scattering to.
 * @param dims:Pointer point to dimensions of output.
 * @param rank:Pointer point to rank of output.
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopScatterGetOutputDim(
    const topsopTensorHandle_t operand, int64_t *dims, int64_t *rank);

/**
 * @brief This function performs the nms function
 *
 * @param output: Selected indices from the boxes tensor. [num_selected_indices,
 * 3], the selected index format is [batch_index, class_index, box_index].
 * @param boxes: An input tensor with shape [num_batches, spatial_dimension, 4].
 *               The single box data format is indicated by center_point_box.
 * @param scores: An input tensor with shape [num_batches, num_classes,
 * spatial_dimension].
 * @param center_point_box: Integer indicate the format of the box data. The
 * default is 0. 0 - the box data is supplied as [y1, x1, y2, x2] where (y1, x1)
 * and (y2, x2) are the coordinates of any diagonal pair of box corners and the
 * coordinates can be provided as normalized (i.e., lying in the interval [0,
 * 1]) or absolute. Mostly used for TF models. 1 - the box data is supplied as
 * [x_center, y_center, width, height]. Mostly used for Pytorch models.
 * @param max_output_boxes_per_class: Integer representing the maximum number of
 * boxes to be selected per batch per class. It is a scalar. Default to 0, which
 * means no output.
 * @param iou_threshold: Float representing the threshold for deciding whether
 * boxes overlap too much with respect to IOU. It is scalar. Value range [0, 1].
 * Default to 0.
 * @param score_threshold: Float representing the threshold for deciding when to
 * remove boxes based on score. It is a scalar.
 * @param stream:Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopNms(topsopTensorHandle_t output,
                                       const topsopTensorHandle_t boxes,
                                       const topsopTensorHandle_t scores,
                                       const int32_t center_point_box,
                                       const int64_t max_output_boxes_per_class,
                                       const float iou_threshold,
                                       const float score_threshold,
                                       topsStream_t stream);

/**
 * @brief check whether topsopNms func is supported or not
 * @param boxes: An input tensor with shape [num_batches, spatial_dimension, 4].
 *               The single box data format is indicated by center_point_box.
 * @param scores: An input tensor with shape [num_batches, num_classes,
 * spatial_dimension].
 * @param center_point_box: Integer indicate the format of the box data. The
 * default is 0. 0 - the box data is supplied as [y1, x1, y2, x2] where (y1, x1)
 * and (y2, x2) are the coordinates of any diagonal pair of box corners and the
 * coordinates can be provided as normalized (i.e., lying in the interval [0,
 * 1]) or absolute. Mostly used for TF models. 1 - the box data is supplied as
 * [x_center, y_center, width, height]. Mostly used for Pytorch models.
 * @param max_output_boxes_per_class: Integer representing the maximum number of
 * boxes to be selected per batch per class. It is a scalar. Default to 0, which
 * means no output.
 * @param iou_threshold: Float representing the threshold for deciding whether
 * boxes overlap too much with respect to IOU. It is scalar. Value range [0, 1].
 * Default to 0.
 * @param score_threshold: Float representing the threshold for deciding when to
 * remove boxes based on score. It is a scalar.
 * @return bool
 */

bool TOPSOP_EXPORT
topsopNmsIsSupported(const topsopTensorHandle_t boxes,
                     const topsopTensorHandle_t scores,
                     const int32_t center_point_box,
                     const int64_t max_output_boxes_per_class,
                     const float iou_threshold,
                     const float score_threshold);

/**
 * @brief get output dims info for nms
 * @param boxes: An input tensor with shape [num_batches, spatial_dimension, 4].
 *               The single box data format is indicated by center_point_box.
 * @param scores: An input tensor with shape [num_batches, num_classes,
 * spatial_dimension].
 * @param center_point_box: Integer indicate the format of the box data. The
 * default is 0. 0 - the box data is supplied as [y1, x1, y2, x2] where (y1, x1)
 * and (y2, x2) are the coordinates of any diagonal pair of box corners and the
 * coordinates can be provided as normalized (i.e., lying in the interval [0,
 * 1]) or absolute. Mostly used for TF models. 1 - the box data is supplied as
 * [x_center, y_center, width, height]. Mostly used for Pytorch models.
 * @param max_output_boxes_per_class: Integer representing the maximum number of
 * boxes to be selected per batch per class. It is a scalar. Default to 0, which
 * means no output.
 * @param iou_threshold: Float representing the threshold for deciding whether
 * boxes overlap too much with respect to IOU. It is scalar. Value range [0, 1].
 * Default to 0.
 * @param score_threshold: Float representing the threshold for deciding when to
 * remove boxes based on score. It is a scalar.
 * @param stream:Tops stream
 * @param dims:Pointer point to dimensions of output.
 * @param rank:Pointer point to rank of output.
 * @return topsopStatus_t
 */

topsopStatus_t TOPSOP_EXPORT
topsopNmsGetOutputDim(const topsopTensorHandle_t boxes,
                      const topsopTensorHandle_t scores,
                      const int32_t center_point_box,
                      const int64_t max_output_boxes_per_class,
                      const float iou_threshold,
                      const float score_threshold,
                      topsStream_t stream,
                      int64_t *dims,
                      int64_t *rank);

/**
 * @brief This func performs the iota function
 *
 * @param output : tensor handle of output
 * @param iota_dimension  : dimension to gen iota data
 * @param stream : device stream.
 *
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopIota(topsopTensorHandle_t output,
                                        int64_t iota_dimension,
                                        topsStream_t stream);
/**
 * @brief check whether topsopGather func is support or not
 *
 * @param output_dim_info : output size info
 * @param iota_dimension  : dimension to gen iota data
 * @param dataType        : which data type to gen iota data
 *
 * @return BOOL
 */
bool TOPSOP_EXPORT topsopIotaIsSupported(topsopSize_t output_dim_info,
                                         int64_t iota_dimention,
                                         topsopDataType_t dataType);
/**
 * @brief This func performs the stft function
 *
 * @param output: Stft result
 * @param input : the input tensor
 * @param window : (Tensor, optional) the optional window function. Default:
 None
 * @param n_fft : (int) size of Fourier transform
 * @param hop_length : (int, optional) the distance between neighboring sliding
 window frames.
 *                     Default: None (treated as equal to floor(n_fft / 4))
 * @param win_length : (int, optional) the size of window frame and STFT filter.
 *                     Default: None (treated as equal to n_fft)
 * @param pad_mode : (str, optional) controls the padding method used when
 center is True. Default: "reflect"
 * @param center : (bool, optional) whether to pad input on both sides. so that
 the t-th frame is centered at time t*hop_length. Default: True
 * @param normalized : (bool, optional) controls whether to return the
 normalized STFT results Default: False
 * @param onesided (bool, optional) controls whether to return half of results
 to avoid redundancy for real inputs. Default: True for real input and window,
 False otherwise.
 * @param return_complex : (bool, optional) whether to return a complex tensor,
                   or a real tensor with an extra last dimension for the real
 and imaginary components.
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopStft(topsopTensorHandle_t output,
                                        const topsopTensorHandle_t input,
                                        const topsopTensorHandle_t window,
                                        int64_t n_fft,
                                        int64_t hop_length,
                                        int64_t win_length,
                                        char *pad_mode,
                                        bool center,
                                        bool normalized,
                                        bool onesided,
                                        bool return_complex,
                                        topsStream_t stream);

/**
 * @brief This func check whether the stft support or not.
 *
 * @param output: Stft result
 * @param input : the input tensor
 * @param window : (Tensor, optional) the optional window function. Default:
 None
 * @param n_fft : (int) size of Fourier transform
 * @param hop_length : (int, optional) the distance between neighboring sliding
 window frames.
 *                     Default: None (treated as equal to floor(n_fft / 4))
 * @param win_length : (int, optional) the size of window frame and STFT filter.
 *                     Default: None (treated as equal to n_fft)
 * @param pad_mode : (str, optional) controls the padding method used when
 center is True. Default: "reflect"
 * @param center : (bool, optional) whether to pad input on both sides. so that
 the t-th frame is centered at time t*hop_length. Default: True
 * @param normalized : (bool, optional) controls whether to return the
 normalized STFT results Default: False
 * @param onesided (bool, optional) controls whether to return half of results
 to avoid redundancy for real inputs. Default: True for real input and window,
 False otherwise.
 * @param return_complex : (bool, optional) whether to return a complex tensor,
                   or a real tensor with an extra last dimension for the real
 and imaginary components.
 * @return bool
 */
bool TOPSOP_EXPORT topsopStftIsSupported(const topsopTensorHandle_t input,
                                         const topsopTensorHandle_t window,
                                         int64_t n_fft,
                                         int64_t hop_length,
                                         int64_t win_length,
                                         char *pad_mode,
                                         bool center,
                                         bool normalized,
                                         bool onesided,
                                         bool return_complex);

/**
 * @brief This func get the output shape info for stft
 *
 * @param output: Stft result
 * @param input : the input tensor
 * @param window : (Tensor, optional) the optional window function. Default:
 None
 * @param n_fft : (int) size of Fourier transform
 * @param hop_length : (int, optional) the distance between neighboring sliding
 window frames.
 *                     Default: None (treated as equal to floor(n_fft / 4))
 * @param win_length : (int, optional) the size of window frame and STFT filter.
 *                     Default: None (treated as equal to n_fft)
 * @param pad_mode : (str, optional) controls the padding method used when
 center is True. Default: "reflect"
 * @param center : (bool, optional) whether to pad input on both sides. so that
 the t-th frame is centered at time t*hop_length. Default: True
 * @param normalized : (bool, optional) controls whether to return the
 normalized STFT results Default: False
 * @param onesided (bool, optional) controls whether to return half of results
 to avoid redundancy for real inputs. Default: True for real input and window,
 False otherwise.
 * @param return_complex : (bool, optional) whether to return a complex tensor,
                   or a real tensor with an extra last dimension for the real
 and imaginary components.
 * @param dims:Pointer point to dimensions of output.
 * @param rank:Pointer point to rank of output.
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopStftGetOutputDim(const topsopTensorHandle_t input,
                       const topsopTensorHandle_t window,
                       int64_t n_fft,
                       int64_t hop_length,
                       int64_t win_length,
                       char *pad_mode,
                       bool center,
                       bool normalized,
                       bool onesided,
                       bool return_complex,
                       int64_t *dims,
                       int64_t *rank);

/**
 * @brief This function performs the forward inference layer normalization.
 *
 * @param output Normalized tensor.
 * @param input Tensor to be normalized.
 * @param scale Scale tensor.
 * @param bias Bias tensor.
 * @param axis The first normalization dimension. If rank(input) is r,
 *             normalization will be performed along dimensions [axis,...,r-1].
 * @param epsilon The epsilon value to use to avoid division by zero.
 * @param alpha Scaling factors (in host memory) used to blend the
 *              layer output value with prior value in the destination tensor.
 * @param beta Scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor.
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopLayerNormalizationInference(topsopTensorHandle_t output,
                                  const topsopTensorHandle_t input,
                                  const topsopTensorHandle_t scale,
                                  const topsopTensorHandle_t bias,
                                  const int64_t axis,
                                  const topsopScalar_t epsilon,
                                  const topsopScalar_t alpha,
                                  const topsopScalar_t beta,
                                  topsStream_t stream);

/**
 * @brief Check whether LayerNormalizationInference func is support or not.
 * @param input Tensor to be normalized.
 * @param scale Scale tensor.
 * @param bias Bias tensor.
 * @param axis The first normalization dimension. If rank(input) is r,
 *             normalization will be performed along dimensions [axis,...,r-1].
 * @param epsilon The epsilon value to use to avoid division by zero.
 * @param alpha Scaling factors (in host memory) used to blend the
 *              layer output value with prior value in the destination tensor.
 * @param beta Scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor.
 * @return bool: Bool flag to indicate whether support this case.
 */
bool TOPSOP_EXPORT
topsopLayerNormalizationInferenceIsSupported(const topsopTensorHandle_t input,
                                             const topsopTensorHandle_t scale,
                                             const topsopTensorHandle_t bias,
                                             const int64_t axis,
                                             const topsopScalar_t epsilon,
                                             const topsopScalar_t alpha,
                                             const topsopScalar_t beta);

/**
 * @brief Get output dims of current tensor of LayerNormalizationInference
 * operator.
 * @param input Tensor to be normalized.
 * @param dims Pointer point to dimensions of output.
 * @param rank Pointer point to rank of output.
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopLayerNormalizationInferenceGetOutputDim(
    const topsopTensorHandle_t input, int64_t *dims, int64_t *rank);

/**
 * @brief This function performs the forward training layer normalization.
 *
 * @param output Normalized tensor.
 * @param mean Mean of input along dimensions [axis,...,r-1].
 * @param rstd Reciprocal of standard deviation of input along dimensions
 *             [axis,...,r-1].
 * @param input Tensor to be normalized.
 * @param scale Scale tensor.
 * @param bias Bias tensor.
 * @param axis The first normalization dimension. If rank(input) is r,
 *             normalization will be performed along dimensions [axis,...,r-1].
 * @param epsilon The epsilon value to use to avoid division by zero.
 * @param alpha Scaling factors (in host memory) used to blend the
 *              layer output value with prior value in the destination tensor.
 * @param beta Scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor.
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopLayerNormalizationTraining(topsopTensorHandle_t output,
                                 topsopTensorHandle_t mean,
                                 topsopTensorHandle_t rstd,
                                 const topsopTensorHandle_t input,
                                 const topsopTensorHandle_t scale,
                                 const topsopTensorHandle_t bias,
                                 const int64_t axis,
                                 const topsopScalar_t epsilon,
                                 const topsopScalar_t alpha,
                                 const topsopScalar_t beta,
                                 topsStream_t stream);

/**
 * @brief Check whether LayerNormalizationTraining func is support or not.
 * @param input Tensor to be normalized.
 * @param scale Scale tensor.
 * @param bias Bias tensor.
 * @param axis The first normalization dimension. If rank(input) is r,
 *             normalization will be performed along dimensions [axis,...,r-1].
 * @param epsilon The epsilon value to use to avoid division by zero.
 * @param alpha Scaling factors (in host memory) used to blend the
 *              layer output value with prior value in the destination tensor.
 * @param beta Scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor.
 * @return bool: Bool flag to indicate whether support this case.
 */
bool TOPSOP_EXPORT
topsopLayerNormalizationTrainingIsSupported(const topsopTensorHandle_t input,
                                            const topsopTensorHandle_t scale,
                                            const topsopTensorHandle_t bias,
                                            const int64_t axis,
                                            const topsopScalar_t epsilon,
                                            const topsopScalar_t alpha,
                                            const topsopScalar_t beta);

/**
 * @brief Get output dims of current tensor of LayerNormalizationTraining
 * operator.
 * @param input Tensor to be normalized.
 * @param axis The first normalization dimension. If rank(input) is r,
 *             normalization will be performed along dimensions [axis,...,r-1].
 * @param output_dims Pointer point to dimensions of output.
 * @param output_rank Pointer point to rank of output.
 * @param mean_dims Pointer point to dimensions of mean.
 * @param mean_rank Pointer point to rank of mean.
 * @param rstd_dims Pointer point to dimensions of rstd.
 * @param rstd_rank Pointer point to rank of rstd.
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopLayerNormalizationTrainingGetOutputDim(const topsopTensorHandle_t input,
                                             const int64_t axis,
                                             int64_t *output_dims,
                                             int64_t *output_rank,
                                             int64_t *mean_dims,
                                             int64_t *mean_rank,
                                             int64_t *rstd_dims,
                                             int64_t *rstd_rank);

/**
 * @brief This function performs the backward layer normalization.
 *
 * @param grad_input Gradient of input.
 * @param grad_scale Gradient of scale.
 * @param grad_bias Gradient of bias.
 * @param input Tensor to be normalized at forward.
 * @param scale Scale tensor.
 * @param mean Mean of input along dimensions [axis,...,r-1].
 * @param rstd Reciprocal of standard deviation of input along dimensions
 *             [axis,...,r-1].
 * @param grad_output Gradient of forward output.
 * @param axis The first normalization dimension. If rank(input) is r,
 *             normalization will be performed along dimensions [axis,...,r-1].
 * @param epsilon The epsilon value to use to avoid division by zero.
 * @param alpha Scaling factors (in host memory) used to blend the
 *              layer output value with prior value in the destination tensor.
 * @param beta Scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor.
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopLayerNormalizationBackward(topsopTensorHandle_t grad_input,
                                 topsopTensorHandle_t grad_scale,
                                 topsopTensorHandle_t grad_bias,
                                 const topsopTensorHandle_t input,
                                 const topsopTensorHandle_t scale,
                                 const topsopTensorHandle_t mean,
                                 const topsopTensorHandle_t rstd,
                                 const topsopTensorHandle_t grad_output,
                                 const int64_t axis,
                                 const topsopScalar_t epsilon,
                                 const topsopScalar_t alpha,
                                 const topsopScalar_t beta,
                                 topsStream_t stream);

/**
 * @brief Check whether LayerNormalizationBackward func is support or not.
 * @param input Tensor to be normalized at forward.
 * @param scale Scale tensor.
 * @param mean Mean of input along dimensions [axis,...,r-1].
 * @param rstd Reciprocal of standard deviation of input along dimensions
 *             [axis,...,r-1].
 * @param grad_output Gradient of forward output.
 * @param axis The first normalization dimension. If rank(input) is r,
 *             normalization will be performed along dimensions [axis,...,r-1].
 * @param epsilon The epsilon value to use to avoid division by zero.
 * @param alpha Scaling factors (in host memory) used to blend the
 *              layer output value with prior value in the destination tensor.
 * @param beta Scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor.
 * @return bool: Bool flag to indicate whether support this case.
 */
bool TOPSOP_EXPORT topsopLayerNormalizationBackwardIsSupported(
    const topsopTensorHandle_t input,
    const topsopTensorHandle_t scale,
    const topsopTensorHandle_t mean,
    const topsopTensorHandle_t rstd,
    const topsopTensorHandle_t grad_output,
    const int64_t axis,
    const topsopScalar_t epsilon,
    const topsopScalar_t alpha,
    const topsopScalar_t beta);

/**
 * @brief Get output dims of current tensor of LayerNormalizationBackward
 * operator.
 * @param input Tensor to be normalized at forward.
 * @param axis The first normalization dimension. If rank(input) is r,
 *             normalization will be performed along dimensions [axis,...,r-1].
 * @param grad_input_dims Pointer point to dimensions of grad_input.
 * @param grad_input_rank Pointer point to rank of grad_input.
 * @param grad_scale_dims Pointer point to dimensions of grad_scale.
 * @param grad_scale_rank Pointer point to rank of grad_scale.
 * @param grad_bias_dims Pointer point to dimensions of grad_bias.
 * @param grad_bias_rank Pointer point to rank of grad_bias.
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopLayerNormalizationBackwardGetOutputDim(const topsopTensorHandle_t input,
                                             const int64_t axis,
                                             int64_t *grad_input_dims,
                                             int64_t *grad_input_rank,
                                             int64_t *grad_scale_dims,
                                             int64_t *grad_scale_rank,
                                             int64_t *grad_bias_dims,
                                             int64_t *grad_bias_rank);

/**
 * @brief This function performs the forward inference instance normalization.
 *
 * @param output Normalized tensor.
 * @param input Tensor to be normalized.
 * @param scale Scale tensor.
 * @param bias Bias tensor.
 * @param batch_idx Specifies index of batch axis.
 * @param feature_idx Specifies index of channels axis.
 * @param epsilon The epsilon value to use to avoid division by zero.
 * @param alpha Scaling factors (in host memory) used to blend the
 *              layer output value with prior value in the destination tensor.
 * @param beta Scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor.
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopInstanceNormalizationInference(topsopTensorHandle_t output,
                                     const topsopTensorHandle_t input,
                                     const topsopTensorHandle_t scale,
                                     const topsopTensorHandle_t bias,
                                     const int64_t batch_idx,
                                     const int64_t feature_idx,
                                     const topsopScalar_t epsilon,
                                     const topsopScalar_t alpha,
                                     const topsopScalar_t beta,
                                     topsStream_t stream);

/**
 * @brief Check whether InstanceNormalizationInference func is support or not.
 * @param input Tensor to be normalized.
 * @param scale Scale tensor.
 * @param bias Bias tensor.
 * @param batch_idx Specifies index of batch axis.
 * @param feature_idx Specifies index of channels axis.
 * @param epsilon The epsilon value to use to avoid division by zero.
 * @param alpha Scaling factors (in host memory) used to blend the
 *              layer output value with prior value in the destination tensor.
 * @param beta Scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor.
 * @return bool: Bool flag to indicate whether support this case.
 */
bool TOPSOP_EXPORT topsopInstanceNormalizationInferenceIsSupported(
    const topsopTensorHandle_t input,
    const topsopTensorHandle_t scale,
    const topsopTensorHandle_t bias,
    const int64_t batch_idx,
    const int64_t feature_idx,
    const topsopScalar_t epsilon,
    const topsopScalar_t alpha,
    const topsopScalar_t beta);

/**
 * @brief Get output dims of current tensor of InstanceNormalizationInference
 * operator.
 * @param input Tensor to be normalized.
 * @param dims Pointer point to dimensions of output.
 * @param rank Pointer point to rank of output.
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopInstanceNormalizationInferenceGetOutputDim(
    const topsopTensorHandle_t input, int64_t *dims, int64_t *rank);

/**
 * @brief This function performs the forward training instance normalization.
 *
 * @param output Normalized tensor.
 * @param mean Mean of input computed per channel.
 * @param rstd Reciprocal of standard deviation of input computed per channel.
 * @param input Tensor to be normalized.
 * @param scale Scale tensor.
 * @param bias Bias tensor.
 * @param batch_idx Specifies index of batch axis.
 * @param feature_idx Specifies index of channels axis.
 * @param epsilon The epsilon value to use to avoid division by zero.
 * @param alpha Scaling factors (in host memory) used to blend the
 *              layer output value with prior value in the destination tensor.
 * @param beta Scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor.
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopInstanceNormalizationTraining(topsopTensorHandle_t output,
                                    topsopTensorHandle_t mean,
                                    topsopTensorHandle_t rstd,
                                    const topsopTensorHandle_t input,
                                    const topsopTensorHandle_t scale,
                                    const topsopTensorHandle_t bias,
                                    const int64_t batch_idx,
                                    const int64_t feature_idx,
                                    const topsopScalar_t epsilon,
                                    const topsopScalar_t alpha,
                                    const topsopScalar_t beta,
                                    topsStream_t stream);

/**
 * @brief Check whether InstanceNormalizationTraining func is support or not.
 *
 * @param input Tensor to be normalized.
 * @param scale Scale tensor.
 * @param bias Bias tensor.
 * @param batch_idx Specifies index of batch axis.
 * @param feature_idx Specifies index of channels axis.
 * @param epsilon The epsilon value to use to avoid division by zero.
 * @param alpha Scaling factors (in host memory) used to blend the
 *              layer output value with prior value in the destination tensor.
 * @param beta Scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor.
 * @return bool: Bool flag to indicate whether support this case.
 */
bool TOPSOP_EXPORT
topsopInstanceNormalizationTrainingIsSupported(const topsopTensorHandle_t input,
                                               const topsopTensorHandle_t scale,
                                               const topsopTensorHandle_t bias,
                                               const int64_t batch_idx,
                                               const int64_t feature_idx,
                                               const topsopScalar_t epsilon,
                                               const topsopScalar_t alpha,
                                               const topsopScalar_t beta);

/**
 * @brief Get output dims of current tensor of InstanceNormalizationTraining
 * operator.
 * @param input Tensor to be normalized.
 * @param batch_idx Specifies index of batch axis.
 * @param feature_idx Specifies index of channels axis.
 * @param output_dims Pointer point to dimensions of output.
 * @param output_rank Pointer point to rank of output.
 * @param mean_dims Pointer point to dimensions of mean.
 * @param mean_rank Pointer point to rank of mean.
 * @param rstd_dims Pointer point to dimensions of rstd.
 * @param rstd_rank Pointer point to rank of rstd.
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopInstanceNormalizationTrainingGetOutputDim(
    const topsopTensorHandle_t input,
    const int64_t batch_idx,
    const int64_t feature_idx,
    int64_t *output_dims,
    int64_t *output_rank,
    int64_t *mean_dims,
    int64_t *mean_rank,
    int64_t *rstd_dims,
    int64_t *rstd_rank);

/**
 * @brief This function performs the backward instance normalization operator.
 *
 * @param grad_input Gradient of input.
 * @param grad_scale Gradient of scale.
 * @param grad_bias Gradient of bias.
 * @param input Tensor to be normalized at forward.
 * @param scale Scale tensor.
 * @param mean Mean of input computed per channel.
 * @param rstd Reciprocal of standard deviation of input computed per channel.
 * @param grad_output Gradient of forward output.
 * @param batch_idx Specifies index of batch axis.
 * @param feature_idx Specifies index of channels axis.
 * @param epsilon The epsilon value to use to avoid division by zero.
 * @param alpha Scaling factors (in host memory) used to blend the
 *              layer output value with prior value in the destination tensor.
 * @param beta Scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor.
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopInstanceNormalizationBackward(topsopTensorHandle_t grad_input,
                                    topsopTensorHandle_t grad_scale,
                                    topsopTensorHandle_t grad_bias,
                                    const topsopTensorHandle_t input,
                                    const topsopTensorHandle_t scale,
                                    const topsopTensorHandle_t mean,
                                    const topsopTensorHandle_t rstd,
                                    const topsopTensorHandle_t grad_output,
                                    const int64_t batch_idx,
                                    const int64_t feature_idx,
                                    const topsopScalar_t epsilon,
                                    const topsopScalar_t alpha,
                                    const topsopScalar_t beta,
                                    topsStream_t stream);

/**
 * @brief Check whether InstanceNormalizationBackward func is support or not.
 *
 * @param input Tensor to be normalized.
 * @param scale Scale tensor.
 * @param mean Mean of input computed per channel.
 * @param rstd Reciprocal of standard deviation of input computed per channel.
 * @param grad_output Gradient of forward output.
 * @param batch_idx Specifies index of batch axis.
 * @param feature_idx Specifies index of channels axis.
 * @param epsilon The epsilon value to use to avoid division by zero.
 * @param alpha Scaling factors (in host memory) used to blend the
 *              layer output value with prior value in the destination tensor.
 * @param beta Scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor.
 * @return bool: Bool flag to indicate whether support this case.
 */
bool TOPSOP_EXPORT topsopInstanceNormalizationBackwardIsSupported(
    const topsopTensorHandle_t input,
    const topsopTensorHandle_t scale,
    const topsopTensorHandle_t mean,
    const topsopTensorHandle_t rstd,
    const topsopTensorHandle_t grad_output,
    const int64_t batch_idx,
    const int64_t feature_idx,
    const topsopScalar_t epsilon,
    const topsopScalar_t alpha,
    const topsopScalar_t beta);

/**
 * @brief Get output dims of current tensor of InstanceNormalizationBackward
 * operator.
 * @param input Tensor to be normalized.
 * @param feature_idx Specifies index of channels axis.
 * @param grad_input_dims Pointer point to dimensions of grad_input.
 * @param grad_input_rank Pointer point to rank of grad_input.
 * @param grad_scale_dims Pointer point to dimensions of grad_scale.
 * @param grad_scale_rank Pointer point to rank of grad_scale.
 * @param grad_bias_dims Pointer point to dimensions of grad_bias.
 * @param grad_bias_rank Pointer point to rank of grad_bias.
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopInstanceNormalizationBackwardGetOutputDim(
    const topsopTensorHandle_t input,
    const int64_t feature_idx,
    int64_t *grad_input_dims,
    int64_t *grad_input_rank,
    int64_t *grad_scale_dims,
    int64_t *grad_scale_rank,
    int64_t *grad_bias_dims,
    int64_t *grad_bias_rank);

/**
 * @brief This function performs the forward inference group normalization.
 *
 * @param output Normalized tensor.
 * @param input Tensor to be normalized.
 * @param scale Scale tensor.
 * @param bias Bias tensor.
 * @param batch_idx Specifies index of batch axis.
 * @param feature_idx Specifies index of channels axis.
 * @param num_groups Divide the channels into this number of groups over which
 *                   normalization statistics are computed. This number must be
 *                   commensurate with the number of channels in inputs.
 * @param epsilon The epsilon value to use to avoid division by zero.
 * @param alpha Scaling factors (in host memory) used to blend the
 *              layer output value with prior value in the destination tensor.
 * @param beta Scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor.
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopGroupNormalizationInference(topsopTensorHandle_t output,
                                  const topsopTensorHandle_t input,
                                  const topsopTensorHandle_t scale,
                                  const topsopTensorHandle_t bias,
                                  const int64_t batch_idx,
                                  const int64_t feature_idx,
                                  const int64_t num_groups,
                                  const topsopScalar_t epsilon,
                                  const topsopScalar_t alpha,
                                  const topsopScalar_t beta,
                                  topsStream_t stream);

/**
 * @brief Check whether GroupNormalizationInference func is support or not.
 * @param input Tensor to be normalized.
 * @param scale Scale tensor.
 * @param bias Bias tensor.
 * @param batch_idx Specifies index of batch axis.
 * @param feature_idx Specifies index of channels axis.
 * @param num_groups Divide the channels into this number of groups over which
 *                   normalization statistics are computed. This number must be
 *                   commensurate with the number of channels in inputs.
 * @param epsilon The epsilon value to use to avoid division by zero.
 * @param alpha Scaling factors (in host memory) used to blend the
 *              layer output value with prior value in the destination tensor.
 * @param beta Scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor.
 * @return bool: Bool flag to indicate whether support this case.
 */
bool TOPSOP_EXPORT
topsopGroupNormalizationInferenceIsSupported(const topsopTensorHandle_t input,
                                             const topsopTensorHandle_t scale,
                                             const topsopTensorHandle_t bias,
                                             const int64_t batch_idx,
                                             const int64_t feature_idx,
                                             const int64_t num_groups,
                                             const topsopScalar_t epsilon,
                                             const topsopScalar_t alpha,
                                             const topsopScalar_t beta);

/**
 * @brief Get output dims of current tensor of GroupNormalizationInference
 * operator.
 * @param input Tensor to be normalized.
 * @param dims Pointer point to dimensions of output.
 * @param rank Pointer point to rank of output.
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopGroupNormalizationInferenceGetOutputDim(
    const topsopTensorHandle_t input, int64_t *dims, int64_t *rank);

/**
 * @brief This function performs the forward training group normalization.
 *
 * @param output Normalized tensor.
 * @param mean Mean of input computed separately over each group.
 * @param rstd Reciprocal of standard deviation of input computed separately
 *             over each group.
 * @param input Tensor to be normalized.
 * @param scale Scale tensor.
 * @param bias Bias tensor.
 * @param batch_idx Specifies index of batch axis.
 * @param feature_idx Specifies index of channels axis.
 * @param num_groups Divide the channels into this number of groups over which
 *                   normalization statistics are computed. This number must be
 *                   commensurate with the number of channels in inputs.
 * @param epsilon The epsilon value to use to avoid division by zero.
 * @param alpha Scaling factors (in host memory) used to blend the
 *              layer output value with prior value in the destination tensor.
 * @param beta Scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor.
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopGroupNormalizationTraining(topsopTensorHandle_t output,
                                 topsopTensorHandle_t mean,
                                 topsopTensorHandle_t rstd,
                                 const topsopTensorHandle_t input,
                                 const topsopTensorHandle_t scale,
                                 const topsopTensorHandle_t bias,
                                 const int64_t batch_idx,
                                 const int64_t feature_idx,
                                 const int64_t num_groups,
                                 const topsopScalar_t epsilon,
                                 const topsopScalar_t alpha,
                                 const topsopScalar_t beta,
                                 topsStream_t stream);

/**
 * @brief Check whether GroupNormalizationTraining func is support or not.
 * @param input Tensor to be normalized.
 * @param scale Scale tensor.
 * @param bias Bias tensor.
 * @param batch_idx Specifies index of batch axis.
 * @param feature_idx Specifies index of channels axis.
 * @param num_groups Divide the channels into this number of groups over which
 *                   normalization statistics are computed. This number must be
 *                   commensurate with the number of channels in inputs.
 * @param epsilon The epsilon value to use to avoid division by zero.
 * @param alpha Scaling factors (in host memory) used to blend the
 *              layer output value with prior value in the destination tensor.
 * @param beta Scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor.
 * @return bool: Bool flag to indicate whether support this case.
 */
bool TOPSOP_EXPORT
topsopGroupNormalizationTrainingIsSupported(const topsopTensorHandle_t input,
                                            const topsopTensorHandle_t scale,
                                            const topsopTensorHandle_t bias,
                                            const int64_t batch_idx,
                                            const int64_t feature_idx,
                                            const int64_t num_groups,
                                            const topsopScalar_t epsilon,
                                            const topsopScalar_t alpha,
                                            const topsopScalar_t beta);

/**
 * @brief Get output dims of current tensor of GroupNormalizationTraining
 * operator.
 * @param input Tensor to be normalized.
 * @param batch_idx Specifies index of batch axis.
 * @param num_groups Divide the channels into this number of groups over which
 *                   normalization statistics are computed. This number must be
 *                   commensurate with the number of channels in inputs.
 * @param output_dims Pointer point to dimensions of output.
 * @param output_rank Pointer point to rank of output.
 * @param mean_dims Pointer point to dimensions of mean.
 * @param mean_rank Pointer point to rank of mean.
 * @param rstd_dims Pointer point to dimensions of rstd.
 * @param rstd_rank Pointer point to rank of rstd.
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopGroupNormalizationTrainingGetOutputDim(const topsopTensorHandle_t input,
                                             const int64_t batch_idx,
                                             const int64_t num_groups,
                                             int64_t *output_dims,
                                             int64_t *output_rank,
                                             int64_t *mean_dims,
                                             int64_t *mean_rank,
                                             int64_t *rstd_dims,
                                             int64_t *rstd_rank);

/**
 * @brief This function performs the backward group normalization.
 *
 * @param grad_input Gradient of input.
 * @param grad_scale Gradient of scale.
 * @param grad_bias Gradient of bias.
 * @param input Tensor to be normalized.
 * @param scale Scale tensor.
 * @param mean Mean of input computed separately over each group.
 * @param rstd Reciprocal of standard deviation of input computed separately
 *             over each group.
 * @param grad_output Gradient of forward output.
 * @param batch_idx Specifies index of batch axis.
 * @param feature_idx Specifies index of channels axis.
 * @param num_groups Divide the channels into this number of groups over which
 *                   normalization statistics are computed. This number must be
 *                   commensurate with the number of channels in inputs.
 * @param epsilon The epsilon value to use to avoid division by zero.
 * @param alpha Scaling factors (in host memory) used to blend the
 *              layer output value with prior value in the destination tensor.
 * @param beta Scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor.
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopGroupNormalizationBackward(topsopTensorHandle_t grad_input,
                                 topsopTensorHandle_t grad_scale,
                                 topsopTensorHandle_t grad_bias,
                                 const topsopTensorHandle_t input,
                                 const topsopTensorHandle_t scale,
                                 const topsopTensorHandle_t mean,
                                 const topsopTensorHandle_t rstd,
                                 const topsopTensorHandle_t grad_output,
                                 const int64_t batch_idx,
                                 const int64_t feature_idx,
                                 const int64_t num_groups,
                                 const topsopScalar_t epsilon,
                                 const topsopScalar_t alpha,
                                 const topsopScalar_t beta,
                                 topsStream_t stream);

/**
 * @brief Check whether GroupNormalizationBackward func is support or not.
 * @param input Tensor to be normalized.
 * @param scale Scale tensor.
 * @param mean Mean of input computed separately over each group.
 * @param rstd Reciprocal of standard deviation of input computed separately
 *             over each group.
 * @param grad_output Gradient of forward output.
 * @param batch_idx Specifies index of batch axis.
 * @param feature_idx Specifies index of channels axis.
 * @param num_groups Divide the channels into this number of groups over which
 *                   normalization statistics are computed. This number must be
 *                   commensurate with the number of channels in inputs.
 * @param epsilon The epsilon value to use to avoid division by zero.
 * @param alpha Scaling factors (in host memory) used to blend the
 *              layer output value with prior value in the destination tensor.
 * @param beta Scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor.
 * @return bool: Bool flag to indicate whether support this case.
 */
bool TOPSOP_EXPORT topsopGroupNormalizationBackwardIsSupported(
    const topsopTensorHandle_t input,
    const topsopTensorHandle_t scale,
    const topsopTensorHandle_t mean,
    const topsopTensorHandle_t rstd,
    const topsopTensorHandle_t grad_output,
    const int64_t batch_idx,
    const int64_t feature_idx,
    const int64_t num_groups,
    const topsopScalar_t epsilon,
    const topsopScalar_t alpha,
    const topsopScalar_t beta);

/**
 * @brief Get output dims of current tensor of GroupNormalizationBackward
 * operator.
 * @param input Tensor to be normalized.
 * @param feature_idx Specifies index of channels axis.
 * @param grad_input_dims Pointer point to dimensions of grad_input.
 * @param grad_input_rank Pointer point to rank of grad_input.
 * @param grad_scale_dims Pointer point to dimensions of grad_scale.
 * @param grad_scale_rank Pointer point to rank of grad_scale.
 * @param grad_bias_dims Pointer point to dimensions of grad_bias.
 * @param grad_bias_rank Pointer point to rank of grad_bias.
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopGroupNormalizationBackwardGetOutputDim(const topsopTensorHandle_t input,
                                             const int64_t feature_idx,
                                             int64_t *grad_input_dims,
                                             int64_t *grad_input_rank,
                                             int64_t *grad_scale_dims,
                                             int64_t *grad_scale_rank,
                                             int64_t *grad_bias_dims,
                                             int64_t *grad_bias_rank);

/**
 * @brief This function performs the reverse function.
 *
 * @param output Output tensor.
 * @param input_value Input tensor.
 * @param dims The axis to flip on.
 * @param alpha Scaling factors (in host memory) used to blend the
 *              layer output value with prior value in the destination tensor.
 * @param beta Scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor.
 * @param stream Tops stream.
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopReverse(topsopTensorHandle_t output,
                                           const topsopTensorHandle_t input,
                                           topsopSize_t dims,
                                           topsopScalar_t alpha,
                                           topsopScalar_t beta,
                                           topsStream_t stream);

/**
 * @brief check whether topsopReverse func is support or not
 * @param input The input tensor.
 * @return bool
 */
bool TOPSOP_EXPORT topsopReverseIsSupported(const topsopTensorHandle_t input);

/**
 * @brief Get output dims of current tensor of reverse operator
 * @param input The Input tensor.
 * @param dims The dims of output tensor.
 * @param rank The rank of output tensor.
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopReverseGetOutputDim(
    const topsopTensorHandle_t input, int64_t *dims, int64_t *rank);

/**
 * @brief This function performs the copy function.
 *
 * @param output Output tensor.
 * @param input_value Input tensor.
 * @param alpha Scaling factors (in host memory) used to blend the
 *              layer output value with prior value in the destination tensor.
 * @param beta Scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor.
 * @param stream Tops stream.
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopCopy(topsopTensorHandle_t output,
                                        const topsopTensorHandle_t input,
                                        topsopScalar_t alpha,
                                        topsopScalar_t beta,
                                        topsStream_t stream);

/**
 * @brief check whether topsopCopy func is support or not
 * @param input The input tensor.
 * @return bool
 */
bool TOPSOP_EXPORT topsopCopyIsSupported(const topsopTensorHandle_t input);

/**
 * @brief Get output dims of current tensor of copy operator
 * @param input The input tensor.
 * @param dims The dims of output tensor.
 * @param rank The rank of output tensor.
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopCopyGetOutputDim(
    const topsopTensorHandle_t input, int64_t *dims, int64_t *rank);

/**
 * @brief This function computes the gradient of the rectified linear function.
 *
 * @param grad_input Output differential tensor
 * @param grad_output Input differential tensor
 * @param in_out Input tensor
 * @param reluNanOpt Enumerant to specify the Nan propagation mode
 * @param alpha scaling factors (in host memory) used to blend the
 *              layer output value with prior value in the destination tensor
 * @param beta scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopReluBackward(topsopTensorHandle_t grad_input,
                   const topsopTensorHandle_t grad_output,
                   const topsopTensorHandle_t in_out,
                   topsopNanPropagation_t reluNanOpt,
                   topsopScalar_t alpha,
                   topsopScalar_t beta,
                   topsStream_t stream);

/**
 * @brief This function computes the gradient of the sigmoid function.
 *
 * @param grad_input Output differential tensor
 * @param grad_output Input differential tensor
 * @param in_out Input tensor
 * @param reluNanOpt Enumerant to specify the Nan propagation mode
 * @param alpha scaling factors (in host memory) used to blend the
 *              layer output value with prior value in the destination tensor
 * @param beta scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopSigmoidBackward(topsopTensorHandle_t grad_input,
                      const topsopTensorHandle_t grad_output,
                      const topsopTensorHandle_t in_out,
                      topsopNanPropagation_t reluNanOpt,
                      topsopScalar_t alpha,
                      topsopScalar_t beta,
                      topsStream_t stream);

/**
 * @brief This function computes the gradient of the clipped rectified
 *        linear function.
 *
 * @param grad_input Output differential tensor
 * @param grad_output Input differential tensor
 * @param in_out Input tensor
 * @param reluNanOpt Enumerant to specify the Nan propagation mode
 * @param coef Floating point number
 * @param alpha scaling factors (in host memory) used to blend the
 *              layer output value with prior value in the destination tensor
 * @param beta scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopClippedReluBackward(topsopTensorHandle_t grad_input,
                          const topsopTensorHandle_t grad_output,
                          const topsopTensorHandle_t in_out,
                          topsopNanPropagation_t reluNanOpt,
                          topsopScalar_t coef,
                          topsopScalar_t alpha,
                          topsopScalar_t beta,
                          topsStream_t stream);

/**
 * @brief This function computes the gradient of the exponential
 *        linear function.
 *
 * @param grad_input Output differential tensor
 * @param grad_output Input differential tensor
 * @param in_out Input tensor
 * @param reluNanOpt Enumerant to specify the Nan propagation mode
 * @param coef Floating point number
 * @param alpha scaling factors (in host memory) used to blend the
 *              layer output value with prior value in the destination tensor
 * @param beta scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopEluBackward(topsopTensorHandle_t grad_input,
                  const topsopTensorHandle_t grad_output,
                  const topsopTensorHandle_t in_out,
                  topsopNanPropagation_t reluNanOpt,
                  topsopScalar_t coef,
                  topsopScalar_t alpha,
                  topsopScalar_t beta,
                  topsStream_t stream);

/**
 * @brief This function computes the gradient of the hyperbolic
 *        tangent function.
 *
 * @param grad_input Output differential tensor
 * @param grad_output Input differential tensor
 * @param in_out Input tensor
 * @param reluNanOpt Enumerant to specify the Nan propagation mode
 * @param alpha scaling factors (in host memory) used to blend the
 *              layer output value with prior value in the destination tensor
 * @param beta scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopTanhBackward(topsopTensorHandle_t grad_input,
                   const topsopTensorHandle_t grad_output,
                   const topsopTensorHandle_t in_out,
                   topsopNanPropagation_t reluNanOpt,
                   topsopScalar_t alpha,
                   topsopScalar_t beta,
                   topsStream_t stream);

/**
 * @brief This function computes the gradient of the swish function.
 *
 * @param grad_input Output differential tensor
 * @param grad_output Input differential tensor
 * @param in_out Input tensor
 * @param reluNanOpt Enumerant to specify the Nan propagation mode
 * @param alpha scaling factors (in host memory) used to blend the
 *              layer output value with prior value in the destination tensor
 * @param beta scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopSwishBackward(topsopTensorHandle_t grad_input,
                    const topsopTensorHandle_t grad_output,
                    const topsopTensorHandle_t in_out,
                    topsopNanPropagation_t reluNanOpt,
                    topsopScalar_t alpha,
                    topsopScalar_t beta,
                    topsStream_t stream);

/**
 * @brief This function computes the gradient ofthe leaky relu function.
 *
 * @param grad_input Output differential tensor
 * @param grad_output Input differential tensor
 * @param in_out Input tensor
 * @param reluNanOpt Enumerant to specify the Nan propagation mode
 * @param coef Floating point number
 * @param alpha scaling factors (in host memory) used to blend the
 *              layer output value with prior value in the destination tensor
 * @param beta scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopLeakyReluBackward(topsopTensorHandle_t grad_input,
                        const topsopTensorHandle_t grad_output,
                        const topsopTensorHandle_t in_out,
                        topsopNanPropagation_t reluNanOpt,
                        topsopScalar_t coef,
                        topsopScalar_t alpha,
                        topsopScalar_t beta,
                        topsStream_t stream);

/**
 * @brief check whether current relu backward operator support or not
 *
 * @param grad_output Input differential tensor
 * @param in_out Input tensor
 * @param reluNanOpt Enumerant to specify the Nan propagation mode
 * @param alpha scaling factors (in host memory) used to blend the
 *              layer output value with prior value in the destination tensor
 * @param beta scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor
 * @return bool
 */
bool TOPSOP_EXPORT
topsopReluBackwardIsSupported(const topsopTensorHandle_t grad_output,
                              const topsopTensorHandle_t in_out,
                              topsopNanPropagation_t reluNanOpt,
                              topsopScalar_t alpha,
                              topsopScalar_t beta);

/**
 * @brief get output dims of current tensor relu backward operator
 *
 * @param grad_output Input differential tensor
 * @param in_out Input tensor
 * @param dims The dims of output tensor
 * @param rank The rank of output tensor
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopReluBackwardGetOutputDim(const topsopTensorHandle_t grad_output,
                               const topsopTensorHandle_t in_out,
                               int64_t *dims,
                               int64_t *rank);

/**
 * @brief check whether current sigmoid backward operator support or not
 *
 * @param grad_output Input differential tensor
 * @param in_out Input tensor
 * @param reluNanOpt Enumerant to specify the Nan propagation mode
 * @param alpha scaling factors (in host memory) used to blend the
 *              layer output value with prior value in the destination tensor
 * @param beta scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor
 * @return bool
 */
bool TOPSOP_EXPORT
topsopSigmoidBackwardIsSupported(const topsopTensorHandle_t grad_output,
                                 const topsopTensorHandle_t in_out,
                                 topsopNanPropagation_t reluNanOpt,
                                 topsopScalar_t alpha,
                                 topsopScalar_t beta);

/**
 * @brief get output dims of current tensor sigmoid backward operator
 *
 * @param grad_output Input differential tensor
 * @param in_out Input tensor
 * @param dims The dims of output tensor
 * @param rank The rank of output tensor
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopSigmoidBackwardGetOutputDim(const topsopTensorHandle_t grad_output,
                                  const topsopTensorHandle_t in_out,
                                  int64_t *dims,
                                  int64_t *rank);

/**
 * @brief check whether current clipped relu backward operator support or not
 *
 * @param grad_output Input differential tensor
 * @param in_out Input tensor
 * @param reluNanOpt Enumerant to specify the Nan propagation mode
 * @param coef Floating point number
 * @param alpha scaling factors (in host memory) used to blend the
 *              layer output value with prior value in the destination tensor
 * @param beta scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor
 * @return bool
 */
bool TOPSOP_EXPORT
topsopClippedReluBackwardIsSupported(const topsopTensorHandle_t grad_output,
                                     const topsopTensorHandle_t in_out,
                                     topsopNanPropagation_t reluNanOpt,
                                     topsopScalar_t coef,
                                     topsopScalar_t alpha,
                                     topsopScalar_t beta);

/**
 * @brief get output dims of current tensor clipped relu backward operator
 *
 * @param grad_output Input differential tensor
 * @param in_out Input tensor
 * @param dims The dims of output tensor
 * @param rank The rank of output tensor
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopClippedReluBackwardGetOutputDim(const topsopTensorHandle_t grad_output,
                                      const topsopTensorHandle_t in_out,
                                      int64_t *dims,
                                      int64_t *rank);

/**
 * @brief check whether current elu backward operator support or not
 *
 * @param grad_output Input differential tensor
 * @param in_out Input tensor
 * @param reluNanOpt Enumerant to specify the Nan propagation mode
 * @param coef Floating point number
 * @param alpha scaling factors (in host memory) used to blend the
 *              layer output value with prior value in the destination tensor
 * @param beta scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor
 * @return bool
 */
bool TOPSOP_EXPORT
topsopEluBackwardIsSupported(const topsopTensorHandle_t grad_output,
                             const topsopTensorHandle_t in_out,
                             topsopNanPropagation_t reluNanOpt,
                             topsopScalar_t coef,
                             topsopScalar_t alpha,
                             topsopScalar_t beta);

/**
 * @brief get output dims of current tensor elu backward operator
 *
 * @param grad_output Input differential tensor
 * @param in_out Input tensor
 * @param dims The dims of output tensor
 * @param rank The rank of output tensor
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopEluBackwardGetOutputDim(const topsopTensorHandle_t grad_output,
                              const topsopTensorHandle_t in_out,
                              int64_t *dims,
                              int64_t *rank);

/**
 * @brief check whether current tanh backward operator support or not
 *
 * @param grad_output Input differential tensor
 * @param in_out Input tensor
 * @param reluNanOpt Enumerant to specify the Nan propagation mode
 * @param alpha scaling factors (in host memory) used to blend the
 *              layer output value with prior value in the destination tensor
 * @param beta scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor
 * @return bool
 */
bool TOPSOP_EXPORT
topsopTanhBackwardIsSupported(const topsopTensorHandle_t grad_output,
                              const topsopTensorHandle_t in_out,
                              topsopNanPropagation_t reluNanOpt,
                              topsopScalar_t alpha,
                              topsopScalar_t beta);

/**
 * @brief get output dims of current tensor tanh backward operator
 *
 * @param grad_output Input differential tensor
 * @param in_out Input tensor
 * @param dims The dims of output tensor
 * @param rank The rank of output tensor
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopTanhBackwardGetOutputDim(const topsopTensorHandle_t grad_output,
                               const topsopTensorHandle_t in_out,
                               int64_t *dims,
                               int64_t *rank);

/**
 * @brief check whether current swish backward operator support or not
 *
 * @param grad_output Input differential tensor
 * @param in_out Input tensor
 * @param reluNanOpt Enumerant to specify the Nan propagation mode
 * @param alpha scaling factors (in host memory) used to blend the
 *              layer output value with prior value in the destination tensor
 * @param beta scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor
 * @return bool
 */
bool TOPSOP_EXPORT
topsopSwishBackwardIsSupported(const topsopTensorHandle_t grad_output,
                               const topsopTensorHandle_t in_out,
                               topsopNanPropagation_t reluNanOpt,
                               topsopScalar_t alpha,
                               topsopScalar_t beta);

/**
 * @brief get output dims of current tensor swish backward operator
 *
 * @param grad_output Input differential tensor
 * @param in_out Input tensor
 * @param dims The dims of output tensor
 * @param rank The rank of output tensor
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopSwishBackwardGetOutputDim(const topsopTensorHandle_t grad_output,
                                const topsopTensorHandle_t in_out,
                                int64_t *dims,
                                int64_t *rank);

/**
 * @brief check whether current leaky relu backward operator support or not
 *
 * @param grad_output Input differential tensor
 * @param in_out Input tensor
 * @param reluNanOpt Enumerant to specify the Nan propagation mode
 * @param coef Floating point number
 * @param alpha scaling factors (in host memory) used to blend the
 *              layer output value with prior value in the destination tensor
 * @param beta scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor
 * @return bool
 */
bool TOPSOP_EXPORT
topsopLeakyReluBackwardIsSupported(const topsopTensorHandle_t grad_output,
                                   const topsopTensorHandle_t in_out,
                                   topsopNanPropagation_t reluNanOpt,
                                   topsopScalar_t coef,
                                   topsopScalar_t alpha,
                                   topsopScalar_t beta);

/**
 * @brief get output dims of current tensor leaky relu backward operator
 *
 * @param grad_output Input differential tensor
 * @param in_out Input tensor
 * @param dims The dims of output tensor
 * @param rank The rank of output tensor
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopLeakyReluBackwardGetOutputDim(const topsopTensorHandle_t grad_output,
                                    const topsopTensorHandle_t in_out,
                                    int64_t *dims,
                                    int64_t *rank);

/**
 * @brief This function generate random uniform data operation.
 *
 * @param output Output tensor
 * @param seed Seed used to initialize random number generator states
 * @param offset offset used to initialize random number generator states
 * @param lower_limit Scalar of type T specifying lower limit of interval
 * @param upper_limit Scalar of type T specifying upper limit of interval
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopRngUniform(topsopTensorHandle_t output,
                                              const uint64_t seed,
                                              const uint64_t offset,
                                              const topsopScalar_t lower_limit,
                                              const topsopScalar_t upper_limit,
                                              topsStream_t stream);

/**
 * @brief check whether current rng uniform operator support or not
 *
 * @param output Output tensor which will hold rng uniform data
 * @param seed Seed used to initialize random number generator states.
 *             The lowest 37 bits of it can not be all zeros.
 * @param offset offset used to initialize random number generator states
 * @param lower_limit Scalar of type T specifying lower limit of interval
 * @param upper_limit Scalar of type T specifying upper limit of interval
 * @return bool
 */
bool TOPSOP_EXPORT
topsopRngUniformIsSupported(const topsopTensorHandle_t output,
                            const uint64_t seed,
                            const uint64_t offset,
                            const topsopScalar_t lower_limit,
                            const topsopScalar_t upper_limit);

/**
 * @brief This function performs forward dropout operation.
 *
 * @param output Result tensor
 * @param mask  Mask tensor
 * @param input Input tensor
 * @param dropout The probability with which the value from input is
 *                set to zero during the dropout layer
 * @param seed Seed used to initialize random number generator states
 * @param offset offset used to initialize random number generator states
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopDropout(topsopTensorHandle_t output,
                                           topsopTensorHandle_t mask,
                                           const topsopTensorHandle_t input,
                                           const float dropout,
                                           const uint64_t seed,
                                           const uint64_t offset,
                                           topsStream_t stream);

/**
 * @brief This function performs forward dropout operation.
 *        Only used for topsdnn adaptation.
 *
 * @param output Result tensor
 * @param mask  Mask tensor
 * @param input Input tensor
 * @param dropout The probability with which the value from input is
 *                set to zero during the dropout layer
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopDropoutForDNN(topsopTensorHandle_t output,
                    topsopTensorHandle_t mask,
                    const topsopTensorHandle_t input,
                    const float dropout,
                    topsStream_t stream);

/**
 * @brief check whether current dropout operator support or not
 *
 * @param input Input tensor
 * @param dropout The probability with which the value from input is
 *                set to zero during the dropout layer
 * @param seed Seed used to initialize random number generator states
 * @param offset offset used to initialize random number generator states
 * @return topsopStatus_t
 */
bool TOPSOP_EXPORT topsopDropoutIsSupported(const topsopTensorHandle_t input,
                                            const float dropout,
                                            const uint64_t seed,
                                            const uint64_t offset);

/**
 * @brief get dims and rank of output and mask tensor of
 *        current tensor dropout operator
 *
 * @param input Input tensor
 * @param output_dims The dims of output tensor
 * @param output_rank The rank of output tensor
 * @param mask_dims The dims of mask tensor
 * @param mask_rank The rank of mask tensor
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopDropoutGetOutputDim(const topsopTensorHandle_t input,
                          int64_t *output_dims,
                          int64_t *output_rank,
                          int64_t *mask_dims,
                          int64_t *mask_rank);

/**
 * @brief This function is used to query the amount of space required to
 * store the states of the random number generators.
 *
 * @param stateSizeInBytes Amount of device memory needed to store random
 *                         generator states
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopDropoutGetStatesSize(int64_t *stateSizeInBytes);

/**
 * @brief This function initializes dropout operation.
 *
 * @param states Pointer to user-allocated device memory that will hold
 *               random number generator states
 * @param stateSizeInBytes Specifies the size in bytes of the provided memory
 *                         for the states
 * @param seed Seed used to initialize random number generator states
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopSetDropout(void *states,
                                              int64_t stateSizeInBytes,
                                              uint64_t seed,
                                              topsStream_t stream);

/**
 * @brief This function queries the fields of a previously initialized
 * dropout operation.
 *
 * @param states Pointer to user-allocated device memory that holds random
 *               number generator states
 * @param stateSizeInBytes Amount of device memory needed to store random
 *                         generator states
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopGetDropout(void *states,
                                              int64_t stateSizeInBytes,
                                              topsStream_t stream);

/**
 * @brief This function restores a dropout operation to
 * a previously saved-off state.
 *
 * @param states Pointer to device memory that holds random number
 *               generator states
 * @param stateSizeInBytes Amount of device memory needed to store random
 *                         generator states
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopRestoreDropout(void *states,
                                                  int64_t stateSizeInBytes,
                                                  topsStream_t stream);

/**
 * @brief This function performs backward dropout operation.
 *
 * @param output Result tensor
 * @param input Input tensor
 * @param mask  Mask tensor
 * @param dropout The probability with which the value from input is
 *                set to zero during the dropout layer
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopDropoutBackward(topsopTensorHandle_t output,
                      const topsopTensorHandle_t input,
                      const topsopTensorHandle_t mask,
                      const float dropout,
                      topsStream_t stream);

/**
 * @brief check whether current dropout backward operator support or not
 *
 * @param input Input tensor
 * @param mask  Mask tensor
 * @param dropout The probability with which the value from input is
 *                set to zero during the dropout layer
 * @return topsopStatus_t
 */
bool TOPSOP_EXPORT
topsopDropoutBackwardIsSupported(const topsopTensorHandle_t input,
                                 const topsopTensorHandle_t mask,
                                 const float dropout);

/**
 * @brief get output dims of current tensor dropout backward operator
 *
 * @param input Input tensor
 * @param mask Mask tensor
 * @param dims The dims of output tensor
 * @param rank The rank of output tensor
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopDropoutBackwardGetOutputDim(const topsopTensorHandle_t input,
                                  const topsopTensorHandle_t mask,
                                  int64_t *dims,
                                  int64_t *rank);

/**
 * @brief This function reshapes the tenosr to a new shape.
 *
 * @param output Output tensor
 * @param input Input tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopReshape(topsopTensorHandle_t output,
                                           const topsopTensorHandle_t input,
                                           topsStream_t stream);

/**
 * @brief check whether reshape func is support or not
 * @param input The input tensor.
 * @param output The output tensor.
 * @return bool
 */
bool TOPSOP_EXPORT topsopReshapeIsSupported(const topsopTensorHandle_t input,
                                            const topsopTensorHandle_t output);

/**
 * @brief This function performs the get_dim_size function.
 *
 * @param output Output tensor.
 * @param input_value Input tensor.
 * @param dim The dim to get input shape size.
 * @param stream Tops stream.
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopGetDimSize(topsopTensorHandle_t output,
                                              const topsopTensorHandle_t input,
                                              int64_t dim,
                                              topsStream_t stream);

/**
 * @brief check whether topsopGetDimSize func is support or not
 * @param input The input tensor.
 * @param dim The dim to get input shape size.
 * @return bool
 */
bool TOPSOP_EXPORT topsopGetDimSizeIsSupported(const topsopTensorHandle_t input,
                                               int64_t dim);

/**
 * @brief Get output dims of current tensor of get_dim_size operator
 * @param input The input tensor.
 * @param dims The dims of output tensor.
 * @param rank The rank of output tensor.
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopGetDimSizeGetOutputDim(
    const topsopTensorHandle_t input, int64_t *dims, int64_t *rank);

/**
 * @brief This function performs the set_dim_size function.
 *
 * @param output Output tensor.
 * @param input_value Input tensor.
 * @param dim The dim to get input shape size.
 * @param size The size to set.
 * @param alpha Scaling factors (in host memory) used to blend the
 *              layer output value with prior value in the destination tensor.
 * @param beta Scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor.
 * @param stream Tops stream.
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopSetDimSize(topsopTensorHandle_t output,
                                              const topsopTensorHandle_t input,
                                              int64_t dim,
                                              int64_t size,
                                              topsopScalar_t alpha,
                                              topsopScalar_t beta,
                                              topsStream_t stream);

/**
 * @brief check whether topsopSetDimSize func is support or not
 * @param input The input tensor.
 * @param dim The dim to get input shape size.
 * @param size The size to set.
 * @param alpha Scaling factors (in host memory) used to blend the
 *              layer output value with prior value in the destination tensor.
 * @param beta Scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor.
 * @return bool
 */
bool TOPSOP_EXPORT topsopSetDimSizeIsSupported(const topsopTensorHandle_t input,
                                               int64_t dim,
                                               int64_t size,
                                               topsopScalar_t alpha,
                                               topsopScalar_t beta);

/**
 * @brief Get output dims of current tensor of set_dim_size operator
 * @param input The input tensor.
 * @param dim The dim to get input shape size.
 * @param size The size to set.
 * @param dims The dims of output tensor.
 * @param rank The rank of output tensor.
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopSetDimSizeGetOutputDim(const topsopTensorHandle_t input,
                             int64_t dim,
                             int64_t size,
                             int64_t *dims,
                             int64_t *rank);

/**
 * @brief This function performs the roipooling function.
 *
 * @param output Output tensor.
 * @param input_value Input tensor.
 * @param boxes The box coordinates in (x1, y1, x2, y2) format where the regions
 *              will be taken from.
 * @param output_size The size of the output (in bins or pixels) after the
 *                    pooling is performed, as (height, width).
 * @param spatial_scale A scaling factor that maps the box coordinates to the
 *                      input coordinates.
 * @param alpha Scaling factors (in host memory) used to blend the
 *              layer output value with prior value in the destination tensor.
 * @param beta Scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor.
 * @param stream Tops stream.
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopRoipooling(topsopTensorHandle_t out,
                                              const topsopTensorHandle_t input,
                                              const topsopTensorHandle_t boxes,
                                              topsopSize_t output_size,
                                              topsopScalar_t spatial_scale,
                                              topsopScalar_t alpha,
                                              topsopScalar_t beta,
                                              topsStream_t stream);

/**
 * @brief check whether topsopRoipooling func is support or not
 * @param input The input tensor.
 * @param boxes The box coordinates in (x1, y1, x2, y2) format where the regions
 *              will be taken from.
 * @return bool
 */
bool TOPSOP_EXPORT topsopRoipoolingIsSupported(
    const topsopTensorHandle_t input, const topsopTensorHandle_t boxes);

/**
 * @brief Get output dims of current tensor of roipooling operator
 * @param input The input tensor.
 * @param boxes The box coordinates in (x1, y1, x2, y2) format where the regions
 *              will be taken from.
 * @param output_size The size of the output (in bins or pixels) after the
 *                    pooling is performed, as (height, width).
 * @param dims The dims of output tensor.
 * @param rank The rank of output tensor.
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopRoipoolingGetOutputDim(const topsopTensorHandle_t input,
                             const topsopTensorHandle_t boxes,
                             topsopSize_t output_size,
                             int64_t *dims,
                             int64_t *rank);

/**
 * @brief Converts an image from one color space to another. For more info, plz
 * check: http://wiki.enflame.cn/pages/viewpage.action?pageId=130547478
 *
 * @param input input image.
 * @param output output image of the same size and depth as src.
 * @param code color space convertion code.
 * @param output_channel number of channels in the destination image; if the
 * parameter is 0, the number of the channels is derived automatically from src
 * and code. ITS UNNECESARY BUT DEFINED BY OPENCV.
 * @param stream Tops stream.
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopCvtColor(const topsopTensorHandle_t input,
                                            topsopTensorHandle_t output,
                                            topsopColorConversionCodes_t code,
                                            topsopScalar_t output_channel,
                                            topsStream_t stream);

/**
 * @brief check whether color space convert code is supported.
 * @param input The input tensor.
 * @param output The output tensor.
 * @param code color space convertion code.
 * @return bool
 */
bool TOPSOP_EXPORT topsopCvtColorIsSupported(topsopColorConversionCodes_t code,
                                             const topsopTensorHandle_t input);

/**
 * @brief Get output dims of current tensor of Color Space Convertion operator
 * @param input The input tensor.
 * @param dims The dims of output tensor.
 * @param rank The rank of output tensor.
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopCvtColorGetOutputDim(topsopColorConversionCodes_t code,
                           const topsopTensorHandle_t input,
                           int64_t *dims,
                           int64_t *rank);

/**
 * @brief This function fills elements of input tensor with value
 *        where mask is True.
 *
 * @param output Result tensor
 * @param mask  Mask tensor
 * @param input Input tensor
 * @param value the value to fill in with
 * @param alpha scaling factors (in host memory) used to blend the
 *              layer output value with prior value in the destination tensor
 * @param beta scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopMaskedFill(topsopTensorHandle_t output,
                                              const topsopTensorHandle_t input,
                                              const topsopTensorHandle_t mask,
                                              topsopScalar_t value,
                                              topsopScalar_t alpha,
                                              topsopScalar_t beta,
                                              topsStream_t stream);

/**
 * @brief check whether current masked fill operator support or not
 *
 * @param input Input tensor
 * @param mask  Mask tensor
 * @param alpha scaling factors (in host memory) used to blend the
 *              layer output value with prior value in the destination tensor
 * @param beta scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor
 * @return bool
 */
bool TOPSOP_EXPORT topsopMaskedFillIsSupported(const topsopTensorHandle_t input,
                                               const topsopTensorHandle_t mask,
                                               topsopScalar_t alpha,
                                               topsopScalar_t beta);

/**
 * @brief get output dims of current tensor masked fill operator
 *
 * @param input Input tensor
 * @param mask  Mask tensor
 * @param dims The dims of output tensor
 * @param rank The rank of output tensor
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopMaskedFillGetOutputDim(const topsopTensorHandle_t input,
                             const topsopTensorHandle_t mask,
                             int64_t *dims,
                             int64_t *rank);

/**
 * @brief This function performs the function that put values from the tensor
 *        values into the tensor self using the indices specified in indices
 *
 * @param self: index put input and result
 * @param indices: tensors arrray used to index into self
 * @param indices_num: indice number
 * @param values: tensor of same dtype as self
 * @param dim_start: indice dim start index in value
 * @param accumulate: if accumulate is True, the elements in values
 *              are added to self. If accumulate is False, the behavior
 *              is undefined if indices contain duplicate elements.
 * @param alpha scaling factors (in host memory) used to blend the
 *              layer output value with prior value in the destination tensor
 * @param beta scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor
 * @param stream:Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopIndexPut(topsopTensorHandle_t self,
                                            const topsopTensorHandle_t *indices,
                                            const int64_t indices_num,
                                            const topsopTensorHandle_t values,
                                            int64_t dim_start,
                                            bool accumulate,
                                            topsopScalar_t alpha,
                                            topsopScalar_t beta,
                                            topsStream_t stream);

/**
 * @brief check whether index put operator support or not
 *
 * @param self: index put input and result
 * @param indices: tensors arrray used to index into self
 * @param indices_num: indice number
 * @param values: tensor of same dtype as self
 * @param dim_start: indice dim start index in value
 * @param accumulate: if accumulate is True, the elements in values
 *              are added to self. If accumulate is False, the behavior
 *              is undefined if indices contain duplicate elements.
 * @param alpha scaling factors (in host memory) used to blend the
 *              layer output value with prior value in the destination tensor
 * @param beta scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor
 * @return bool
 */
bool TOPSOP_EXPORT
topsopIndexPutIsSupported(topsopTensorHandle_t self,
                          const topsopTensorHandle_t *indices,
                          const int64_t indices_num,
                          const topsopTensorHandle_t values,
                          int64_t dim_start,
                          bool accumulate,
                          topsopScalar_t alpha,
                          topsopScalar_t beta);

/**
 * @brief get output dims of current tensor index put operator
 *
 * @param self: index put input
 * @param dims The dims of output tensor
 * @param rank The rank of output tensor
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopIndexPutGetOutputDim(
    topsopTensorHandle_t self, int64_t *dims, int64_t *rank);

/**
 * @brief This function performs the function that
 *   d_out = a_in + alpha_in * b_in * c_in
 *
 * @param d_out: the output tensor
 * @param a_in: the tensor to be added
 * @param alpha_in: multiplier for b_in ∗ c_in
 * @param b_in: the tensor to be multiplied
 * @param c_in: the tensor to be multiplied
 * @param stream:Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopAddcmul(topsopTensorHandle_t d_out,
                                           const topsopTensorHandle_t a_in,
                                           const topsopTensorHandle_t b_in,
                                           const topsopTensorHandle_t c_in,
                                           const topsopScalar_t alpha_in,
                                           topsStream_t stream);

/**
 * @brief This function check whether topsopAddcmul support or not
 *
 * @param a_in: the tensor to be added
 * @param b_in: the tensor to be multiplied
 * @param c_in: the tensor to be multiplied
 * @param alpha_in: multiplier for b_in * c_in
 * @return bool
 */
bool TOPSOP_EXPORT topsopAddcmulIsSupported(const topsopTensorHandle_t a_in,
                                            const topsopTensorHandle_t b_in,
                                            const topsopTensorHandle_t c_in,
                                            const topsopScalar_t alpha_in);

/**
 * @brief get output dims of current topsopAddcmul
 *
 * @param a_in: the tensor to be added
 * @param b_in: the tensor to be multiplied
 * @param c_in: the tensor to be multiplied
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopAddcmulGetOutputDim(const topsopTensorHandle_t a_in,
                          const topsopTensorHandle_t b_in,
                          const topsopTensorHandle_t c_in,
                          int64_t *dims,
                          int64_t *rank);

/**
 * @brief This function performs the function that
 *   d_out = a_in + alpha_in * b_in / c_in
 *
 * @param d_out: the output tensor
 * @param a_in: the tensor to be added
 * @param b_in: the numerator tensor
 * @param c_in: the denominator tensor
 * @param alpha_in: multiplier for b_in / c_in
 * @param stream:Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopAddcdiv(topsopTensorHandle_t d_out,
                                           const topsopTensorHandle_t a_in,
                                           const topsopTensorHandle_t b_in,
                                           const topsopTensorHandle_t c_in,
                                           const topsopScalar_t alpha_in,
                                           topsStream_t stream);

/**
 * @brief This function check whether topsopAddcdiv support or not
 *
 * @param a_in: the tensor to be added
 * @param b_in: the numerator tensor
 * @param c_in: the denominator tensor
 * @param alpha_in: multiplier for b_in / c_in
 * @return bool
 */
bool TOPSOP_EXPORT topsopAddcdivIsSupported(const topsopTensorHandle_t a_in,
                                            const topsopTensorHandle_t b_in,
                                            const topsopTensorHandle_t c_in,
                                            const topsopScalar_t alpha_in);

/**
 * @brief get output dims of current topsopAddcdiv
 *
 * @param a_in: the tensor to be added
 * @param b_in: the numerator tensor
 * @param c_in: the denominator tensor
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopAddcdivGetOutputDim(const topsopTensorHandle_t a_in,
                          const topsopTensorHandle_t b_in,
                          const topsopTensorHandle_t c_in,
                          int64_t *dims,
                          int64_t *rank);

/**
 * @brief This function performs the function that
 *   b_out = a_in > threshold_in ? b_in : value_in
 *
 * @param c_out: the output tensor
 * @param a_in: the input tensor
 * @param b_in: the input tensor
 * @param threshold_in: The value to threshold at
 * @param value_in: The value to replace with
 * @param stream:Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopThreshold(topsopTensorHandle_t c_out,
                                             const topsopTensorHandle_t a_in,
                                             const topsopTensorHandle_t b_in,
                                             const topsopScalar_t threshold_in,
                                             const topsopScalar_t value_in,
                                             topsStream_t stream);

/**
 * @brief This function check whether topsopThreshold support or not.
 *
 * @param a_in: the input tensor
 * @param b_in: the input tensor
 * @param threshold_in: The value to threshold at
 * @param value_in: The value to replace with
 * @return bool
 */
bool TOPSOP_EXPORT topsopThresholdIsSupported(const topsopTensorHandle_t a_in,
                                              const topsopTensorHandle_t b_in,
                                              const topsopScalar_t threshold_in,
                                              const topsopScalar_t value_in);

/**
 * @brief get output dims of current topsopThreshold
 *
 * @param a_in: the input tensor
 * @param b_in: the input tensor
 * @param dims: The dims of output tensor.
 * @param rank: The ranks of output tensor.
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopThresholdGetOutputDim(const topsopTensorHandle_t a_in,
                            const topsopTensorHandle_t b_in,
                            int64_t *dims,
                            int64_t *rank);

/**
 * @brief This function returns the product of elements of input
 *        in the dimension dim.
 *
 * @param output: The output tensor.
 * @param input: The input tensor.
 * @param dims_in: The dimensions to reduce.
 * @param keepdims: Whether the output tensor has dim retained or no.
 * @param stream: Tops stream.
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsflameProd(topsopTensorHandle_t output,
                                           const topsopTensorHandle_t input,
                                           const topsopSize_t dims_in,
                                           bool keepdims,
                                           topsStream_t stream);

/**
 * @brief This function check whether topsflameProd support or not.
 *
 * @param input: The input tensor.
 * @param dims_in: The dimensions to reduce.
 * @param keepdims: Whether the output tensor has dim retained or no.
 * @return bool
 */
bool TOPSOP_EXPORT topsflameProdIsSupported(const topsopTensorHandle_t output,
                                            const topsopTensorHandle_t input,
                                            const topsopSize_t dims_in,
                                            const bool keepdim);

/**
 * @brief get output dims of current topsflameProd.
 *
 * @param input:The input tensor.
 * @param dims_in: The dimensions to reduce.
 * @param keepdims: Whether the output tensor has dim retained or no.
 * @param dims: The dims of output tensor.
 * @param rank: The ranks of output tensor.
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsflameProdGetOutputDim(const topsopTensorHandle_t input,
                          const topsopSize_t dims_in,
                          const bool keepdim,
                          int64_t *dims,
                          int64_t *rank);

/**
 * @brief This function return a new tensor which indexes the self tensor
 *        using the entries in indices.
 *
 * @param output: The output tensor.
 * @param self: The input tensor.
 * @param indices: The tensor list contianing the indices to index.
 * @param indices_num: The number of index tensor.
 * @param indice_start_dim: The dimension in which the first index tensor
 *                          applying for.
 * @param stream:Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopIndex(topsopTensorHandle_t output,
                                         const topsopTensorHandle_t self,
                                         const topsopTensorHandle_t *indices,
                                         const int64_t indices_num,
                                         const int64_t indice_start_dim,
                                         topsStream_t stream);

/**
 * @brief This function check whether topsopIndex is supported or not.
 *
 * @param self: The input tensor.
 * @param indices: The tensor list contianing the indices to index.
 * @param indices_num: The number of index tensor.
 * @param indice_start_dim: The dimension in which the first index tensor
 *                          applying for.
 * @return bool
 */
bool TOPSOP_EXPORT topsopIndexIsSupported(const topsopTensorHandle_t self,
                                          const topsopTensorHandle_t *indices,
                                          const int64_t indices_num,
                                          const int64_t indice_start_dim);

/**
 * @brief This function get the shape of output tensor of topsopIndex.
 *
 * @param self: The input tensor.
 * @param indices: The tensor list contianing the indices to index.
 * @param indices_num: The number of index tensor.
 * @param indice_start_dim: The dimension in which the first index tensor
 *                          applying for.
 * @param dims: The dims of output tensor.
 * @param rank: The ranks of output tensor.
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopIndexGetOutputDim(const topsopTensorHandle_t self,
                        const topsopTensorHandle_t *indices,
                        const int64_t indices_num,
                        const int64_t indice_start_dim,
                        int64_t *dims,
                        int64_t *rank);

/**
 * @brief This function returns True if any element in the row evaluate to
 *        True and False otherwise, for each row of input in the given
 *        dimension dim.
 *
 * @param output: The output tensor.
 * @param input: The input tensor.
 * @param dim: The dimension to reduce.
 * @param keepdim: Whether the output tensor has dim retained or not.
 * @param stream: Tops stream.
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsflameAny(topsopTensorHandle_t output,
                                          const topsopTensorHandle_t input,
                                          const int64_t dim,
                                          const bool keepdim,
                                          const topsStream_t stream);

/**
 * @brief This function check whether topsopAny support or not.
 *
 * @param input: The input tensor.
 * @param dim: The dimension to do the operation over, target dim.
 * @return bool
 */
bool TOPSOP_EXPORT topsflameAnyIsSupported(const topsopTensorHandle_t input,
                                           const int64_t dim);

/**
 * @brief Get output dims of current topsopAny
 *
 * @param input: The input tensor.
 * @param dim: The dimension to do the operation over, target dim.
 * @param keepdim: Whether the output tensor has dim retained or not.
 * @param dims: The dims of output tensor.
 * @param rank: The ranks of output tensor.
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsflameAnyGetOutputDim(const topsopTensorHandle_t input,
                         const int64_t dim,
                         const bool keepdim,
                         int64_t *dims,
                         int64_t *rank);

/**
 * @brief This function returns True if all element in the row evaluate to
 *        True and False otherwise, for each row of input in the given
 *        dimension dim.
 *
 * @param output: The output tensor.
 * @param input: The input tensor.
 * @param dim: The dimension to reduce.
 * @param keepdim: Whether the output tensor has dim retained or not.
 * @param stream: Tops stream.
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsflameAll(topsopTensorHandle_t output,
                                          const topsopTensorHandle_t input,
                                          const int64_t dim,
                                          const bool keepdim,
                                          const topsStream_t stream);

/**
 * @brief This function check whether topsopAll support or not.
 *
 * @param input: The input tensor.
 * @param dim: The dimension to do the operation over, target dim.
 * @return bool
 */
bool TOPSOP_EXPORT topsflameAllIsSupported(const topsopTensorHandle_t input,
                                           const int64_t dim);

/**
 * @brief Get output dims of current topsopAll
 *
 * @param input: The input tensor.
 * @param dim: The dimension to do the operation over, target dim.
 * @param keepdim: Whether the output tensor has dim retained or not.
 * @param dims: The dims of output tensor.
 * @param rank: The ranks of output tensor.
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsflameAllGetOutputDim(const topsopTensorHandle_t input,
                         const int64_t dim,
                         const bool keepdim,
                         int64_t *dims,
                         int64_t *rank);

/**
 * @brief This function is used to eliminate non-consecutive duplicate values.
 * @param output: the output tensor: list of unique scalar elements.
 * @param inverse_indices: (optional) if return_inverse is True, there will be
 *                         an additional returned tensor (same shape as input)
 *                         representing the indices for where elements in the
 *                         original input map to in the output; otherwise, this
 *                         function will only return a single tensor.
 * @param counts: (optional) if return_counts is True, there will be an
 * additional returned tensor (same shape as output or output.size(dim), if dim
 *                was specified) representing the number of occurrences for each
 *                unique value or tensor.
 * @param input: The input tensor.
 * @param sorted Whether to sort the unique elements in ascending order before
 *               returning as output. Default: true
 * @param return_inverse Whether to also return the indices for where elements
 *                       in the original input ended up in the returned unique
 *                       list. Default: false
 * @param return_counts Whether to also return the counts for each unique
 * element Default: false
 * @param alpha scaling factors (in host memory) used to blend the
 *              layer output value with prior value in the destination tensor
 * @param beta scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor
 * @param stream:Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopUnique2(topsopTensorHandle_t output,
                                           topsopTensorHandle_t inverse_indices,
                                           topsopTensorHandle_t counts,
                                           const topsopTensorHandle_t input,
                                           const bool sorted,
                                           const bool return_inverse,
                                           const bool return_counts,
                                           const topsopScalar_t alpha,
                                           const topsopScalar_t beta,
                                           const topsStream_t stream);

/**
 * @brief This function check whether topsopUnique is supported or not.
 * @param input: The input tensor.
 * @param sorted Whether to sort the unique elements in ascending order before
 *               returning as output. Default: true
 * @param return_inverse Whether to also return the indices for where elements
 *                       in the original input ended up in the returned unique
 *                       list. Default: false
 * @param return_counts Whether to also return the counts for each unique
 * element Default: false
 * @param alpha Scaling factors (in host memory) used to blend the
 *              layer output value with prior value in the destination tensor.
 * @param beta Scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor.
 * @return bool
 */
bool TOPSOP_EXPORT topsopUnique2IsSupported(const topsopTensorHandle_t input,
                                            const bool sorted,
                                            const bool return_inverse,
                                            const bool return_counts,
                                            const topsopScalar_t alpha,
                                            const topsopScalar_t beta);

/**
 * @brief This function get the shape of output tensor of topsopUnique.
 *
 * @param input: The input tensor.
 * @param dims: The dims of output tensor.
 * @param rank: The ranks of output tensor.
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopUnique2GetOutputDim(const topsopTensorHandle_t input,
                          int64_t *dims,
                          int64_t *rank,
                          const topsStream_t stream);

/**
 * @brief This function just demo for topscc, do not call it.
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsOpAddmulDemo(const topsopTensorHandle_t a_in,
                                              const topsopTensorHandle_t b_in,
                                              topsopTensorHandle_t c_out,
                                              topsStream_t stream);

/**
 * @brief This function returns the cumulative sum of elements of input
 *        in the dimension dim.
 *
 * @param output: The output tensor.
 * @param input: The input tensor.
 * @param dims: The dimension to do the operation over, target dim.
 * @param stream: Tops stream.
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsflameCumsum(topsopTensorHandle_t output,
                                             const topsopTensorHandle_t input,
                                             const topsopSize_t dims,
                                             topsStream_t stream);

/**
 * @brief This function check whether topsflameCumsum support or not.
 *
 * @param input: The input tensor.
 * @param input: The output tensor, contain dtype.
 * @param dims: The dimension to do the operation over, target dim.
 * @return bool
 */
bool TOPSOP_EXPORT topsflameCumsumIsSupported(const topsopTensorHandle_t input,
                                              const topsopTensorHandle_t output,
                                              const topsopSize_t dims);

/**
 * @brief get output dims of current topsflameCumsum
 *
 * @param input: the input tensor.
 * @param dims: The dims of output tensor.
 * @param rank: The ranks of output tensor.
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsflameCumsumGetOutputDim(
    const topsopTensorHandle_t input, int64_t *dims, int64_t *rank);

/**
 * @brief dnn optensor operator
 *
 * @param out The output tensor
 * @param lhs Input lhs tensor
 * @param rhs Input rhs tensor
 * @param opType an enumerant to specify the tensor pointwise math operation.
 * @param nanOpt an enumerated type used to indicate if a given routine
 * should propagate Nan numbers
 * @param alpha1 The multiplier for lhs tensor
 * @param alpha2 The multiplier for rhs tensor
 * @param beta The multiplier for output tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopDnnOptensor(topsopTensorHandle_t out,
                  const topsopTensorHandle_t lhs,
                  const topsopTensorHandle_t rhs,
                  const topsopElementwiseOpType_t opType,
                  const topsopNanPropagation_t nanOpt,
                  const topsopScalar_t alpha1,
                  const topsopScalar_t alpha2,
                  const topsopScalar_t beta,
                  topsStream_t stream);
/**
 * @brief check whether current dnn optensor operator support or not
 *
 * @param lhs Input lhs tensor
 * @param rhs Input rhs tensor
 * @return bool
 */
bool TOPSOP_EXPORT
topsopDnnOptensorIsSupported(const topsopTensorHandle_t lhs,
                             const topsopTensorHandle_t rhs,
                             const topsopElementwiseOpType_t opType,
                             const topsopNanPropagation_t nanOpt);
/**
 * @brief get output dims of current tensor dnn optensor operator
 *
 * @param lhs Input lhs tensor
 * @param rhs Input rhs tensor
 * @param dims The dims of output tensor
 * @param rank The rank of output tensor
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopDnnOptensorGetOutputDim(const topsopTensorHandle_t lhs,
                              const topsopTensorHandle_t rhs,
                              int64_t *dims,
                              int64_t *rank);
/**
 * @brief This function performs the hardswish function.
 *
 * @param output Result tensor
 * @param input Input tensor
 * @param reluNanOpt Enumerant to specify the Nan propagation mode
 * @param alpha scaling factors (in host memory) used to blend the
 *              layer output value with prior value in the destination tensor
 * @param beta scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopHardSwishForward(topsopTensorHandle_t output,
                       const topsopTensorHandle_t input,
                       topsopNanPropagation_t reluNanOpt,
                       topsopScalar_t alpha,
                       topsopScalar_t beta,
                       topsStream_t stream);
/**
 * @brief check whether current hardswish operator support or not
 *
 * @param input Input tensor
 * @param reluNanOpt Enumerant to specify the Nan propagation mode
 * @param alpha scaling factors (in host memory) used to blend the
 *              layer output value with prior value in the destination tensor
 * @param beta scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor
 * @return bool
 */
bool TOPSOP_EXPORT
topsopHardSwishForwardIsSupported(const topsopTensorHandle_t input,
                                  topsopNanPropagation_t reluNanOpt,
                                  topsopScalar_t alpha,
                                  topsopScalar_t beta);
/**
 * @brief get output dims of current tensor swish forward operator
 *
 * @param input Input tensor
 * @param dims The dims of output tensor
 * @param rank The rank of output tensor
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopHardSwishForwardGetOutputDim(
    const topsopTensorHandle_t input, int64_t *dims, int64_t *rank);

/**
 * @brief This function computes the gradient of the hardswish function.
 *
 * @param grad_input Output differential tensor
 * @param grad_output Input differential tensor
 * @param in_out Input tensor
 * @param reluNanOpt Enumerant to specify the Nan propagation mode
 * @param alpha scaling factors (in host memory) used to blend the
 *              layer output value with prior value in the destination tensor
 * @param beta scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT
topsopHardSwishBackward(topsopTensorHandle_t grad_input,
                        const topsopTensorHandle_t grad_output,
                        const topsopTensorHandle_t in_out,
                        topsopNanPropagation_t reluNanOpt,
                        topsopScalar_t alpha,
                        topsopScalar_t beta,
                        topsStream_t stream);
/**
 * @brief check whether current hardswish backward operator support or not
 *
 * @param grad_output Input differential tensor
 * @param in_out Input tensor
 * @param reluNanOpt Enumerant to specify the Nan propagation mode
 * @param alpha scaling factors (in host memory) used to blend the
 *              layer output value with prior value in the destination tensor
 * @param beta scaling factors (in host memory) used to blend the
 *             layer output value with prior value in the destination tensor
 * @return bool
 */
bool TOPSOP_EXPORT
topsopHardSwishBackwardIsSupported(const topsopTensorHandle_t grad_output,
                                   const topsopTensorHandle_t in_out,
                                   topsopNanPropagation_t reluNanOpt,
                                   topsopScalar_t alpha,
                                   topsopScalar_t beta);
/**
 * @brief get output dims of current tensor hardswish backward operator
 *
 * @param grad_output Input differential tensor
 * @param in_out Input tensor
 * @param dims The dims of output tensor
 * @param rank The rank of output tensor
 * @return topsopStatus_t
 */

topsopStatus_t TOPSOP_EXPORT
topsopHardSwishBackwardGetOutputDim(const topsopTensorHandle_t grad_output,
                                    const topsopTensorHandle_t in_out,
                                    int64_t *dims,
                                    int64_t *rank);
/**
 * @brief This function performs the isfinite function.
 *
 * @param output Result tensor
 * @param input Input tensor
 * @param stream Tops stream
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopIsfinite(topsopTensorHandle_t output,
                                            const topsopTensorHandle_t input,
                                            topsStream_t stream);
/**
 * @brief check whether current isfinite operator support or not
 *
 * @param output output tensor
 * @return bool
 */
bool TOPSOP_EXPORT topsopIsfiniteIsSupported(const topsopTensorHandle_t output);
/**
 * @brief get output dims of current tensor swish forward operator
 *
 * @param input Input tensor
 * @param dims The dims of output tensor
 * @param rank The rank of output tensor
 * @return topsopStatus_t
 */
topsopStatus_t TOPSOP_EXPORT topsopIsfiniteGetOutputDim(
    const topsopTensorHandle_t input, int64_t *dims, int64_t *rank);

#if defined(__cplusplus)
}
#endif

#endif /* TOPSOP_OPS_H_ */  // NOLINT

// Doxygen end group topsop_ops.h
/** @} */
