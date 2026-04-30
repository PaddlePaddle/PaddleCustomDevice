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

#pragma once

#include "common/gcu_funcs.h"
#include "kernels/funcs/topsaten_op_launch.h"

namespace custom_kernel {

DenseTensor MaybeCreateOrTrans(
    const phi::CustomContext& dev_ctx,
    const DenseTensor& src,
    const std::unordered_map<DataType, DataType>& tans_map,
    bool need_cast = true);

DenseTensor MaybeCreateOrTrans64To32bits(const phi::CustomContext& dev_ctx,
                                         const DenseTensor& src,
                                         bool need_cast = true);

DenseTensor MaybeCreateOrTransFp16ToFp32(const phi::CustomContext& dev_ctx,
                                         const DenseTensor& src,
                                         bool need_cast = true);

void MaybeTransResult(const phi::CustomContext& dev_ctx,
                      const DenseTensor& result,
                      DenseTensor* dst);

void Broadcast(const phi::CustomContext& dev_ctx,
               const DenseTensor& src,
               DenseTensor* dst);

DenseTensor Broadcast(const phi::CustomContext& dev_ctx,
                      const DenseTensor& src,
                      const std::vector<int64_t>& output_shapes);

void Cast(const phi::CustomContext& dev_ctx,
          const DenseTensor& x,
          const DataType& dtype,
          DenseTensor* out);

DenseTensor Cast(const phi::CustomContext& dev_ctx,
                 const DenseTensor& x,
                 const DataType& dtype);

DenseTensor CastOrCopyToPinnedMemory(const phi::CustomContext& dev_ctx,
                                     const DenseTensor& x,
                                     const DataType& dtype);

DenseTensor ReshapeWithoutCopy(const DenseTensor& src,
                               const std::vector<int64_t>& out_shapes);

DenseTensor TensorEmpty(const phi::CustomContext& dev_ctx,
                        const DenseTensorMeta& meta);

DenseTensor TensorOnes(const phi::CustomContext& dev_ctx,
                       const DenseTensorMeta& meta);

DenseTensor TensorZeros(const phi::CustomContext& dev_ctx,
                        const DenseTensorMeta& meta);

// meta reuse ops
DenseTensor Add(const phi::CustomContext& dev_ctx,
                const DenseTensor& x,
                const DenseTensor& y,
                const DenseTensorMeta& out_meta);

DenseTensor Add(const phi::CustomContext& dev_ctx,
                const DenseTensor& x,
                const DenseTensor& y);

DenseTensor Subtract(const phi::CustomContext& dev_ctx,
                     const DenseTensor& x,
                     const DenseTensor& y,
                     const DenseTensorMeta& out_meta);

DenseTensor Subtract(const phi::CustomContext& dev_ctx,
                     const DenseTensor& x,
                     const DenseTensor& y);

void SliceBase(const phi::CustomContext& dev_ctx,
               const DenseTensor& x,
               const std::vector<int64_t>& axes,
               const std::vector<int64_t>& starts,
               DenseTensor* out);

}  // namespace custom_kernel
