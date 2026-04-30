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

namespace custom_kernel {
namespace layout_trans {
const std::vector<int64_t> kNCHW_to_NHWC = {0, 2, 3, 1};
const std::vector<int64_t> kNHWC_to_NCHW = {0, 3, 1, 2};
}  // namespace layout_trans

bool EnableTransposeOptimize();

void SetLayout(DenseTensor& tensor,  // NOLINT
               const common::DataLayout& layout);

void Transpose(const phi::CustomContext& dev_ctx,
               const DenseTensor& x,
               const std::vector<int64_t>& axis,
               DenseTensor* out);

DenseTensor Transpose(const phi::CustomContext& dev_ctx,
                      const DenseTensor& x,
                      const std::vector<int64_t>& axis);

bool DataPdCustomNHWC(const DenseTensor& tensor);

bool DataPdCustomNHWC(const std::vector<DenseTensor>& tensors);

// ////////////////  Permuted funcs ////////////////
void PermutedShapeWithcontiguousStrides(
    DenseTensor& tensor,  // NOLINT
    const std::vector<int64_t>& permutation,
    const common::DataLayout& layout = common::DataLayout::kNCHW);

void RecoverPdCustomNHWCMeta(DenseTensor& tensor);  // NOLINT

void PermutedStridesWithoutShape(
    DenseTensor& tensor,  // NOLINT
    const std::vector<int64_t>& shape_perm,
    const std::vector<int64_t>& strides_perm,
    const common::DataLayout& layout = common::DataLayout::kNCHW);

void PermutedShapeAndStrides(
    DenseTensor& tensor,  // NOLINT
    const std::vector<int64_t>& permutation,
    const common::DataLayout& layout = common::DataLayout::kNCHW);

// ////////////////  Transpose funcs ////////////////
DenseTensor NCHWTransToPdOriginNHWC(const phi::CustomContext& dev_ctx,
                                    const DenseTensor& x);

DenseTensor NCHWTransToPdCustomNHWC(const phi::CustomContext& dev_ctx,
                                    const DenseTensor& x);

DenseTensor NCHWTransToAtenNHWC(const phi::CustomContext& dev_ctx,
                                const DenseTensor& x);

DenseTensor PdCustomNHWCTransToNCHW(const phi::CustomContext& dev_ctx,
                                    const DenseTensor& x);

DenseTensor PdOriginNHWCTransToNCHW(const phi::CustomContext& dev_ctx,
                                    const DenseTensor& x);

// ////////////////  Represent funcs ////////////////
DenseTensor NoNeedTransNCHWRepresentAsOriginNHWC(const DenseTensor& x);

void PdCustomNHWCRepresentAsAtenNHWC(DenseTensor& x,  // NOLINT
                                     bool weight_or_output = false);

void AtenNHWCRepresentAsPdCustomNHWC(DenseTensor& x,  // NOLINT
                                     bool raw_output = false);

void OriginNHWCRepresentAsAtenNHWC(DenseTensor& x);  // NOLINT

void AtenNHWCRepresentAsOriginNHWC(DenseTensor& x);  // NOLINT

void PdCustomNHWCRepresentAsOriginNHWC(DenseTensor& x,  // NOLINT
                                       bool raw_output = false);

void OriginNHWCRepresentAsPdCustomNHWC(DenseTensor& x);  // NOLINT

void RepresentPdCustomNHWC(DenseTensor& x);  // NOLINT

}  // namespace custom_kernel
