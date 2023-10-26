/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <map>
#include <unordered_map>
#include <vector>

#include "dtu/hlir_builder/hlir_builder.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/utils/blank.h"
#include "paddle/utils/variant.h"

namespace backend {

using GcuOp = builder::Op;
using GcuOpPtr = std::shared_ptr<GcuOp>;
using GcuPrimitiveType = builder::PrimitiveType;
using GcuType = builder::Type;
using GcuShape = std::vector<int64_t>;
using GcuBuilder = builder::Builder;
using GcuBuilderPtr = std::shared_ptr<builder::Builder>;
using GcuGraphPtr = std::shared_ptr<hlir::Module>;

// ATTR DEFINE
const char* const kAttrOpOutVarName = "_op_out_var_name";

namespace RunningMode {
const char* const SERIAL = "serial";
const char* const ADAPTIVE = "adaptive";
const char* const FORCE_SERIAL = "force_serial";
}  // namespace RunningMode

using TensorNameMap = std::map<std::string, std::vector<std::string>>;
using GcuAttribute = paddle::variant<paddle::blank,
                                     int,
                                     float,
                                     std::string,
                                     std::vector<int>,
                                     std::vector<float>,
                                     std::vector<std::string>,
                                     bool,
                                     std::vector<bool>,
                                     int64_t,
                                     std::vector<int64_t>,
                                     std::vector<double>,
                                     double>;
using GcuAttributeMap = std::unordered_map<std::string, GcuAttribute>;

enum class Layout : int { NCHW, NHWC, HWCN, NCDHW, NDHWC, DHWCN };

}  // namespace backend
