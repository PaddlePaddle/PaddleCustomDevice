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
//
// Part of the following code in this file is from
//     https://gitee.com/ascend/pytorch/blob/master/torch_npu/csrc/framework/FormatHelper.h
//     Git commit hash: 3fc20b8b30d18e2da3f73888ae47effe5a3d1ca2 (v1.5.0)
// Retain the following license from the original files:
//
// Copyright (c) 2020 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "acl/acl.h"
#include "glog/logging.h"
#include "paddle/extension.h"
#include "paddle/phi/extension.h"

aclDataType ConvertToNpuDtype(phi::DataType dtype);
aclFormat ConvertToNpuFormat(phi::DataLayout layout);

using FormatShape = std::vector<int64_t>;

class FormatHelper {
 public:
  static FormatShape GetStorageShape(const aclFormat storage_format,
                                     const FormatShape origin_dims);

 private:
  static char* GetFormatName(const aclFormat& format);

 private:
  using shapeInfer = std::function<FormatShape(FormatShape dims)>;
  typedef struct FormatInfo_ {
    aclFormat format = ACL_FORMAT_ND;
    aclFormat baseFormat = ACL_FORMAT_ND;
    shapeInfer func = nullptr;
    char formatName[30] = {0};
    bool isPadded = false;
  } FormatInfo;
  static std::unordered_map<aclFormat, FormatInfo> info;
};
