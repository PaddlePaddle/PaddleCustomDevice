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

#pragma once

#include "acl/acl.h"
#include "glog/logging.h"
#include "paddle/extension.h"
#include "paddle/phi/extension.h"

template <typename T>
std::string GetVectorString(const std::vector<T>& shape);

std::string GetDataBufferString(const aclDataBuffer* buf);

std::string GetTensorDescString(const aclTensorDesc* desc);

std::string GetOpDescString(std::vector<aclTensorDesc*> descs,
                            const std::string msg);

std::string GetOpInfoString(std::vector<aclTensorDesc*> descs,
                            std::vector<aclDataBuffer*> buffs,
                            const std::string msg);
