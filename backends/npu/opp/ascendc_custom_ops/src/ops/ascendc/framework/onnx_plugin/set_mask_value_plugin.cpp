/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "register/register.h"

namespace domi {
    // Onnx ParseParams
    Status ParseParamSetMaskValue(const ge::Operator &opSrc, ge::Operator &opDest)
    {
        return SUCCESS;
    }

    static std::vector<ge::AscendString> g_supportedOnnxVersion ({
        "ai.onnx::8::SetMaskValue",
        "ai.onnx::9::SetMaskValue",
        "ai.onnx::10::SetMaskValue",
        "ai.onnx::11::SetMaskValue",
        "ai.onnx::12::SetMaskValue",
        "ai.onnx::13::SetMaskValue",
        "ai.onnx::14::SetMaskValue",
        "ai.onnx::15::SetMaskValue",
        "ai.onnx::16::SetMaskValue",
        "ai.onnx::17::SetMaskValue",
        "ai.onnx::18::SetMaskValue",
    });

    // register SetMaskValue op info to GE
    REGISTER_CUSTOM_OP("SetMaskValue")                    // Set the registration name of operator
        .FrameworkType(ONNX)                          // Operator name with the original framework
        .OriginOpType(g_supportedOnnxVersion)   // Set the original frame type of the operator
        .ParseParamsByOperatorFn(ParseParamSetMaskValue);
} // namespace domi
