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
    Status ParseParamSetValueByFlagsAndIdx(const ge::Operator &opSrc, ge::Operator &opDest)
    {
        return SUCCESS;
    }

    static std::vector<ge::AscendString> g_supportedOnnxVersion ({
        "ai.onnx::8::SetValueByFlagsAndIdx",
        "ai.onnx::9::SetValueByFlagsAndIdx",
        "ai.onnx::10::SetValueByFlagsAndIdx",
        "ai.onnx::11::SetValueByFlagsAndIdx",
        "ai.onnx::12::SetValueByFlagsAndIdx",
        "ai.onnx::13::SetValueByFlagsAndIdx",
        "ai.onnx::14::SetValueByFlagsAndIdx",
        "ai.onnx::15::SetValueByFlagsAndIdx",
        "ai.onnx::16::SetValueByFlagsAndIdx",
        "ai.onnx::17::SetValueByFlagsAndIdx",
        "ai.onnx::18::SetValueByFlagsAndIdx",
    });

    // register SetValueByFlagsAndIdx op info to GE
    REGISTER_CUSTOM_OP("SetValueByFlagsAndIdx")                    // Set the registration name of operator
        .FrameworkType(ONNX)                          // Operator name with the original framework
        .OriginOpType(g_supportedOnnxVersion)   // Set the original frame type of the operator
        .ParseParamsByOperatorFn(ParseParamSetValueByFlagsAndIdx);
} // namespace domi
