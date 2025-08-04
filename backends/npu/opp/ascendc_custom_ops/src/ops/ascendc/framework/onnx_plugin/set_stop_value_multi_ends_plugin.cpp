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
    Status ParseParamSetStopValueMultiEnds(const ge::Operator &opSrc, ge::Operator &opDest)
    {
        return SUCCESS;
    }

    static std::vector<ge::AscendString> g_supportedOnnxVersion ({
        "ai.onnx::8::SetStopValueMultiEnds",
        "ai.onnx::9::SetStopValueMultiEnds",
        "ai.onnx::10::SetStopValueMultiEnds",
        "ai.onnx::11::SetStopValueMultiEnds",
        "ai.onnx::12::SetStopValueMultiEnds",
        "ai.onnx::13::SetStopValueMultiEnds",
        "ai.onnx::14::SetStopValueMultiEnds",
        "ai.onnx::15::SetStopValueMultiEnds",
        "ai.onnx::16::SetStopValueMultiEnds",
        "ai.onnx::17::SetStopValueMultiEnds",
        "ai.onnx::18::SetStopValueMultiEnds",
    });

    // register SetStopValueMultiEnds op info to GE
    REGISTER_CUSTOM_OP("SetStopValueMultiEnds")                    // Set the registration name of operator
        .FrameworkType(ONNX)                          // Operator name with the original framework
        .OriginOpType(g_supportedOnnxVersion)   // Set the original frame type of the operator
        .ParseParamsByOperatorFn(ParseParamSetStopValueMultiEnds);
} // namespace domi
