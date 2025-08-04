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
    Status ParseParamTokenPenaltyMultiScores(const ge::Operator &opSrc, ge::Operator &opDest)
    {
        return SUCCESS;
    }

    static std::vector<ge::AscendString> g_supportedOnnxVersion ({
        "ai.onnx::8::TokenPenaltyMultiScores",
        "ai.onnx::9::TokenPenaltyMultiScores",
        "ai.onnx::10::TokenPenaltyMultiScores",
        "ai.onnx::11::TokenPenaltyMultiScores",
        "ai.onnx::12::TokenPenaltyMultiScores",
        "ai.onnx::13::TokenPenaltyMultiScores",
        "ai.onnx::14::TokenPenaltyMultiScores",
        "ai.onnx::15::TokenPenaltyMultiScores",
        "ai.onnx::16::TokenPenaltyMultiScores",
        "ai.onnx::17::TokenPenaltyMultiScores",
        "ai.onnx::18::TokenPenaltyMultiScores",
    });

    // register TokenPenaltyMultiScores op info to GE
    REGISTER_CUSTOM_OP("TokenPenaltyMultiScores")                    // Set the registration name of operator
        .FrameworkType(ONNX)                          // Operator name with the original framework
        .OriginOpType(g_supportedOnnxVersion)   // Set the original frame type of the operator
        .ParseParamsByOperatorFn(ParseParamTokenPenaltyMultiScores);
} // namespace domi
