/*
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
#pragma once

#include <atb/atb_infer.h>
#include <atb/svector.h>

enum LlamaLmheadTensorId {
    IN_HIDDENSTATES = 0,
    IN_NORMWEIGHT,
    IN_MATMULWEIGHT,
	OUT_LMHEADOUT,
	INTERMIDATE_INPUTNORMOUT,
	INTERMIDATE_LINEAR_OUT,
};

struct LlamaLmheadParam {
    float rmsNormEps = 0;
	bool transpose = true;
	int nranks = 0;
    void *hcclComm = nullptr;
};

atb::Status LlamaLmheadOperation(const LlamaLmheadParam &param,
                                 atb::Operation **operation);
