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
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SetStopValueMultiSeqsTilingData)
TILING_DATA_FIELD_DEF(int32_t, bs);
TILING_DATA_FIELD_DEF(int32_t, length);
TILING_DATA_FIELD_DEF(int32_t, stop_seqs_num);
TILING_DATA_FIELD_DEF(int32_t, stop_seqs_max_len);
TILING_DATA_FIELD_DEF(int32_t, eos_len);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SetStopValueMultiSeqs, SetStopValueMultiSeqsTilingData)
}