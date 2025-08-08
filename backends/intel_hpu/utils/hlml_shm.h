// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may
// not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef BACKENDS_INTEL_HPU_UTILS_HLML_SHM_H_
#define BACKENDS_INTEL_HPU_UTILS_HLML_SHM_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#define HLML_SHM_DEVICE_NAME_PREFIX "/mem_usage_accel"
#define HLML_SHM_VERSION 1
#define HLML_SHM_UPDATE_PERIOD_IN_SECS 10
#define HLML_SHM_DATA_SIZE sizeof(struct hlml_shm_data)

/*
 * Attention: When adding fields, needs to increase the version, and take care
 * of compatibility.
 */
struct hlml_shm_data {
  uint64_t version;
  uint64_t timestamp;
  uint64_t used_mem_in_bytes;
} __attribute__((packed));

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // BACKENDS_INTEL_HPU_UTILS_HLML_SHM_H_
