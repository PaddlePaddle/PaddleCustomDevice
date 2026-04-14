#!/bin/bash

# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

export FLAGS_prim_enable_dynamic=true 
export FLAGS_prim_all=true

# CINN related FLAG
export FLAGS_use_cinn=false
export FLAGS_group_schedule_tiling_first=true
# PIR mode
export FLAGS_enable_pir_api=true

# print Program IR
export FLAGS_print_ir=true

# debug log
export GLOG_v=0
export GLOG_vmodule=ap_generic_drr_pass=6

export CUDA_VISIBLE_DEVICES=11
export FLAGS_enable_ap=1

PADDLE_ROOT="${PADDLE_ROOT:-/path/to/your/paddle/build}"
export PYTHONPATH="${PADDLE_ROOT}/python:$PYTHONPATH"
export AP_WORKSPACE_DIR="/tmp/ap_workspace"
export AP_PATH="${PADDLE_ROOT}/python/paddle/apy/sys:${PADDLE_ROOT}/python/paddle/apy/matmul_pass:$AP_PATH"

python test_matmul_epilogue.py 2>&1 | tee output.log
