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

#include "kernels/funcs/npu_op_runner.h"

#include "acl/acl_op_compiler.h"
#include "kernels/funcs/npu_enforce.h"
#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/string_helper.h"
#ifndef PADDLE_ON_INFERENCE
#include "pybind11/pybind11.h"
#endif
#include "runtime/flags.h"
#include "runtime/runtime.h"

#ifndef PADDLE_ON_INFERENCE
#define PY_GIL_RELEASE(expr)              \
  if (PyGILState_Check()) {               \
    pybind11::gil_scoped_release release; \
    expr;                                 \
  } else {                                \
    expr;                                 \
  }
#else
#define PY_GIL_RELEASE(expr) \
  { expr; }
#endif

static aclDataBuffer *float_status_buffer_ = NULL;
static aclTensorDesc *float_status_desc_ = NULL;

FLAGS_DEFINE_bool(npu_check_nan_inf, false, "check nan/inf of all npu kernels");
FLAGS_DEFINE_bool(npu_blocking_run, false, "enable sync for all npu kernels");
FLAGS_DEFINE_bool(
    npu_storage_format,
    false,
    "Enable NPU Storage Format for Ascend910 performance improvement.");
FLAGS_DEFINE_bool(npu_scale_aclnn, false, "use aclnn scale kernel");
FLAGS_DEFINE_bool(npu_split_aclnn, false, "use aclnn split kernel");
FLAGS_DEFINE_bool(npu_jit_compile, true, "enable npu jit compile");
