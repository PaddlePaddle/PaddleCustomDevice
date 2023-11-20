/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
// common help func
#include "backend/equivalence_trans/utils.h"
// INSENSITIVE OPS
#include "backend/equivalence_trans/insensitive_ops/accuracy.h"
#include "backend/equivalence_trans/insensitive_ops/activation.h"
#include "backend/equivalence_trans/insensitive_ops/adam.h"
#include "backend/equivalence_trans/insensitive_ops/adamw.h"
#include "backend/equivalence_trans/insensitive_ops/atan.h"
#include "backend/equivalence_trans/insensitive_ops/cos.h"
#include "backend/equivalence_trans/insensitive_ops/elementwise_binary.h"
#include "backend/equivalence_trans/insensitive_ops/elementwise_unary.h"
#include "backend/equivalence_trans/insensitive_ops/floor.h"
#include "backend/equivalence_trans/insensitive_ops/gelu.h"
#include "backend/equivalence_trans/insensitive_ops/log.h"
#include "backend/equivalence_trans/insensitive_ops/matmul_v2.h"
#include "backend/equivalence_trans/insensitive_ops/maximum.h"
#include "backend/equivalence_trans/insensitive_ops/mean.h"
#include "backend/equivalence_trans/insensitive_ops/minimum.h"
#include "backend/equivalence_trans/insensitive_ops/momentum.h"
#include "backend/equivalence_trans/insensitive_ops/reduce_x.h"
#include "backend/equivalence_trans/insensitive_ops/rmsprop.h"
#include "backend/equivalence_trans/insensitive_ops/sqrt.h"
#include "backend/equivalence_trans/insensitive_ops/tanh.h"
