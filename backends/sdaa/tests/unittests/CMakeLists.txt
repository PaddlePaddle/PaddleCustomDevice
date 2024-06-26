# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License

file(
  GLOB TEST_OPS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

# TODO(zhanggq): skip test_set_value_op_sdaa ut
string(REPLACE "test_set_value_op_sdaa" "" TEST_OPS "${TEST_OPS}")

foreach(TEST_OP ${TEST_OPS})
  py_test_modules(${TEST_OP} MODULES ${TEST_OP})
endforeach()

py_test_modules(test_dropout_a100 MODULES test_dropout_op_sdaa ENVS
                RANDOM_ALIGN_NV_DEVICE=A100)
py_test_modules(test_dropout_v100 MODULES test_dropout_op_sdaa ENVS
                RANDOM_ALIGN_NV_DEVICE=V100)
py_test_modules(test_softmax_high_precision MODULES test_softmax_sdaa ENVS
                HIGH_PRECISION_OP_LIST=softmax,softmax_grad)
py_test_modules(
  test_softmax_with_cross_entropy_high_precision
  MODULES
  test_softmax_with_cross_entropy_op_sdaa
  ENVS
  HIGH_PRECISION_OP_LIST=softmax_with_cross_entropy,softmax_with_cross_entropy_grad
)
set_tests_properties(test_no_event_pool PROPERTIES ENVIRONMENT
                                                   FLAGS_sdaa_reuse_event=false)
set_tests_properties(
  test_dy2static_mlp_sdaa
  PROPERTIES
    ENVIRONMENT
    "FLAGS_enable_pir_with_pt_in_dy2st=0;FLAGS_allocator_strategy=naive_best_fit"
)
