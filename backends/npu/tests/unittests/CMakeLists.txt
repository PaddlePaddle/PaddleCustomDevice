# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

file(GLOB TEST_OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

foreach(TEST_OP ${TEST_OPS})
    py_test_modules(${TEST_OP} MODULES ${TEST_OP})
endforeach(TEST_OP)

set_tests_properties(${TEST_OPS} PROPERTIES TIMEOUT 1000)

set_tests_properties(test_check_nan_inf_op_npu PROPERTIES ENVIRONMENT FLAGS_ascend_check_nan_inf=1)
set_tests_properties(test_adam_op_npu PROPERTIES ENVIRONMENT FLAGS_USE_STANDALONE_EXECUTOR=0)
set_tests_properties(test_adamw_op_npu PROPERTIES ENVIRONMENT FLAGS_USE_STANDALONE_EXECUTOR=0)
set_tests_properties(test_cos_op_npu PROPERTIES ENVIRONMENT FLAGS_USE_STANDALONE_EXECUTOR=0)
set_tests_properties(test_elementwise_div_op_npu PROPERTIES ENVIRONMENT FLAGS_USE_STANDALONE_EXECUTOR=0)
set_tests_properties(test_elementwise_max_op_npu PROPERTIES ENVIRONMENT FLAGS_USE_STANDALONE_EXECUTOR=0)
set_tests_properties(test_elementwise_sub_op_npu PROPERTIES ENVIRONMENT FLAGS_USE_STANDALONE_EXECUTOR=0)
set_tests_properties(test_elementwise_mod_op_npu PROPERTIES ENVIRONMENT FLAGS_USE_STANDALONE_EXECUTOR=0)
set_tests_properties(test_elementwise_pow_op_npu PROPERTIES ENVIRONMENT FLAGS_USE_STANDALONE_EXECUTOR=0)
set_tests_properties(test_elementwise_min_op_npu PROPERTIES ENVIRONMENT FLAGS_USE_STANDALONE_EXECUTOR=0)
set_tests_properties(test_gelu_op_npu PROPERTIES ENVIRONMENT FLAGS_USE_STANDALONE_EXECUTOR=0)
set_tests_properties(test_leaky_relu_op_npu PROPERTIES ENVIRONMENT FLAGS_USE_STANDALONE_EXECUTOR=0)
set_tests_properties(test_log_op_npu PROPERTIES ENVIRONMENT FLAGS_USE_STANDALONE_EXECUTOR=0)
set_tests_properties(test_momentum_op_npu PROPERTIES ENVIRONMENT FLAGS_USE_STANDALONE_EXECUTOR=0)
set_tests_properties(test_pow_op_npu PROPERTIES ENVIRONMENT FLAGS_USE_STANDALONE_EXECUTOR=0)
set_tests_properties(test_reduce_sum_op_npu PROPERTIES ENVIRONMENT FLAGS_USE_STANDALONE_EXECUTOR=0)
set_tests_properties(test_rmsprop_op_npu PROPERTIES ENVIRONMENT FLAGS_USE_STANDALONE_EXECUTOR=0)
set_tests_properties(test_sgd_op_npu PROPERTIES ENVIRONMENT FLAGS_USE_STANDALONE_EXECUTOR=0)
set_tests_properties(test_slice_op_npu PROPERTIES ENVIRONMENT FLAGS_USE_STANDALONE_EXECUTOR=0)
set_tests_properties(test_softmax_op_npu PROPERTIES ENVIRONMENT FLAGS_USE_STANDALONE_EXECUTOR=0)
set_tests_properties(test_sqrt_op_npu PROPERTIES ENVIRONMENT FLAGS_USE_STANDALONE_EXECUTOR=0)
set_tests_properties(test_square_op_npu PROPERTIES ENVIRONMENT FLAGS_USE_STANDALONE_EXECUTOR=0)
set_tests_properties(test_tanh_op_npu PROPERTIES ENVIRONMENT FLAGS_USE_STANDALONE_EXECUTOR=0)
set_tests_properties(test_relu6_op_npu PROPERTIES ENVIRONMENT FLAGS_USE_STANDALONE_EXECUTOR=0)
set_tests_properties(test_relu_op_npu PROPERTIES ENVIRONMENT FLAGS_USE_STANDALONE_EXECUTOR=0)
set_tests_properties(test_softmax_with_cross_entropy_op_npu PROPERTIES ENVIRONMENT FLAGS_USE_STANDALONE_EXECUTOR=0)

set(GRAPH_ENGINE_TEST 
test_abs_op_npu.py
test_fill_any_like_op_npu.py
test_fill_constant_op_npu.py
test_flatten_contiguous_range_op_npu.py
test_cast_op_npu.py
test_conv2d_op_npu.py
test_pool2d_op_npu.py
test_relu_op_npu.py
test_scale_op_npu.py
test_top_k_v2_op_npu.py
test_uniform_random_op_npu.py
test_gaussian_random_op_npu.py
test_elementwise_add_op_npu.py
test_elementwise_mul_op_npu.py
test_reduce_mean_op_npu.py
test_batch_norm_op_npu.py
test_softmax_with_cross_entropy_op_npu.py
test_accuracy_op_npu.py
test_matmulv2_op_npu.py)

foreach(TEST_OP ${GRAPH_ENGINE_TEST})
    string(REPLACE ".py" "_graph_engine" TEST_NAME ${TEST_OP})
    string(REPLACE ".py" "" TEST_OP ${TEST_OP})
    py_test_modules(${TEST_NAME} MODULES ${TEST_OP} ENVS FLAGS_use_graph_engine=1)
endforeach(TEST_OP)
