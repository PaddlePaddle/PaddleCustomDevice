# ##############################################################################
# ############################  paddle jit  ####################################
# ##############################################################################
include(mise.cmake)

set(test_list
    "test_conv2d"
    "test_lstm"
    "test_masked_select"
    "test_abs"
    "test_accuracy"
    "test_argmax"
    "test_argmin"
    "test_assign_value"
    "test_atan"
    "test_bilinear_interp_v2"
    "test_bmm"
    "test_check_finite_and_unscale"
    "test_clip"
    "test_concat"
    "test_conv3d"
    "test_conv3d_transpose"
    "test_cos"
    "test_depthwise_conv2d"
    "test_elementwise_add"
    "test_elementwise_div"
    "test_elementwise_mul"
    "test_elementwise_pow"
    "test_equal"
    "test_expand_as"
    "test_exp"
    "test_fc"
    "test_fill_constant"
    "test_flip"
    "test_floor"
    "test_full_like"
    "test_gelu"
    "test_greater_equal"
    "test_greater_than"
    "test_grid_sample"
    "test_hard_sigmoid"
    "test_hard_swish"
    "test_increment"
    "test_iou_similarity"
    "test_isinf"
    "test_label_smooth"
    "test_leaky_relu"
    "test_less_equal"
    "test_less_than"
    "test_logical_and"
    "test_log"
    "test_log_softmax"
    "test_maximum"
    "test_mean"
    "test_meshgrid"
    "test_minimum"
    "test_nearest_interp_v2"
    "test_not_equal"
    "test_one_hot"
    "test_one_hot_v2"
    "test_prior_box"
    "test_randperm"
    "test_range"
    "test_reciprocal"
    "test_reduce_max"
    "test_reduce_mean"
    "test_reduce_min"
    "test_reduce_prod"
    "test_reduce_sum"
    "test_relu6"
    "test_relu"
    "test_reshape"
    "test_reverse"
    "test_roll"
    "test_scatter"
    "test_shape"
    "test_sigmoid"
    "test_sign"
    "test_size"
    "test_slice"
    "test_softmax"
    "test_split"
    "test_sqrt"
    "test_squared_l2_norm"
    "test_square"
    "test_squeeze_v2"
    "test_stack"
    "test_strided_slice"
    "test_swish"
    "test_tanh"
    "test_tile"
    "test_transpose"
    "test_tril_triu"
    "test_truncated_gaussian_random"
    "test_unstack"
    "test_yolo_box")

function(add_py_test_unittests_jit TARGET_NAME NUM)
  add_py_test(
    PROJECT
    scorpio
    PLATFORM
    silicon
    REGRESSION
    ci
    daily
    weekly
    CATEGORY
    func
    OS
    ubuntu
    MODULE
    paddle_unittests_jit
    ID
    ${NUM}
    TIMEOUT
    3600
    NAME
    py_${TARGET_NAME}
    COMMAND
    "python -m pytest ${TARGET_NAME}.py"
    ENVIRONMENT
    "FLAGS_use_stride_kernel=false;\
                           PADDLE_GCU_USE_JIT_KERNELS_ONLY=true;")
endfunction()

set(test_list_num 0)
foreach(TEST_NAME ${test_list})
  math(EXPR test_list_num "${test_list_num} + 1")
  add_py_test_unittests_jit(${TEST_NAME} ${test_list_num})
  message(STATUS "with unittests_jit pass: ${TEST_NAME}, id : ${test_list_num}")
endforeach()
