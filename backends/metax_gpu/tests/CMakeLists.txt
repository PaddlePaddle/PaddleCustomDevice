cmake_minimum_required(VERSION 3.10)
project(PythonUnitTests)

enable_testing()

find_package(Python REQUIRED COMPONENTS Interpreter)

set(PADDLE_LEGACY_TEST_PATH
    ${CMAKE_CURRENT_LIST_DIR}/../../../Paddle/test/legacy_test)
set(METAX_UNIT_TEST_PATH ${CMAKE_CURRENT_LIST_DIR}/unit_test)

file(GLOB_RECURSE PYTHON_TEST_SCRIPTS "${METAX_UNIT_TEST_PATH}/*.py")

if(NOT TEST_LIST_FILE)
  message(
    STATUS
      "<TEST_LIST_FILE> is not set, default test list [ ${CMAKE_CURRENT_LIST_DIR}/default.txt ] will be used."
  )
  file(STRINGS ${CMAKE_CURRENT_LIST_DIR}/default.txt TEST_PROGRAMS)

else()
  if(NOT EXISTS ${TEST_LIST_FILE})
    message(FATAL_ERROR "<TEST_LIST_FILE> is not exist, please check it again.")
  endif()

  file(STRINGS ${TEST_LIST_FILE} TEST_PROGRAMS)

  if(NOT TEST_PROGRAMS)
    message(FATAL_ERROR "<TEST_LIST_FILE> is empty.")
  endif()

  set(PYTHON_TEST_SCRIPTS "")
endif()

foreach(test_name ${TEST_PROGRAMS})
  set(CURRENT_TEST_PROGRAM ${PADDLE_LEGACY_TEST_PATH}/${test_name}.py)
  if(NOT EXISTS ${CURRENT_TEST_PROGRAM})
    message(WARNING "${CURRENT_TEST_PROGRAM} is not exist, skip it.")
  else()
    list(APPEND PYTHON_TEST_SCRIPTS ${CURRENT_TEST_PROGRAM})
  endif()
endforeach()

list(REMOVE_DUPLICATES PYTHON_TEST_SCRIPTS)

if(NOT TEST_LIST_FILE)
  list(
    REMOVE_ITEM
    PYTHON_TEST_SCRIPTS
    # Metax unit test
    ${METAX_UNIT_TEST_PATH}/test_matmul_op_metax.py
    # 精度问题
    ${PADDLE_LEGACY_TEST_PATH}/test_sum_op.py
    ${PADDLE_LEGACY_TEST_PATH}/test_max_op.py
    ${PADDLE_LEGACY_TEST_PATH}/test_cumsum_op.py
    # core.cudnnversion
    ${PADDLE_LEGACY_TEST_PATH}/test_softmax_with_cross_entropy_op.py
    ${PADDLE_LEGACY_TEST_PATH}/test_softmax_op.py
    ${PADDLE_LEGACY_TEST_PATH}/test_elementwise_add_op.py
    ${PADDLE_LEGACY_TEST_PATH}/test_gather_op.py
    # op_test.py 里 self._get_places()接口的适配问题
    ${PADDLE_LEGACY_TEST_PATH}/test_elementwise_pow_op.py
    ${PADDLE_LEGACY_TEST_PATH}/test_layer_norm_op.py
    # device == "gpu" 适配问题
    ${PADDLE_LEGACY_TEST_PATH}/test_index_add_op.py
    # paddle-gpu 报错一致
    ${PADDLE_LEGACY_TEST_PATH}/test_elementwise_div_op.py
    ${PADDLE_LEGACY_TEST_PATH}/test_stack_op.py
    ${PADDLE_LEGACY_TEST_PATH}/test_logical_op.py
    ${PADDLE_LEGACY_TEST_PATH}/test_mean_op.py
    # paddle.device.cuda.get_device_properties
    ${PADDLE_LEGACY_TEST_PATH}/test_transpose_op.py
    ${PADDLE_LEGACY_TEST_PATH}/test_randint_op.py
    ${PADDLE_LEGACY_TEST_PATH}/test_uniform_random_op.py
    # needs check_grad with fp64 precision
    ${PADDLE_LEGACY_TEST_PATH}/test_c_embedding_op.py
    # CUDAPinnedPlace 问题
    ${PADDLE_LEGACY_TEST_PATH}/test_slice_op.py
    ${PADDLE_LEGACY_TEST_PATH}/test_compare_op.py)
endif()

if(LOG_OUTPUT_DIR AND NOT EXISTS ${LOG_OUTPUT_DIR})
  file(MAKE_DIRECTORY ${LOG_OUTPUT_DIR})
  message(WARNING "${LOG_OUTPUT_DIR} is not exist, create it now.")
endif()

foreach(test_script ${PYTHON_TEST_SCRIPTS})
  get_filename_component(test_name ${test_script} NAME_WE)

  if(LOG_OUTPUT_DIR)
    set(test_log_file "${LOG_OUTPUT_DIR}/${test_name}.log")

    add_test(
      NAME "python_${test_name}"
      COMMAND sh -c
              "${Python_EXECUTABLE} ${test_script} > ${test_log_file} 2>&1"
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

  else()
    add_test(
      NAME "python_${test_name}"
      COMMAND ${Python_EXECUTABLE} ${test_script}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  endif()

  set_tests_properties("python_${test_name}" PROPERTIES TIMEOUT 600)
endforeach()
