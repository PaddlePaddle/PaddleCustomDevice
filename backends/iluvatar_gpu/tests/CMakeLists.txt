cmake_minimum_required(VERSION 3.14)
project(MyProject)

enable_testing()
find_package(Python REQUIRED COMPONENTS Interpreter)

set(SCRIPT_DIR ${CMAKE_CURRENT_LIST_DIR})
set(LEGACY_TEST_PATH "${SCRIPT_DIR}/../../../Paddle/test/legacy_test")

set(EXCLUDED_TESTS
    "test_ops_roi_align.py" "test_tensor_fill_diagonal_tensor_op_iluvatar.py"
    "test_tensor_fill_diagonal_op_iluvatar.py" "test_bincount_op_iluvatar.py")

list(JOIN EXCLUDED_TESTS " or " EXCLUDE_EXPR)
set(K_FILTER "not (${EXCLUDE_EXPR})")

add_test(
  NAME pytest_main
  COMMAND ${Python_EXECUTABLE} -m pytest ${CMAKE_SOURCE_DIR}/unittests/ -k
          ${K_FILTER} --tb=short -v
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

set_tests_properties(
  pytest_main
  PROPERTIES
    ENVIRONMENT "PYTHONPATH=${LEGACY_TEST_PATH}:$ENV{PYTHONPATH}"
    "LD_PRELOAD=${LD_LIBRARY_PATH}/libcuda.so.1"
    "CUSTOM_DEVICE_ROOT=${CMAKE_BINARY_DIR}/python/paddle_custom_device/")
