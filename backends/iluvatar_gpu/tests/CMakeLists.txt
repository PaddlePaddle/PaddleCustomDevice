cmake_minimum_required(VERSION 3.14)
project(MyProject)

enable_testing()
find_package(Python REQUIRED COMPONENTS Interpreter)

set(SCRIPT_DIR ${CMAKE_CURRENT_LIST_DIR})
set(LEGACY_TEST_PATH "${SCRIPT_DIR}/../../../Paddle/test/legacy_test")

add_test(
  NAME pytest_main
  COMMAND ${Python_EXECUTABLE} -m pytest ${CMAKE_SOURCE_DIR}/unittests/
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

set_tests_properties(
  pytest_main
  PROPERTIES
    ENVIRONMENT "PYTHONPATH=${LEGACY_TEST_PATH}:$ENV{PYTHONPATH}"
    "LD_PRELOAD=${LD_LIBRARY_PATH}/libcuda.so.1"
    "CUSTOM_DEVICE_ROOT=${CMAKE_BINARY_DIR}/python/paddle_custom_device/")
