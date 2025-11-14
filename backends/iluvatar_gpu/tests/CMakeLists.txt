cmake_minimum_required(VERSION 3.14)
project(MyProject)

enable_testing()
find_package(Python REQUIRED COMPONENTS Interpreter)

set(SCRIPT_DIR ${CMAKE_CURRENT_LIST_DIR})
set(LEGACY_TEST_PATH "${SCRIPT_DIR}/../../../Paddle/test/legacy_test")

set(EXCLUDED_TESTS
    "test_ops_roi_align.py" "test_tensor_fill_diagonal_tensor_op_iluvatar.py"
    "test_tensor_fill_diagonal_op_iluvatar.py" "test_bincount_op_iluvatar.py")

# Find all test files
file(GLOB TEST_FILES "${CMAKE_SOURCE_DIR}/unittests/test_*.py")

# Create a separate CTest test for each test file
foreach(TEST_FILE ${TEST_FILES})
  get_filename_component(TEST_NAME ${TEST_FILE} NAME)

  # Check if it's in the exclusion list
  set(IS_EXCLUDED FALSE)
  foreach(EXCLUDED_TEST ${EXCLUDED_TESTS})
    if(TEST_NAME STREQUAL EXCLUDED_TEST)
      set(IS_EXCLUDED TRUE)
      break()
    endif()
  endforeach()

  # If not in the exclusion list, add the test
  if(NOT IS_EXCLUDED)
    # Get the filename without extension as the test name
    get_filename_component(TEST_BASE_NAME ${TEST_FILE} NAME_WE)

    add_test(
      NAME ${TEST_BASE_NAME}
      COMMAND ${Python_EXECUTABLE} -m pytest ${TEST_FILE} --tb=short
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    set_tests_properties(
      ${TEST_BASE_NAME}
      PROPERTIES
        ENVIRONMENT "PYTHONPATH=${LEGACY_TEST_PATH}:$ENV{PYTHONPATH}"
        "LD_PRELOAD=${LD_LIBRARY_PATH}/libcuda.so.1"
        "CUSTOM_DEVICE_ROOT=${CMAKE_BINARY_DIR}/python/paddle_custom_device/")
  endif()
endforeach()
