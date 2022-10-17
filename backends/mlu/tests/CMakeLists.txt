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

function(py_test_modules TARGET_NAME)
    set(options SERIAL)
    set(oneValueArgs "")
    set(multiValueArgs MODULES DEPS ENVS)
    cmake_parse_arguments(py_test_modules "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    add_test(NAME ${TARGET_NAME}
        COMMAND ${CMAKE_COMMAND} -E env CUSTOM_DEVICE_ROOT=${CMAKE_BINARY_DIR}/python/paddle-plugins/ PYTHONPATH=${PYTHON_SOURCE_DIR}:$ENV{PYTHONPATH} ${py_test_modules_ENVS}
        python ${PYTHON_SOURCE_DIR}/tools/test_runner.py ${py_test_modules_MODULES}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

    if (py_test_modules_SERIAL)
        set_property(TEST ${TARGET_NAME} PROPERTY RUN_SERIAL 1)
    endif()
endfunction()

function(bash_test_modules TARGET_NAME)
    if(NOT WITH_TESTING)
        return()
    endif()

    set(options SERIAL)
    set(oneValueArgs TIMEOUT START_BASH)
    set(multiValueArgs MODULES DEPS ENVS LABELS)
    cmake_parse_arguments(bash_test_modules "${options}" "${oneValueArgs}"
                          "${multiValueArgs}" ${ARGN})
  
    set(timeout 350)
    if(${bash_test_modules_TIMEOUT})
        set(timeout ${bash_test_modules_TIMEOUT})
    endif()
  
    if(WITH_COVERAGE)
        add_test(
          NAME ${TARGET_NAME}
          COMMAND
            ${CMAKE_COMMAND} -E env CUSTOM_DEVICE_ROOT=${CMAKE_BINARY_DIR}/python/paddle-plugins/ 
            PYTHONPATH=${PADDLE_BINARY_DIR}/python
            TEST_TARGET_NAME=${TARGET_NAME} TEST_TIMEOUT=${timeout}
            ${bash_test_modules_ENVS} WITH_COVERAGE=ON
            COVERAGE_FILE=${PADDLE_BINARY_DIR}/python-coverage.data bash
            ${CMAKE_CURRENT_BINARY_DIR}/${bash_test_modules_START_BASH}
          WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    else()
        add_test(
          NAME ${TARGET_NAME}
          COMMAND
            ${CMAKE_COMMAND} -E env PYTHONPATH=${PADDLE_BINARY_DIR}/python
            TEST_TARGET_NAME=${TARGET_NAME} TEST_TIMEOUT=${timeout}
            ${bash_test_modules_ENVS} bash
            ${CMAKE_CURRENT_BINARY_DIR}/${bash_test_modules_START_BASH}
          WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    endif()
  
    if(bash_test_modules_SERIAL)
        set_property(TEST ${TARGET_NAME} PROPERTY RUN_SERIAL 1)
    endif()
  
    if(bash_test_modules_LABELS)
        set_tests_properties(${TARGET_NAME} PROPERTIES LABELS
                                                     ${bash_test_modules_LABELS})
    endif()
endfunction()

py_test_modules(test_MNIST_model MODULES test_MNIST_model)

add_subdirectory(unittests)
