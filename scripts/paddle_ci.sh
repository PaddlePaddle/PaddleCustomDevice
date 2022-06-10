#!/usr/bin/env bash

# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#=================================================
#                   For Paddle CI
#=================================================

set -ex

function print_usage() {
    echo -e "\nUsage:
    ./paddle_ci.sh [OPTION]"

    echo -e "\nOptions:
    custom_npu: run custom_npu tests
    custom_cpu: run custom_cpu tests
    "
}

function init() {
    WORKSPACE_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")/../../" && pwd )"
    export WORKSPACE_ROOT

    # For paddle easy debugging
    export FLAGS_call_stack_level=2
}

function custom_npu_test() {
    # paddle install
    pip install hypothesis
    pip install ${WORKSPACE_ROOT}/Paddle/build/python/dist/*whl

    # custom_npu build and install
    cd ${WORKSPACE_ROOT}/PaddleCustomDevice/backends/npu
    mkdir build && cd build
    cmake .. -DWITH_TESTING=ON
    if [[ "$?" != "0" ]];then
        exit 7;
    fi
    make -j8
    if [[ "$?" != "0" ]];then
        exit 7;
    fi
    pip install dist/*.whl

    # run ut
    ut_total_startTime_s=`date +%s`
    ctest --output-on-failure
    EXIT_CODE=$?
    ut_total_endTime_s=`date +%s`
    echo "TestCases Total Time: $[ $ut_total_endTime_s - $ut_total_startTime_s ]s"
    if [[ "$EXIT_CODE" != "0" ]];then
        exit 8;
    fi
}

function custom_cpu_test() {
    # paddle install
    pip install hypothesis
    pip install ${WORKSPACE_ROOT}/Paddle/build/python/dist/*whl

    # custom_cpu build and install
    cd ${WORKSPACE_ROOT}/PaddleCustomDevice/backends/custom_cpu
    mkdir build && cd build
    cmake .. -DWITH_TESTING=ON
    if [[ "$?" != "0" ]];then
        exit 7;
    fi
    make -j8
    if [[ "$?" != "0" ]];then
        exit 7;
    fi
    pip install dist/*.whl

    # run ut
    ut_total_startTime_s=`date +%s`
    ctest --output-on-failure
    EXIT_CODE=$?
    ut_total_endTime_s=`date +%s`
    echo "TestCases Total Time: $[ $ut_total_endTime_s - $ut_total_startTime_s ]s"
    if [[ "$EXIT_CODE" != "0" ]];then
        exit 8;
    fi
}

function main() {
    local CMD=$1 
    init
    case $CMD in
      custom_npu)
        custom_npu_test
      custom_cpu)
        custom_cpu_test
        ;;
      *)
        print_usage
        exit 1
        ;;
    esac
    echo "paddle_ci script finished as expected"
}

main $@
