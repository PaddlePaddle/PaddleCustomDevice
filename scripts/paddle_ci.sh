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

failed_test_lists=''
tmp_dir=`mktemp -d`

function collect_failed_tests() {
    for file in `ls $tmp_dir`; do
        exit_code=0
        grep -q 'The following tests FAILED:' $tmp_dir/$file||exit_code=$?
        if [ $exit_code -ne 0 ]; then
            failuretest=''
        else
            failuretest=`grep -A 10000 'The following tests FAILED:' $tmp_dir/$file | sed 's/The following tests FAILED://g'|sed '/^$/d'`
            failed_test_lists="${failed_test_lists}
            ${failuretest}"
        fi
    done
}

function print_usage() {
    echo -e "\nUsage:
    ./paddle_ci.sh [OPTION]"

    echo -e "\nOptions:
    custom_npu: run custom_npu tests
    custom_cpu: run custom_cpu tests
    "
}

function init() {
    REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")/../" && pwd )"
    export REPO_ROOT

    # For paddle easy debugging
    export FLAGS_call_stack_level=2
}

function show_ut_retry_result() {
    SYSTEM=`uname -s`
    retry_unittests_ut_name=$(echo "$retry_unittests_record" | grep -oEi "\-.+\(" | sed 's/(//' | sed 's/- //' )
    retry_unittests_record_judge=$(echo ${retry_unittests_ut_name}| tr ' ' '\n' | sort | uniq -c | awk '{if ($1 >=4) {print $2}}')
    if [ -z "${retry_unittests_record_judge}" ];then
        echo "========================================"
        echo "There are failed tests, which have been successful after re-run:"
        echo "========================================"
        echo "The following tests have been re-ran:"
        echo "${retry_unittests_record}"
    else
        failed_ut_re=$(echo "${retry_unittests_record_judge}" | awk BEGIN{RS=EOF}'{gsub(/\n/,"|");print}')
        echo "========================================"
        echo "There are failed tests, which have been executed re-run,but success rate is less than 50%:"
        echo "Summary Failed Tests... "
        echo "========================================"
        echo "The following tests FAILED: "
        echo "${retry_unittests_record}" | sort -u | grep -E "$failed_ut_re"
        exit 8;
    fi
}
function custom_npu_test() {
    # install paddlepaddle daily build cpu version
    pip install hypothesis
    wget -q https://paddle-wheel.bj.bcebos.com/develop/linux/linux-cpu-mkl-avx/paddlepaddle-0.0.0-cp37-cp37m-linux_x86_64.whl
    pip install -U paddlepaddle-0.0.0-cp37-cp37m-linux_x86_64.whl && rm -rf paddlepaddle*whl

    # custom_npu build and install
    cd ${REPO_ROOT}/backends/npu/
    bash tools/compile.sh
    if [[ "$?" != "0" ]];then
        exit 7;
    fi
    cd ${REPO_ROOT}/backends/npu/build
    pip install dist/*.whl

    # run ut
    ut_total_startTime_s=`date +%s`
    tmpfile_rand=`date +%s%N`
    tmpfile=$tmp_dir/$tmpfile_rand
    GLOG_v=4 ctest --output-on-failure | tee $tmpfile;
    collect_failed_tests

    # add unit test retry for NPU
    rm -f $tmp_dir/*
    exec_times=0
    retry_unittests_record=''
    retry_time=4
    exec_time_array=('first' 'second' 'third' 'fourth')
    parallel_failed_tests_exec_retry_threshold=120
    exec_retry_threshold=30
    is_retry_execuate=0
    rerun_ut_startTime_s=`date +%s`
    if [ -n "$failed_test_lists" ];then
        if [ ${TIMEOUT_DEBUG_HELP:-OFF} == "ON" ];then
            bash $PADDLE_ROOT/tools/timeout_debug_help.sh "$failed_test_lists"    # cat logs for tiemout uts which killed by ctest
        fi
        need_retry_ut_str=$(echo "$failed_test_lists" | grep -oEi "\-.+\(.+\)" | sed 's/(.\+)//' | sed 's/- //' )
        need_retry_ut_arr=(${need_retry_ut_str})
        need_retry_ut_count=${#need_retry_ut_arr[@]}
        retry_unittests=$(echo "$failed_test_lists" | grep -oEi "\-.+\(.+\)" | sed 's/(.\+)//' | sed 's/- //' )
        while ( [ $exec_times -lt $retry_time ] )
            do
                if [[ "${exec_times}" == "0" ]] ;then
                    if [ $need_retry_ut_count -lt $parallel_failed_tests_exec_retry_threshold ];then
                        is_retry_execuate=0
                    else
                        is_retry_execuate=1
                    fi
                elif [[ "${exec_times}" == "1" ]] ;then
                    need_retry_ut_str=$(echo "$failed_test_lists" | grep -oEi "\-.+\(.+\)" | sed 's/(.\+)//' | sed 's/- //' )
                    need_retry_ut_arr=(${need_retry_ut_str})
                    need_retry_ut_count=${#need_retry_ut_arr[@]} 
                    if [ $need_retry_ut_count -lt $exec_retry_threshold ];then
                        is_retry_execuate=0
                    else
                        is_retry_execuate=1
                    fi
                fi
                if [[ "$is_retry_execuate" == "0" ]];then
                    set +e
                    retry_unittests_record="$retry_unittests_record$failed_test_lists"
                    failed_test_lists_ult=`echo "${failed_test_lists}" |grep -Po '[^ ].*$'`
                    set -e
                    if [[ "${exec_times}" == "1" ]] || [[ "${exec_times}" == "3" ]];then
                        if [[ "${failed_test_lists}" == "" ]];then
                            break
                        else
                            retry_unittests=$(echo "$failed_test_lists" | grep -oEi "\-.+\(.+\)" | sed 's/(.\+)//' | sed 's/- //' )
                        fi
                    fi
                    echo "========================================="
                    echo "This is the ${exec_time_array[$exec_times]} time to re-run"
                    echo "========================================="
                    echo "The following unittest will be re-run:"
                    echo "${retry_unittests}"                    
                    for line in ${retry_unittests[@]} ;
                        do
                            if [[ "$one_card_retry" == "" ]]; then
                                one_card_retry="^$line$"
                            else
                                one_card_retry="$one_card_retry|^$line$"
                            fi
                        done

                    if [[ "$one_card_retry" != "" ]]; then
                        ctest -R "$one_card_retry" --output-on-failure | tee $tmpfile;
                    fi
                    exec_times=$[$exec_times+1]
                    failed_test_lists=''
                    collect_failed_tests
                    rm -f $tmp_dir/*
                    one_card_retry=''
                else 
                    break
                fi

            done
    fi
    rerun_ut_endTime_s=`date +%s` 
    echo "Rerun TestCases Total Time: $[ $rerun_ut_endTime_s - $rerun_ut_startTime_s ]s" 
    ut_total_endTime_s=`date +%s`
    echo "TestCases Total Time: $[ $ut_total_endTime_s - $ut_total_startTime_s ]s"
    if [[ "$EXIT_CODE" != "0" ]];then
        show_ut_retry_result
    fi
set -ex
}

function custom_cpu_test() {
    # paddle install
    pip install hypothesis
    wget -q https://paddle-wheel.bj.bcebos.com/develop/linux/linux-cpu-mkl-avx/paddlepaddle-0.0.0-cp37-cp37m-linux_x86_64.whl
    pip install -U paddlepaddle-0.0.0-cp37-cp37m-linux_x86_64.whl && rm -rf paddlepaddle*whl

    # custom_cpu build and install
    cd ${REPO_ROOT}/backends/custom_cpu
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
        ;;
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
