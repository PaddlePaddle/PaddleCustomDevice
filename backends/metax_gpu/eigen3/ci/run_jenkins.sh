#!/bin/bash
set -e
BUILD_PATH=$(pwd)/../build/
CI_SCRIPTS_PATH=$(pwd)

declare -a PROGS
declare -a CASES
declare -a ENABLED
declare -a PROG_NAMES

CPU_CORE_NUM=$(cat /proc/cpuinfo | grep "cpu cores"|wc -l)
PROCESS_THREADS=$((${CPU_CORE_NUM} / 6))
if [ $MCEIGEN_PROCESS_THREADS ];then
    PROCESS_THREADS=$MCEIGEN_PROCESS_THREADS
fi

prerequisite()
{
    python3 -m pip list | grep junitparser || python3 -m pip install junitparser
}

seek_all_cases()
{
    for ts in $($1 --gtest_list_tests | sed -n -E '/MCCOMPILER/d;/CMODEL/d;/^\w/{h;d};G;s/((\w|\/)+)(\n|.*\n)(.+)/\4\1/;p'); do
        PROGS+=($1)
        PROG_NAMES+=($2)
        CASES+=($ts)
        ENABLED+=(0)
    done
}

identify_valid_cases() {
    length=${#CASES[@]}
    while read line || [[ -n ${line} ]]
    do
        for (( i = 0; i < $length ; i++ )); do
            if [[ "${CASES[$i]}" == ${line} ]]; then
                echo "enable ${CASES[$i]} $i"
                ENABLED[$i]=1
            else
                :
            fi
        done
    done < $1
}

gen_valid_cases_list() {

    echo > $1/.tests

    local test_bin_name="need_test_bin_name"

    cat ${2} | while read ifilter
    do
        local test_name=`echo ${ifilter} | cut -d "#" -f 1`
        if [ "${test_name}" != "" ]; then
            echo "${test_name}" >> $1/.tests
        fi
    done
}

find_xml() {
    for p in "$*"; do
        find $p -name "*.xml"
    done
}

upload_testboard()
{
    if [ -n ${TESTBOARD_SUBJECT} ]; then
        find_xml $* | xargs tar zcvf ${BUILD_PATH}/tmp.tar.gz && \
	    curl -F project="$TESTBOARD_SUBJECT" -F "tarball=@${BUILD_PATH}/tmp.tar.gz;type=application/x-tar" 172.16.80.18/upload
    else
        echo "No testboard subject set, please export TESTBOARD_SUBJECT=xxx"
    fi
}

print_result() {
    results=($(cat $1/.results | awk '{print $2}'))
    length=${#results[@]}
    err=0

    for v in "${results[@]}"; do
        if [[ "$v" != 0 ]]; then
        : $((err+=1))
        fi
    done

    printf "\n=========================== Test Summary(total: %d, failed: %d)===================================\n" $length $err
    cat $1/.results
    echo "======================================================================================"
    return $err
}

run_tests()
{
    local T_ERR=0
    export ISU_FASTMODEL=1
    export USE_TDUMP=OFF
    export TMEM_LOG=OFF

    #identify_valid_cases ${CI_SCRIPTS_PATH}/${1}_list.txt
    gen_valid_cases_list ${BUILD_PATH} ${CI_SCRIPTS_PATH}/${1}_list.txt
    make -s -f ${CI_SCRIPTS_PATH}/run_test.mk -j ${PROCESS_THREADS} || T_ERR=$((T_ERR+=$?))

    return $T_ERR
}

main()
{
    case $1 in
        checkin) ;;
        nightly) ;;
        weekly) ;;
        *)
            echo "Invalid arguments"
            exit 1
    esac

    local ERR=0
    startTime_s=`date +%s`

    MAX_SUCESS_NUM=1
    if [[ ! -v MCEIGEN_MAX_SHOW_CASES ]]; then
	    MAX_SUCESS_NUM=1000
    else
	    MAX_SUCESS_NUM=$MCEIGEN_MAX_SHOW_CASES
    fi

    prerequisite || exit 1
    run_tests $1 || ERR=$((ERR+=$?))
    python3 $(pwd)/sendemail.py ${BUILD_PATH} ${MAX_SUCESS_NUM} || ERR=$((ERR+=$?))
    # upload_testboard ${BUILD_PATH}
    print_result ${BUILD_PATH} || ERR=$((ERR+=$?))

    endTime_s=`date +%s`
    sumTime=$[ $endTime_s -$startTime_s ]
    echo "Gtest time:$sumTime s"

    return $ERR
}

echo $GERRIT_REFSPEC
MODE=${1:-checkin}
main "${MODE}"
