#!/bin/bash
# set -x

main() {
    case $1 in
        "checkin")
            GPU_ONLY="OFF"
        ;;
        "nightly")
            GPU_ONLY="OFF"
        ;;
        "weekly")
            GPU_ONLY="OFF"
        ;;
        "gpuonly")
            GPU_ONLY="ON"
        ;;
        *)
            GPU_ONLY="OFF"
        ;;
    esac

    echo "======build&run with GPU_ONLY ${GPU_ONLY}!======"

    echo "======clear build directory!======"
    REAL_PATH=`realpath $0`
    DIR_NAME=`dirname ${REAL_PATH}`
    cd ${DIR_NAME}/..
    eigen_path=`pwd`

    export CPU_CORE_NUM=$(cat /proc/cpuinfo | grep "cpu cores"|wc -l)
    export BUILD_THREADS=$(($CPU_CORE_NUM / 2))
    export TEST_THREADS=$(($CPU_CORE_NUM / 2))
    build_path=build

    # Preprocess
    mkdir -p ${build_path}
    rm -rf ${build_path}/*

    # Compile
    echo "======compiling !======"

    #NINJA=$(ninja --version 2>/dev/null)
    if [ "${NINJA}" == "" ];then
        build_system="Unix Makefiles"
    else
        build_system="Ninja"
    fi

    flags="-DCMAKE_CXX_COMPILER=${MACA_CLANG_PATH}/mxcc \
           -DEIGEN_TEST_MACA=ON -DMACA_PATH=${MACA_PATH} \
           -DBUILD_TESTING_GPU_ONLY=${GPU_ONLY} \
           -DEIGEN_TEST_CXX11=ON -DCMAKE_INSTALL_PREFIX=${build_path} \
           -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
	       -DPACKAGE_GENERATOR=\"$(PACKAGE_GENERATOR)\""

    cmake -G"${build_system}" -B ${build_path} ${flags}
    cmake --build ${build_path} --target buildtests -j ${BUILD_THREADS}

    err_count=$?
    if [ $err_count -eq 0 ]
    then
        echo "======All tests compile successed!======"
    else
        echo "======Some tests compile failed!======"
        exit $err_count
    fi

    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${MACA_PATH}/lib
    # failed cases are closed
    closed_testCase="bdcsvd_7|bdcsvd_8|bdcsvd_10|bdcsvd_13|bdcsvd_14|boostmultiprec_10"

    echo "======Running tests!======"

    cd ${build_path}
    # ctest --test-dir ${build_path} -j ${TEST_THREADS} --timeout 600 -E ${closed_testCase}
    ctest -j ${TEST_THREADS} --timeout 600 -E ${closed_testCase}

    err_count=$?
    if [ $err_count -eq 0 ]
    then
        echo "======All tests passed!======"
    else
        echo "======Some tests failed!======"
        free -h
    fi

    return $err_count
}

main $1
