#!/bin/bash
# set -x

export ISU_FASTMODEL=1
export USE_TDUMP=OFF
export TMEM_LOG=OFF

echo ${PACKAGE_GENERATOR:=DEB}
CPU_CORE_NUM=$(cat /proc/cpuinfo | grep "cpu cores"|wc -l)
BASE_DIR=`realpath $0 | xargs dirname | xargs -I {} realpath {}/../`
BUILD_DIR=${BASE_DIR}/build
BUILD_THREADS=$(($CPU_CORE_NUM / 2))
if [ $MCEIGEN_BUILD_THREADS ];then
    BUILD_THREADS=$MCEIGEN_BUILD_THREADS
fi

function build_tests() {
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

    build_path=build

    # Preprocess
    mkdir -p ${build_path}
    rm -rf ${build_path}/*

    # Compile
    echo "======compiling!======"

    #NINJA=$(ninja --version 2>/dev/null)
    if [ "${NINJA}" == "" ];then
        build_system="Unix Makefiles"
    else
        build_system="Ninja"
    fi

    flags="-DCMAKE_CXX_COMPILER=${MACA_CLANG_PATH}/mxcc \
           -DEIGEN_TEST_MACA=ON -DMACA_PATH=${MACA_PATH} \
           -DBUILD_TESTING_GPU_ONLY=${GPU_ONLY} \
           -DEIGEN_TEST_CXX11=ON -DCMAKE_INSTALL_PREFIX=${build_path}\
           -DCMAKE_PACKAGE_DIR=${DEFAULT_INSTALL_DIR}/deb \
           -DCMAKE_TARGET_ARCH=${ARCH} \
           -DMACA_VERSION=${MACA_VERSION} \
           -DDISTRO_CODE=${DISTRO_CODE} \
           -DPACKAGE_GENERATOR=${PACKAGE_GENERATOR}"

    local ERR=0

    cmake -G"${build_system}" -B ${build_path} ${flags}
    ERR=$((ERR+=$?))
    cmake --build ${build_path} --target buildtests -j ${BUILD_THREADS}
    ERR=$((ERR+=$?))

    if [ $ERR -eq 0 ]
    then
        echo "======All tests compile successed!======"
    else
        echo "======Some tests compile failed!======"
    fi

    return ${ERR}
}

function run_jenkins() {

    local JENKINS_ERR=0

    #Need export MACA_PATH before running run_jenkins.sh
    cd ${BASE_DIR}/ci
    bash run_jenkins.sh ${1:-checkin} || ((JENKINS_ERR+=$?))

    if [ $JENKINS_ERR -ne 0 ]; then
        return $JENKINS_ERR
    else
        echo "=======run_jenkins Successfully========"
    fi

    return $JENKINS_ERR

}

function run_tests() {
    local ERR=0
    build_tests $1
    ERR=$((ERR+=$?))

    if [ $ERR -ne 0 ]; then
        exit $ERR
    fi

    run_jenkins ${1} || ERR=$((ERR+=$?))

    exit $ERR
}

run_tests $1
