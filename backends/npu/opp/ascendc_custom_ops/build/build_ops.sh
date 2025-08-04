#!/bin/bash
# -*- coding: utf-8 -*-

# 构建环境使用CANN主线包，容易引入兼容性问题。同时为了更好地控制对外发布内容，我们
# 在构建环境用msopgen工具生成工程，然后将要发布的算子交付件拷贝到新生成的工程构建
set -e

is_ci_build="n"
current_script_dir=$(dirname $(readlink -f $0))
# 构建过程source该脚本需要传递实际路径，通过参数数量判断是否为构建流程
if [ $# -ne 0 ]; then
    is_ci_build="y"
    current_script_dir=$(realpath $1)
    if [ ! -f ${current_script_dir}/build_ops.sh ]; then
        echo "${current_script_dir}/build_ops.sh not exists"
        exit 1
    fi
    if [ "x${RELEASE_TMP_DIR}" == "x" ]; then
        echo "Did not define correct RELEASE_TMP_DIR"
        exit 1
    fi
    release_path=$(realpath ${RELEASE_TMP_DIR})
    if [ ! -d ${release_path} ]; then
        echo "Invalid RELEASE_TMP_DIR"
        exit 1
    fi
    # 构建环境的toolkit默认安装路径
    local_toolkit=/home/slave1/Ascend/ascend-toolkit/latest
else
    # 对于非构建环境，推荐整包安装，通过source set_env.sh脚本会定义环境变量
    if [ "x${ASCEND_TOOLKIT_HOME}" != "x" ]; then
        local_toolkit=${ASCEND_TOOLKIT_HOME}
    else
        echo "Can not find toolkit path, please set ASCEND_TOOLKIT_HOME"
        echo "eg: export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest"
        exit 1
    fi
fi

msopgen=${local_toolkit}/python/site-packages/bin/msopgen
if [ ! -f ${msopgen} ]; then
    echo "${msopgen} not exists"
    exit 1
fi

function make_package(){
    cd ${current_script_dir}
    rm -rf pkg
    mkdir pkg
    chmod +w vendors
    mv vendors pkg
    chmod -w pkg/vendors
    chmod -w pkg
    ./custom_project/cmake/util/makeself/makeself.sh \
        --header ./custom_project/cmake/util/makeself/makeself-header.sh \
        --gzip --notemp --complevel 4 --nomd5 --sha256 --chown \
        ./pkg aie_ops.run 'aie ops'
}

function build_ops(){
    ori_path=${PWD}
    cd ${current_script_dir}
    rm -rf vendors
    source ${current_script_dir}/build_ascendc_ops.sh
    rm -rf ${current_script_dir}/vendors/aie_ascendc/bin
    rm -rf ${current_script_dir}/vendors/customize/bin
    make_package
    if [ "x${is_ci_build}" == "xy" ]; then
        cp aie_ops.run ${release_path}/
    fi
    cd ${ori_path}
}

build_ops
