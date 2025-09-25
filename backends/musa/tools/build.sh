#!/bin/bash
# Copyright (c) 2025 Moore Threads Technology Co., Ltd("Moore Threads"). All rights reserved.
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

init() {
    RED='\033[0;31m'
    BLUE='\033[0;34m'
    BOLD='\033[1m'
    NONE='\033[0m'
    SCRIPT_NAME='paddle_musa build script'
    CUR_DIR=$(pwd) 
    PADDLE_PATH=../../Paddle
    PADDLE_PATCHES_DIR=${CUR_DIR}/patches/paddle
}

print_usage() {
  echo -e "\n${RED}Options${NONE}:
  ${BLUE}-a/--all${NONE}: build paddlepaddle and paddle_musa
  ${BLUE}-p/--paddle${NONE}: build paddlepaddle only and install
  ${BLUE}-m/--paddle_musa${NONE}: build paddle_musa only and install
  ${BLUE}-t/--test${NONE}: run all unit test
  ${BLUE}-s/--single_test${NONE}: run single unit test
  ${BLUE}-c/--clean${NONE}: clean paddle_musa
  ${BLUE}-h/--help${NONE}: show usage
  "
}

copy_bf16_fp16_file() {
  echo -e "${BLUE}copy bfloat16.h and float16.h to ${PADDLE_PATH} ...${NONE}"
  cp ${CUR_DIR}/hack/cuda_hack/float16.h ${PADDLE_PATH}/paddle/phi/common/float16.h
  cp ${CUR_DIR}/hack/cuda_hack/bfloat16.h ${PADDLE_PATH}/paddle/phi/common/bfloat16.h
}

apply_paddle_patches() {
  # apply patches into paddlepaddle 
  echo -e "${BLUE}Applying patches to ${PADDLE_PATH} ...${NONE}"
  # clean PyTorch before patching
  if [ -d "$PADDLE_PATH/.git" ]; then
    echo -e "${BLUE}Stash and checkout the paddle environment before patching. ${NONE}"
    pushd $PADDLE_PATH
    git stash -u
    popd
  fi
  
  copy_bf16_fp16_file

  for file in $(find ${PADDLE_PATCHES_DIR} -type f -print); do
    if [ "${file##*.}"x = "patch"x ]; then
      echo -e "${BLUE}applying patch: $file ${NONE}"
      pushd $PADDLE_PATH
      git apply --check $file
      git apply $file
      popd
    fi
  done
}

build_paddlepaddle() {
  pushd $PADDLE_PATH
  if [ ! -f "CMakeLists.txt" ];then
    git submodule update --init --recursive;get_pd_ret=$?
    if [ "$get_pd_ret" != 0 ];then    
        echo "get paddlepaddle failed!!!!"
        exit 11
    fi
  fi 
  
  mkdir -p build
  pushd build
  if [ ! -d "CMakeFiles" ];then 
    apply_paddle_patches;apply_ret=$?
    if [ "$apply_ret" != 0 ];then
        echo "apply paddle patches error"
        exit 10
    fi
    cmake .. -DWITH_MKL=ON \
            -DWITH_GPU=OFF \
            -DPY_VERSION=3.10 \
            -DWITH_CINN=OFF \
            -DCMAKE_EXPORT_COMPILE_COMMANDS=on;cmake_ret=$? 
    if [ "$cmake_ret" != 0 ];then
        echo "cmake error"
        exit 9
    fi
  fi
  
  make -j 128;make_ret=$?
  if [ "$cmake_ret" != 0 ];then
      echo "paddle make have error, ret=${make_ret}"
  fi

  if [ ! -f "python/dist/$(ls python/dist)" ]; then
      echo "paddle build failed!!!!!!!"
      exit 12
  fi
  pip uninstall paddlepaddle
  pip install python/dist/paddlepaddle*.whl --force-reinstall
  
  popd
  popd  
}

build_paddle_musa() {
  
  bash tools/compile.sh;build_ret=$?
  if [ "$build_ret" != 0 ];then
      echo "CMake Error Found !!!"
      exit 8;
  fi
  
  pip uninstall paddle_musa
  pip install build/dist/paddle_musa-*.whl
}

run_all_ut() {
  echo "-t unimplement"
}

run_single_ut() {
  echo "-s unimplement"
}

clean() {

  # clean paddlepaddle
  echo $BULE"begin clean paddlepaddle. "$NONE
  pushd $PADDLE_PATH
  rm build -rf
  popd
  echo $BULE"clean paddlepaddle finished. "$NONE

  # clean paddle_musa
  echo $BULE"begin clean paddle_musa. "$NONE
  rm build -rf 
  echo $BULE"clean paddle_musa finished. "$NONE

}

main() {
  init
  while true; do
    case "$1" in
    -a | --all)
      build_paddlepaddle
      build_paddle_musa 
      shift
      ;;
    -p | --paddle)
      build_paddlepaddle
      shift
      ;;
    -m | --paddle_musa)
      build_paddle_musa
      shift
      ;;
    -t | --test)
      run_all_ut
      shift
      ;;
    -s | --single_test)
      run_single_ut
      shift
      ;;
    -c | --clean)
      clean
      shift
      ;;
    -h | --help)
      print_usage
      exit
      ;;
    --)
      shift
      break
      ;;
    *)
      print_usage
      exit 1
      ;;
    esac
  done
}


main $@
