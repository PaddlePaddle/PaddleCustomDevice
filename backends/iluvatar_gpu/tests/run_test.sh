#!/bin/bash

export PYTHONPATH=${PYTHONPATH}:/home/tianyu.zhou/PaddleCustomDevice/Paddle/test/legacy_test
mkdir -p build && cd build && cmake ..
make run_test
cd -
rm -rf build