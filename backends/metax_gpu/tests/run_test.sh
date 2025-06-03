#!/bin/bash

# export PATH=/usr/local/corex-4.3.0/bin:$PATH
# export LD_LIBRARY_PATH=/usr/local/corex-4.3.0/lib
# export LIBRARY_PATH=/usr/local/corex-4.3.0/lib
export PYTHONPATH=${PYTHONPATH}:${PADDLE_SOURCE_DIR}/test/legacy_test

mkdir -p build && cd build && cmake ..
make run_tests
