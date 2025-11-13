#!/bin/bash

# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

# Navigate to unittest_runner directory
cd ../../../tools/unittest_runner

# Run test_runner.py with specified parameters
python test_runner.py --path ../../Paddle/test/legacy_test/ --skip-float64 --disabled-file ../../backends/iluvatar_gpu/tests/disabled_test.txt --rerun-failed

# Check if failed_logs directory has any log files
if [ -d "failed_logs" ] && [ "$(ls -A failed_logs)" ]; then
    echo "Failed tests found in CI. Displaying all failure logs:"
    for log_file in failed_logs/*; do
        if [ -f "$log_file" ]; then
            echo "=== $log_file ==="
            cat "$log_file"
            echo ""
        fi
    done
    exit 1
else
    echo "All tests passed in CI."
    exit 0
fi
