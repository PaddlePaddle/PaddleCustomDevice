#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

"""
Extract passed and failed unit tests from test result file
"""

import os


def extract_test_results(input_file, passed_file, failed_file):
    """
    Extract passed and failed unit tests from input file and write them to separate output files

    Args:
        input_file: Input file path
        passed_file: Output file path for passed tests
        failed_file: Output file path for failed tests
    """
    # Ensure input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} does not exist")
        return

    # Create output directories if they don't exist
    for output_file in [passed_file, failed_file]:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

    # Read input file and extract passed and failed tests
    passed_tests = []
    failed_tests = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Match passed tests
            if line.startswith("[PASSED]"):
                passed_tests.append(line)
            # Match failed tests
            elif line.startswith("[FAILED]"):
                failed_tests.append(line)

    # Write passed tests file
    with open(passed_file, "w", encoding="utf-8") as f:
        for test in passed_tests:
            f.write(test + "\n")

    # Write failed tests file
    with open(failed_file, "w", encoding="utf-8") as f:
        for test in failed_tests:
            f.write(test + "\n")

    print(f"Successfully extracted {len(passed_tests)} passed tests to {passed_file}")
    print(f"Successfully extracted {len(failed_tests)} failed tests to {failed_file}")
    print(f"Total: {len(passed_tests) + len(failed_tests)} test cases")


if __name__ == "__main__":
    # Input file path
    input_file = "tests_result.txt"
    # Output file paths
    passed_file = "passed_tests.txt"
    failed_file = "failed_tests.txt"

    extract_test_results(input_file, passed_file, failed_file)
