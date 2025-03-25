#!/usr/bin/python3

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

import os
import argparse

import xml.etree.ElementTree as ET

cmd_args = {}
parser = argparse.ArgumentParser(
    description="help script for run the test",
    add_help=True,
    formatter_class=argparse.RawTextHelpFormatter,
)
parser.add_argument("--input_file", type=str, help="xml test result input file")

cmd_args.update(vars(parser.parse_args()))

input_file_xml = "test_result.xml"
if cmd_args["input_file"]:
    if os.path.exists(cmd_args["input_file"]) is False:
        print("test result file is not exist")
        exit(2)
    input_file_xml = cmd_args["input_file"]

tree = ET.parse(input_file_xml)
root = tree.getroot()

results = {"passed": 0, "failed": 0, "skipped": 0}

for testcase in root.findall("testcase"):
    name = testcase.get("name")
    if testcase.find("failure") is not None:
        results["failed"] += 1
        print(f"\tTestcase: {name} failed")
    elif testcase.find("skipped") is not None:
        results["skipped"] += 1
    else:
        results["passed"] += 1
        print(f"\tTestcase: {name} passed")

print(f"\tPassed tests: {results['passed']}")
print(f"\tFailed tests: {results['failed']}")
print(f"\tSkipped tests: {results['skipped']}")
