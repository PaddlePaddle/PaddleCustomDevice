#!/usr/bin/env python3

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
Unit test failure log classification script
Analyzes all log files in the failed_logs directory, classifies them by error type, and generates a report
"""

import os
import re


def classify_failed_logs(logs_dir):
    """
    Classifies all log files in the failed_logs directory

    Returns:
    {
        'precision_issues': [],      # List of precision issues
        'kernel_not_registered': [],  # List of unregistered kernel issues
        'float64_issues': [],        # List of float64 issues
        'abnormal_exit': [],         # List of abnormal exits
        'other_issues': []           # List of other issues
    }
    """

    # Store classification results
    results = {
        "precision_issues": [],
        "kernel_not_registered": [],
        "float64_issues": [],
        "abnormal_exit": [],
        "other_issues": [],
    }

    # Iterate through all files in the logs directory
    for log_file in os.listdir(logs_dir):
        if not log_file.endswith(".log"):
            continue

        file_path = os.path.join(logs_dir, log_file)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Analyze file content to determine error types
            error_types = analyze_log_content(content, file_path)

            # Classify based on analysis results
            for error_type in error_types:
                if error_type == "precision":
                    results["precision_issues"].append(log_file)
                elif error_type == "kernel_not_registered":
                    results["kernel_not_registered"].append(log_file)
                elif error_type == "float64":
                    results["float64_issues"].append(log_file)
                elif error_type == "abnormal_exit":
                    results["abnormal_exit"].append(log_file)
                elif error_type == "other":
                    results["other_issues"].append(log_file)

        except Exception as e:
            print(f"Error processing {log_file}: {e}")
            results["other_issues"].append(log_file)

    return results


def analyze_log_content(content, file_path):
    """
    Analyzes log content and returns a list of error types
    A log file may contain multiple error types
    """
    error_types = []

    # First check if it's an abnormal exit (highest priority)
    if is_abnormal_exit(content, file_path):
        error_types.append("abnormal_exit")
        # Abnormal exit may not contain other error information, return directly
        return error_types

    # 1. Check for precision issues
    precision_pattern = r"Not equal to tolerance rtol="
    if re.search(precision_pattern, content):
        error_types.append("precision")

    # 2. Check for kernel not registered issues and float64 issues
    kernel_pattern = r"The kernel with key .* of kernel `.*` is not registered"
    float64_pattern = r"Selected wrong DataType `float64`|DataType `float64`"

    kernel_match = re.search(kernel_pattern, content)
    float64_match = re.search(float64_pattern, content)

    if kernel_match and float64_match:
        # If both kernel not registered and float64 match, classify as float64 issue only
        error_types.append("float64")
    elif kernel_match:
        # If only kernel not registered matches, classify as kernel not registered issue
        error_types.append("kernel_not_registered")

    # 3. If none of the above types match, classify as other issue
    if not error_types:
        error_types.append("other")

    return error_types


def is_abnormal_exit(content, file_path):
    """
    Determines if it's an abnormal exit

    Characteristics of abnormal exit:
    Does not contain "short test summary info"
    """
    # Only check if it contains "short test summary info"
    # Note: Content after "short test summary info" cannot be used as matching basis
    if "short test summary info" in content:
        return False

    return True


def generate_classification_report(results, output_file):
    """
    Generates a classification report and saves it to a txt file
    """
    total_files = sum(len(files) for files in results.values())

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=== Unit Test Failure Log Classification Report ===\n")
        f.write(f"Total failed unit tests: {total_files}\n")
        f.write("\n")

        f.write("1. Precision Issues:\n")
        f.write(f"   Count: {len(results['precision_issues'])}\n")
        for file in sorted(results["precision_issues"]):
            f.write(f"   - {file}\n")
        f.write("\n")

        f.write("2. Kernel Not Registered Issues:\n")
        f.write(f"   Count: {len(results['kernel_not_registered'])}\n")
        for file in sorted(results["kernel_not_registered"]):
            f.write(f"   - {file}\n")
        f.write("\n")

        f.write("3. float64 Issues:\n")
        f.write(f"   Count: {len(results['float64_issues'])}\n")
        for file in sorted(results["float64_issues"]):
            f.write(f"   - {file}\n")
        f.write("\n")

        f.write("4. Abnormal Exits:\n")
        f.write(f"   Count: {len(results['abnormal_exit'])}\n")
        for file in sorted(results["abnormal_exit"]):
            f.write(f"   - {file}\n")
        f.write("\n")

        f.write("5. Other Issues:\n")
        f.write(f"   Count: {len(results['other_issues'])}\n")
        for file in sorted(results["other_issues"]):
            f.write(f"   - {file}\n")


def main():
    """
    Main function
    """
    logs_directory = "failed_logs"
    output_file = "failed_tests_classification.txt"

    # Check if logs directory exists
    if not os.path.exists(logs_directory):
        print(f"Error: Directory {logs_directory} does not exist")
        return

    # Execute classification
    print("Analyzing log files in the failed_logs directory...")
    classification_results = classify_failed_logs(logs_directory)

    # Generate report
    print(f"Generating classification report to {output_file}...")
    generate_classification_report(classification_results, output_file)

    print(f"Classification complete! Results saved to {output_file}")


if __name__ == "__main__":
    main()
