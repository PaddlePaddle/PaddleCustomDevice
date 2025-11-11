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

set -e

SOURCE_DIR="backends/metax_gpu/tests/unittest"
SEARCH_DIR="Paddle/test/legacy_test"
PREFIX_FILE="metax_prefixes.txt"
UNMATCHED_FILE="unmatched_files.txt"
EXIST_FILE="existing_files.txt"
MISS_FILE="missing_files.txt"

# 检查源路径是否存在
if [ ! -d "$SOURCE_DIR" ]; then
    echo "错误: 源路径 '$SOURCE_DIR' 不存在或不是一个目录"
    exit 1
fi

# 检查搜索路径是否存在
if [ ! -d "$SEARCH_DIR" ]; then
    echo "错误: 搜索路径 '$SEARCH_DIR' 不存在或不是一个目录"
    exit 1
fi

# 第一步：提取前缀（根据新规则处理）
echo "第一步：从 '$SOURCE_DIR' 提取文件前缀（按_op/_metax规则）..."
> "$PREFIX_FILE"      # 清空前缀文件
> "$UNMATCHED_FILE"   # 清空未匹配文件列表

find "$SOURCE_DIR" -type f -name "*.py" | while read -r file; do
    filename=$(basename "$file")
    prefix=""

    # 规则1：如果包含_op关键字，提取_op前的所有字符
    if [[ "$filename" == *"_op"* ]]; then
        prefix="${filename%%_op*}"
        echo "提取前缀（_op规则）: $prefix (来自 $filename)"
        echo "$prefix" >> "$PREFIX_FILE"

    # 规则2：如果没有_op但有_metax，提取_metax前的所有字符
    elif [[ "$filename" == *"_metax"* ]]; then
        prefix="${filename%%_metax*}"
        echo "提取前缀（_metax规则）: $prefix (来自 $filename)"
        echo "$prefix" >> "$PREFIX_FILE"

    # 规则3：都不包含，归类到未匹配
    else
        echo "未匹配的文件: $filename（不包含_op和_metax）"
        echo "$filename" >> "$UNMATCHED_FILE"
    fi
done

# 检查是否有提取到前缀或未匹配文件
prefix_count=$(wc -l < "$PREFIX_FILE")
unmatched_count=$(wc -l < "$UNMATCHED_FILE")

echo "提取完成 - 有效前缀: $prefix_count 个，未匹配文件: $unmatched_count 个"

if [ $prefix_count -eq 0 ] && [ $unmatched_count -eq 0 ]; then
    echo "警告: 在 '$SOURCE_DIR' 中未找到任何以 '_metax.py' 结尾的文件"
    exit 0
fi

# 第二步：在搜索路径中查找同名文件（仅搜索当前目录，不包括子文件夹）
echo -e "\n第二步：在 '$SEARCH_DIR' 中搜索同名文件（深度为1）..."
> "$EXIST_FILE"   # 清空存在文件列表
> "$MISS_FILE"    # 清空缺失文件列表

# 逐个处理每个前缀
while read -r prefix; do
    # 跳过空行
    if [ -z "$prefix" ]; then
        continue
    fi

    # 只在搜索路径的直接目录下查找（深度为1）
    found=$(find "$SEARCH_DIR" -maxdepth 1 -type f -name "${prefix}_op.py" -print -quit)

    if [ -n "$found" ]; then
        echo "$prefix -> 找到文件: $found"
        echo "${prefix}_op.py" >> "$EXIST_FILE"
    else
        echo "$prefix -> 未找到同名文件"
        echo "$prefix" >> "$MISS_FILE"
    fi
done < "$PREFIX_FILE"

# 输出结果统计
exist_count=$(wc -l < "$EXIST_FILE")
miss_count=$(wc -l < "$MISS_FILE")

echo -e "\n处理完成！"
echo "找到同名文件的前缀数量: $exist_count（已保存到 $EXIST_FILE）"
echo "未找到同名文件的前缀数量: $miss_count（已保存到 $MISS_FILE）"
echo "未匹配规则的文件数量: $unmatched_count（已保存到 $UNMATCHED_FILE）"
