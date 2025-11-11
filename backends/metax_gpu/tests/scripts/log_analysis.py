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

import json
import os
import fnmatch
import shutil
from enum import Enum


class TestResult(Enum):
    OK = "OK"
    FAILURE = "FAILED"


class LogAnalyzer:
    def __init__(
        self,
        classify_file: str,
        search_path: str,
        pattern: str = None,
        encoding: str = "utf-8",
    ):
        self.__patten = pattern
        self.__search_path = search_path
        self.__encoding = encoding
        self.__statistical_data = {}

        self.__classify_data = self.__read_json_file(classify_file)
        for key, value in self.__classify_data.items():
            self.__statistical_data[key] = {}
            for sub_key in list(value.keys()):
                self.__statistical_data[key][sub_key] = []

        self.__statistical_data[TestResult.OK.value]["noskip"] = []
        self.__statistical_data[TestResult.FAILURE.value]["other"] = []

    def __read_json_file(self, path: str) -> dict:
        with open(path, "r", encoding=self.__encoding) as f:
            data = json.load(f)
        f.close()
        return data

    def __check_path(self, path: str) -> None:
        """
        处理指定路径：
        - 若为文件夹路径：不存在则创建，存在则清空内容
        - 若为文件路径：不存在则创建，存在则清空内容
        """
        try:
            # 判断路径是否存在
            if os.path.exists(path):
                # 路径存在，判断是文件还是文件夹
                if os.path.isfile(path):
                    # 处理文件：清空内容
                    with open(path, "w", encoding="utf-8") as f:
                        f.write("")  # 写入空内容清空文件
                    # print(f"文件已存在，已清空内容: {path}")

                elif os.path.isdir(path):
                    # 处理文件夹：清空所有内容
                    for item in os.listdir(path):
                        item_path = os.path.join(path, item)
                        if os.path.isfile(item_path) or os.path.islink(item_path):
                            os.remove(item_path)  # 删除文件或链接
                        elif os.path.isdir(item_path):
                            shutil.rmtree(item_path)  # 递归删除子文件夹
                    # print(f"文件夹已存在，已清空内容: {path}")
            else:
                # 路径不存在，判断目标类型（根据最后一个元素是否有扩展名）
                # 获取路径的最后一部分
                last_part = os.path.basename(path)

                # 判断是否为文件路径（包含扩展名）
                if "." in last_part and not last_part.endswith("."):
                    # 创建文件（包括父目录）
                    parent_dir = os.path.dirname(path)
                    if parent_dir and not os.path.exists(parent_dir):
                        os.makedirs(parent_dir, exist_ok=True)
                    with open(path, "w", encoding="utf-8") as f:
                        pass  # 创建空文件
                    # print(f"文件不存在，已创建: {path}")

                else:
                    # 创建文件夹（支持多级目录）
                    os.makedirs(path, exist_ok=True)
                    # print(f"文件夹不存在，已创建: {path}")

        except PermissionError:
            print(f"权限错误：无法操作路径 {path}")
        except Exception as e:
            print(f"处理路径时发生错误: {str(e)}")

    def save_result(self, dir_path: str = "./") -> None:
        """
        判断文件夹是否存在：
        - 不存在则创建
        - 存在则清空文件夹内所有内容（保留文件夹本身）
        """

        for key, value in self.__statistical_data.items():
            sub_dir = os.path.join(dir_path, key)
            self.__check_path(sub_dir)

            for sub_key, sub_value in value.items():
                # print(f"{sub_key}: {len(value[sub_key])} - ({sub_value})")
                try:
                    with open(
                        os.path.join(sub_dir, sub_key) + ".txt", "w", encoding="utf-8"
                    ) as f:
                        for op_name in sub_value:
                            if not op_name.endswith("\n"):
                                op_name += "\n"
                            f.write(op_name)
                    # print(f"内容已成功{'追加' if append else '写入'}到 {file_path}")
                except Exception as e:
                    print(f"写入文件失败: {e}")

    def show_result(self) -> None:
        test_counts = 0
        for key, value in self.__statistical_data.items():
            print(f"\n----------  {key}  ----------")
            for sub_key, sub_value in value.items():
                test_counts = test_counts + len(value[sub_key])
                print(f"{sub_key}: {len(value[sub_key])}\n\t{sub_value}\n")
        print(
            f"\n******************* Total log num: {test_counts} *******************\n\n"
        )

    def run(self):
        """
        读取指定目录下符合命名规则的文件，并遍历每一行

        参数:
            search_path: 要搜索的根目录
            pattern: 文件名匹配规则（支持通配符，如 '*.txt', 'file_*.log')
        """
        for dirpath, dirnames, filenames in os.walk(self.__search_path):
            for filename in fnmatch.filter(filenames, self.__patten):
                file_path = os.path.join(dirpath, filename)
                # print(f"\n===== 正在处理文件: {file_path} =====")

                cur_res_type = TestResult.FAILURE
                cur_sub_type = "other"
                finish_early = False

                try:
                    with open(file_path, "r", encoding=self.__encoding) as f:
                        for line in f:
                            for sub_type, sub_type_params in self.__classify_data[
                                cur_res_type.value
                            ].items():
                                for keyword in sub_type_params["rule"]:
                                    if keyword in line:
                                        cur_sub_type = sub_type
                                        if sub_type == "missing":
                                            finish_early = True
                                        break

                                if finish_early:
                                    break

                            if finish_early:
                                break

                            if len(line) >= 2 and line[:2] == "OK":
                                cur_res_type = TestResult.OK
                                cur_sub_type = None
                                for sub_type, sub_type_params in self.__classify_data[
                                    cur_res_type.value
                                ].items():
                                    for rule in sub_type_params["rule"]:
                                        if rule in line:
                                            cur_sub_type = sub_type
                                break

                        op_name = filename.split(".")
                        if cur_sub_type is None:
                            self.__statistical_data[cur_res_type.value][
                                "noskip"
                            ].append(op_name[0])
                        else:
                            self.__statistical_data[cur_res_type.value][
                                cur_sub_type
                            ].append(op_name[0])
                        # print(f"Result: {cur_res_type.value}, type: {cur_sub_type}")
                    f.close()
                except UnicodeDecodeError:
                    print(f"警告: 文件 {file_path} 编码不是 utf-8,跳过处理")
                except Exception as e:
                    print(f"处理文件 {file_path} 时出错: {str(e)}")


if __name__ == "__main__":

    analyzer = LogAnalyzer(
        classify_file="./classify.json",
        search_path="./High_op/logs_output-20251106",
        pattern="test_*.log",
    )

    analyzer.run()
    analyzer.show_result()
    analyzer.save_result("./High_op/logs_output-20251106-result")
