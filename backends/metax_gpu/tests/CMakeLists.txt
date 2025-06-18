cmake_minimum_required(VERSION 3.10)
project(PythonUnitTests)

# 启用测试
enable_testing()

# 查找 Python 解释器
find_package(Python REQUIRED COMPONENTS Interpreter)

# 递归查找所有 Python 测试脚本（匹配 unittest 目录下的 *.py 文件）
file(GLOB_RECURSE PYTHON_TEST_SCRIPTS "unittest/*.py")

# 为每个测试脚本添加 CTest 测试用例
foreach(test_script ${PYTHON_TEST_SCRIPTS})
  # 提取测试名称（去掉路径和扩展名）
  get_filename_component(test_name ${test_script} NAME_WE)
  # 添加测试
  add_test(
    NAME "python_${test_name}"
    COMMAND ${Python_EXECUTABLE} ${test_script}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  # 可选：设置超时时间（秒）
  set_tests_properties("python_${test_name}" PROPERTIES TIMEOUT 360)
endforeach()
