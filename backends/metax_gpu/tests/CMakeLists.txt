cmake_minimum_required(VERSION 3.10)
project(PythonUnitTests)

enable_testing()

find_package(Python REQUIRED COMPONENTS Interpreter)

file(GLOB_RECURSE PYTHON_TEST_SCRIPTS "unittest/*.py")

list(
  APPEND
  PYTHON_TEST_SCRIPTS
  ${CMAKE_CURRENT_LIST_DIR}/../../../Paddle/test/legacy_test/test_tril_triu_op.py
)

list(
  REMOVE_ITEM
  PYTHON_TEST_SCRIPTS
  ${CMAKE_CURRENT_LIST_DIR}/unittest/test_cumsum_op_metax.py
  ${CMAKE_CURRENT_LIST_DIR}/unittest/python_test_max_op_metax.py # Affected by
                                                                 # the
                                                                 # test_sum_op.py
  ${CMAKE_CURRENT_LIST_DIR}/unittest/test_expand_v2_op_metax.py
  ${CMAKE_CURRENT_LIST_DIR}/unittest/test_tril_triu_op_metax.py
  ${CMAKE_CURRENT_LIST_DIR}/unittest/test_squared_l2_norm_op_metax.py)

list(REMOVE_DUPLICATES PYTHON_TEST_SCRIPTS)
foreach(test_script ${PYTHON_TEST_SCRIPTS})
  get_filename_component(test_name ${test_script} NAME_WE)

  add_test(
    NAME "python_${test_name}"
    COMMAND ${Python_EXECUTABLE} ${test_script}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  set_tests_properties("python_${test_name}" PROPERTIES TIMEOUT 360)
endforeach()
