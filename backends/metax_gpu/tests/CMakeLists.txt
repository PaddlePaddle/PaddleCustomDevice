cmake_minimum_required(VERSION 3.10)
project(PythonUnitTests)

enable_testing()

find_package(Python REQUIRED COMPONENTS Interpreter)

file(GLOB_RECURSE PYTHON_TEST_SCRIPTS "unittest/*.py")

foreach(test_script ${PYTHON_TEST_SCRIPTS})
  get_filename_component(test_name ${test_script} NAME_WE)

  add_test(
    NAME "python_${test_name}"
    COMMAND ${Python_EXECUTABLE} ${test_script}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  set_tests_properties("python_${test_name}" PROPERTIES TIMEOUT 360)
endforeach()
