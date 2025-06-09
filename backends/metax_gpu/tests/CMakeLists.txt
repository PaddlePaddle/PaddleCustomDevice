cmake_minimum_required(VERSION 3.14)
project(MyProject)

find_package(Python REQUIRED COMPONENTS Interpreter)

add_custom_target(
  run_tests
  COMMAND pytest ${CMAKE_SOURCE_DIR}/unittest/
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMENT "Running unit tests with pytest")

set(CMAKE_DEFAULT_MAKE_TARGET run_tests)
