cmake_minimum_required(VERSION 3.10)

project(paddle-ascend CXX C)

option(WITH_TESTING    "compile plugin with unit testing"        OFF)
option(WITH_KERNELS    "build custom kernels"                    OFF)

set(PLUGIN_NAME        "paddle_ascend")
set(PLUGIN_VERSION     "0.0.1")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(PLUGIN_SRCS runtime/runtime.cc)
include_directories(${CMAKE_SOURCE_DIR}/runtime)

include(ascend)

if (WITH_KERNELS)
  # kernel source files
  file(GLOB PLUGIN_KERNEL_FILES RELATIVE ${CMAKE_SOURCE_DIR} kernels/*.cc)
  list(APPEND PLUGIN_SRCS ${PLUGIN_KERNEL_FILES})

  # paddle
  include(paddle)

  # third_party
  set(GIT_URL "https://github.com")
  include(third_party)

  # configures
  add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)   # for inference
  add_definitions(-DPADDLE_WITH_CUSTOM_DEVICE)  # for out CustomContext
  add_definitions(-DPADDLE_WITH_CUSTOM_KERNEL)  # for out fluid seperate
endif()

####### build shared library
add_library(${PLUGIN_NAME} SHARED ${PLUGIN_SRCS})
add_dependencies(${PLUGIN_NAME} ascend_cl)
target_link_libraries(${PLUGIN_NAME} PRIVATE ${ascendcl_lib})
target_link_libraries(${PLUGIN_NAME} PRIVATE ${acl_op_compiler_lib})

if (WITH_KERNELS)
 add_dependencies(${PLUGIN_NAME} paddle_install third_party)
 target_link_libraries(${PLUGIN_NAME} PRIVATE ${paddle_lib})
endif()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/setup.py)

####### packing wheel package
add_custom_command(TARGET ${PLUGIN_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_CURRENT_BINARY_DIR}/python/
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/python/
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/python/paddle-plugins/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_BINARY_DIR}/lib${PLUGIN_NAME}.so ${CMAKE_CURRENT_BINARY_DIR}/python/paddle-plugins/
    COMMENT "Creating plugin dirrectories------>>>"
)

add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/python/.timestamp
    COMMAND python3 ${CMAKE_CURRENT_BINARY_DIR}/setup.py bdist_wheel
    DEPENDS ${PLUGIN_NAME}
    COMMENT "Packing whl packages------>>>"
)

add_custom_target(python_package ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/python/.timestamp)
