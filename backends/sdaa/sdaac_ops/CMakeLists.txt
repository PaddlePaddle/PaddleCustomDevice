cmake_minimum_required(VERSION 3.10.2)
project(customsdaacops)

set(CMAKE_C_COMPILER "$ENV{SDAA_ROOT}/bin/tecocc")
set(CMAKE_CXX_COMPILER "$ENV{SDAA_ROOT}/bin/tecocc")

set(CMAKE_C_FLAGS "-std=c99")
set(CMAKE_CXX_FLAGS "-std=c++17")

set(SDAAC_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR} ${SDAA_INC})

file(
  GLOB_RECURSE SDAA_C_OPS
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  *.scpp)

set(complie_options -flto -fPIC)

add_library(sdaacops_objs OBJECT ${SDAA_C_OPS})
set_source_files_properties(${SDAA_C_OPS} PROPERTIES LANGUAGE CXX)
set_target_properties(sdaacops_objs PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(sdaacops_objs PUBLIC ${SDAAC_INCLUDE_DIR})
target_compile_options(sdaacops_objs PRIVATE ${complie_options})
add_custom_target(
  customsdaacops ALL
  COMMAND
    tecocc $<TARGET_OBJECTS:sdaacops_objs> -flto -fPIC -shared --sdaa-link
    -sdaa-static-lib -fuse-ld=lld -lm -L ${SDAA_LIB} -o
    ${CUSTOM_SDAA_C_OPS_LIB}/libcustomsdaacops.a
  COMMAND_EXPAND_LISTS)
add_dependencies(customsdaacops sdaacops_objs)
