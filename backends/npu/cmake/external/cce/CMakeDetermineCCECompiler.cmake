# this should find the compiler for LANG and configure
# CMake(LANG)Compiler.cmake.in

find_program(
  CMAKE_CCE_COMPILER
  NAMES "ccec"
  PATHS "$ENV{PATH}"
  DOC "CCE Compiler")
include(CMakeCCEFunction)

mark_as_advanced(CMAKE_CCE_COMPILER)

message(STATUS "CMAKE_CCE_COMPILER: " ${CMAKE_CCE_COMPILER})
set(CMAKE_CCE_SOURCE_FILE_EXTENSIONS cce;cpp)
set(CMAKE_CCE_COMPILER_ENV_VAR "CCE")
message(STATUS "CMAKE_CURRENT_LIST_DIR: " ${CMAKE_CURRENT_LIST_DIR})

# configure all variables set in this file
configure_file(${CMAKE_CURRENT_LIST_DIR}/CMakeCCECompiler.cmake.in
               ${CMAKE_PLATFORM_INFO_DIR}/CMakeCCECompiler.cmake @ONLY)

message(STATUS "ASCEND_PRODUCT_TYPE:\n" "  ${ASCEND_PRODUCT_TYPE}")
message(STATUS "ASCEND_CORE_TYPE:\n" "  ${ASCEND_CORE_TYPE}")

# /usr/local/Ascend/ascend-toolkit/latest/$(uname -m)-linux/data/platform_config
if(DEFINED ASCEND_PRODUCT_TYPE)
  set(_CMAKE_CCE_COMMON_COMPILE_OPTIONS "--cce-auto-sync")
  if(ASCEND_PRODUCT_TYPE STREQUAL "")
    message(FATAL_ERROR "ASCEND_PRODUCT_TYPE must be non-empty if set.")
  elseif(ASCEND_PRODUCT_TYPE AND NOT ASCEND_PRODUCT_TYPE MATCHES
                                 "^ascend[0-9][0-9][0-9][a-zA-Z]?[1-9]?$")
    message(FATAL_ERROR "ASCEND_PRODUCT_TYPE: ${ASCEND_PRODUCT_TYPE}\n"
                        "is not one of the following: ascend910, ascend310p")
  elseif(ASCEND_PRODUCT_TYPE STREQUAL "ascend910b")
    if(ASCEND_CORE_TYPE STREQUAL "AiCore")
      set(_CMAKE_COMPILE_AS_CCE_FLAG "--cce-aicore-arch=dav-c220-cube")
    elseif(ASCEND_CORE_TYPE STREQUAL "VectorCore")
      set(_CMAKE_COMPILE_AS_CCE_FLAG "--cce-aicore-arch=dav-c220-vec")
    endif()
    set(_CMAKE_CCE_COMPILE_OPTIONS)
  elseif(ASCEND_PRODUCT_TYPE STREQUAL "ascend910")
    if(ASCEND_CORE_TYPE STREQUAL "AiCore")
      set(_CMAKE_COMPILE_AS_CCE_FLAG "--cce-aicore-arch=dav-c100")
    else()
      message(FATAL_ERROR, "only AiCore inside")
    endif()
    set(_CMAKE_CCE_COMPILE_OPTIONS)
  elseif(ASCEND_PRODUCT_TYPE STREQUAL "ascend310p")
    if(ASCEND_CORE_TYPE STREQUAL "AiCore")
      set(_CMAKE_COMPILE_AS_CCE_FLAG "--cce-aicore-arch=dav-m200")
    elseif(ASCEND_CORE_TYPE STREQUAL "VectorCore")
      set(_CMAKE_COMPILE_AS_CCE_FLAG "--cce-aicore-arch=dav-m200-vec")
    endif()
    set(_CMAKE_CCE_COMPILE_OPTIONS
        "-mllvm -cce-aicore-function-stack-size=16000 -mllvm -cce-aicore-fp-ceiling=2 -mllvm -cce-aicore-record-overflow=false"
    )
  endif()

  set(_CMAKE_COMPILE_AS_CCE_FLAG
      "-mllvm -cce-aicore-dcci-insert-for-scalar=false ${_CMAKE_COMPILE_AS_CCE_FLAG}"
  )
  set(_CMAKE_COMPILE_AS_CCE_FLAG
      "-mllvm -cce-aicore-addr-transform ${_CMAKE_COMPILE_AS_CCE_FLAG}")
  set(_CMAKE_COMPILE_AS_CCE_FLAG
      "-mllvm -cce-aicore-record-overflow=true ${_CMAKE_COMPILE_AS_CCE_FLAG}")
  set(_CMAKE_COMPILE_AS_CCE_FLAG
      "-mllvm -cce-aicore-function-stack-size=0x8000 ${_CMAKE_COMPILE_AS_CCE_FLAG}"
  )
  set(_CMAKE_COMPILE_AS_CCE_FLAG
      "-mllvm -cce-aicore-stack-size=0x8000 ${_CMAKE_COMPILE_AS_CCE_FLAG}")
  set(_CMAKE_COMPILE_AS_CCE_FLAG
      "--cce-aicore-only ${_CMAKE_COMPILE_AS_CCE_FLAG}")
endif()

set(_CMAKE_CCE_FLAGS
    "${_CMAKE_CCE_COMMON_COMPILE_OPTIONS} ${_CMAKE_COMPILE_AS_CCE_FLAG}"
    CACHE STRING "")
