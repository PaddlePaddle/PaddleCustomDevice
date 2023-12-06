set(CCE_CMAKE_PATH ${CMAKE_SOURCE_DIR}/cmake/external/cce)
list(APPEND CMAKE_MODULE_PATH ${CCE_CMAKE_PATH})

set(ASCEND_CORE_TYPE VectorCore)
set(ASCEND_PRODUCT_TYPE ascend910b)

enable_language(CCE)

function(ascendc_library TARGET_NAME)
  set(options STATIC static SHARED shared)
  set(oneValueArgs "")
  set(multiValueArgs SRCS DEPS)
  cmake_parse_arguments(ascendc_library "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN})
  if(ascendc_library_SRCS)
    set_source_files_properties(${ascendc_library_SRCS} PROPERTIES LANGUAGE CCE)
    if(ascendc_library_STATIC)
      add_library(${TARGET_NAME} STATIC ${ascendc_library_SRCS})
    elseif(ascendc_library_SHARED)
      add_library(${TARGET_NAME} SHARED ${ascendc_library_SRCS})
    else()
      message(FATAL_ERROR "Must specify STATIC or SHARED")
    endif()
    if(ascendc_library_DEPS)
      add_dependencies(${TARGET_NAME} ${ascendc_library_DEPS})
    endif()
    target_compile_definitions(${TARGET_NAME} PRIVATE TILING_KEY_VAR=0)
    target_compile_options(${TARGET_NAME} PRIVATE -O2 -std=c++17)
    target_include_directories(
      ${TARGET_NAME}
      PRIVATE $ENV{ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw
              $ENV{ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/interface
              $ENV{ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/impl)
  endif()
endfunction()
