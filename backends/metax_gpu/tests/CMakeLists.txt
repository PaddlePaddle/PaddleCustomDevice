cmake_minimum_required(VERSION 3.10)
project(PythonUnitTests)

enable_testing()

find_package(Python REQUIRED COMPONENTS Interpreter)

set(PADDLE_LEGACY_TEST_PATH
    ${CMAKE_CURRENT_LIST_DIR}/../../../Paddle/test/legacy_test)
set(METAX_UNIT_TEST_PATH ${CMAKE_CURRENT_LIST_DIR}/unit_test)

set(NEED_REMOVE_KEYWORDS "attention")

file(GLOB_RECURSE PYTHON_TEST_SCRIPTS "${METAX_UNIT_TEST_PATH}/*.py")

if(NOT TEST_LIST_FILE)
  message(
    STATUS
      "<TEST_LIST_FILE> is not set, default test list [ ${CMAKE_CURRENT_LIST_DIR}/default.txt ] will be used."
  )
  file(STRINGS ${CMAKE_CURRENT_LIST_DIR}/default.txt TEST_PROGRAMS)

else()
  if(NOT EXISTS ${TEST_LIST_FILE})
    message(FATAL_ERROR "<TEST_LIST_FILE> is not exist, please check it again.")
  endif()

  file(STRINGS ${TEST_LIST_FILE} TEST_PROGRAMS)

  if(NOT TEST_PROGRAMS)
    message(FATAL_ERROR "<TEST_LIST_FILE> is empty.")
  endif()

  set(PYTHON_TEST_SCRIPTS "")
endif()

foreach(test_name ${TEST_PROGRAMS})
  set(IS_REMOVE FALSE)

  foreach(keyword ${NEED_REMOVE_KEYWORDS})
    string(FIND "${test_name}" "${keyword}" RES)
    if(NOT RES EQUAL -1)
      set(IS_REMOVE TRUE)
      break()
    endif()
  endforeach()

  if(IS_REMOVE)
    continue()
  endif()

  string(FIND "${test_name}" "metax" METAX_SUFFIX_POS)
  if(NOT METAX_SUFFIX_POS EQUAL -1)
    set(CURRENT_TEST_PROGRAM ${METAX_UNIT_TEST_PATH}/${test_name}.py)
  else()
    set(CURRENT_TEST_PROGRAM ${PADDLE_LEGACY_TEST_PATH}/${test_name}.py)
  endif()

  if(NOT EXISTS ${CURRENT_TEST_PROGRAM})
    message(WARNING "${CURRENT_TEST_PROGRAM} is not exist, skip it.")
  else()
    list(APPEND PYTHON_TEST_SCRIPTS ${CURRENT_TEST_PROGRAM})
  endif()
endforeach()

list(REMOVE_DUPLICATES PYTHON_TEST_SCRIPTS)

if(NOT TEST_LIST_FILE)
  set(NEED_IGNORE_FILE ${CMAKE_CURRENT_LIST_DIR}/ignore.txt)

  if(EXISTS ${NEED_IGNORE_FILE})

    file(STRINGS ${NEED_IGNORE_FILE} ignore_lines)

    set(current_block "")
    set(all_blocks "")

    foreach(line ${ignore_lines})
      if(line MATCHES "^\\[(.+)\\]$")
        set(current_block "${CMAKE_MATCH_1}")
        list(APPEND all_blocks "${current_block}")
        set(ignore_map_${current_block} "")
      elseif(NOT line STREQUAL "")
        if(NOT current_block STREQUAL "")
          list(APPEND ignore_map_${current_block} "${line}")
        endif()
      endif()
    endforeach()

    if(IGNORE_BLOCKS STREQUAL "NONE")
      message(STATUS "IGNORE_BLOCKS=NONE → No ignore blocks applied.")
      set(blocks_to_ignore "")
    elseif(IGNORE_BLOCKS)
      string(REPLACE ";" " " blocks_to_ignore "${IGNORE_BLOCKS}")
      message(STATUS "Only ignoring blocks: ${blocks_to_ignore}")
    else()
      set(blocks_to_ignore "${all_blocks}")
      message(STATUS "Default: ignoring ALL blocks: ${blocks_to_ignore}")
    endif()

    foreach(block ${blocks_to_ignore})
      set(varname "ignore_map_${block}")
      foreach(test_name ${${varname}})
        list(REMOVE_ITEM PYTHON_TEST_SCRIPTS
             "${PADDLE_LEGACY_TEST_PATH}/${test_name}.py")
        list(REMOVE_ITEM PYTHON_TEST_SCRIPTS
             "${METAX_UNIT_TEST_PATH}/${test_name}.py")
      endforeach()
    endforeach()

  endif()
endif()

if(LOG_OUTPUT_DIR AND NOT EXISTS ${LOG_OUTPUT_DIR})
  file(MAKE_DIRECTORY ${LOG_OUTPUT_DIR})
  message(WARNING "${LOG_OUTPUT_DIR} is not exist, create it now.")
endif()

foreach(test_script ${PYTHON_TEST_SCRIPTS})
  get_filename_component(test_name ${test_script} NAME_WE)

  if(LOG_OUTPUT_DIR)
    set(test_log_file "${LOG_OUTPUT_DIR}/${test_name}.log")

    add_test(
      NAME "python_${test_name}"
      COMMAND sh -c
              "${Python_EXECUTABLE} ${test_script} > ${test_log_file} 2>&1"
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

  else()
    add_test(
      NAME "python_${test_name}"
      COMMAND ${Python_EXECUTABLE} ${test_script}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  endif()

  set_tests_properties("python_${test_name}" PROPERTIES TIMEOUT 600)
endforeach()
