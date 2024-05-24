# ##############################################################################
# ################             common ctest             ##################
# ##############################################################################
option(LABEL_CTEST "Generating CTest with Labels" OFF)
# include (add_py_test_n) include (add_cc_test_n)

# ##############################################################################
# # Remove a compiler flag from a specific build target #  _target - The target
# to remove the compile flag from #  _flag   - The compile flag to remove
# ##############################################################################
macro(remove_flag_from_target _target _flag)
  get_target_property(_target_cxx_flags ${_target} COMPILE_OPTIONS)
  if(_target_cxx_flags)
    list(REMOVE_ITEM _target_cxx_flags ${_flag})
    set_target_properties(${_target} PROPERTIES COMPILE_OPTIONS
                                                "${_target_cxx_flags}")
  endif()
endmacro()

function(append value)
  foreach(variable ${ARGN})
    set(${variable}
        "${${variable}} ${value}"
        PARENT_SCOPE)
  endforeach()
endfunction()

# ##############################################################################
# # Append the 'value' to all variables in ARGN list, if condition is true
# ##############################################################################
function(append_if condition value)
  if(${condition})
    foreach(variable ${ARGN})
      set(${variable}
          "${${variable}} ${value}"
          PARENT_SCOPE)
      message("variable = ${variable} ${value}")
    endforeach()
  endif()
endfunction()

# ##############################################################################
# # Add flag to both CMAKE_[C|C++]_FLAGS
# ##############################################################################
macro(add_flag_if_supported flag name)
  check_c_compiler_flag("-Werror ${flag}" "C_SUPPORTS_${name}")
  append_if("C_SUPPORTS_${name}" "${flag}" CMAKE_C_FLAGS)
  check_cxx_compiler_flag("-Werror ${flag}" "CXX_SUPPORTS_${name}")
  append_if("CXX_SUPPORTS_${name}" "${flag}" CMAKE_CXX_FLAGS)
endmacro()

# ##############################################################################
# # Add flag to both CMAKE_[C|C++]_FLAGS, if not support, will print a warn msg
# ##############################################################################
function(add_flag_or_print_warning flag name)
  check_c_compiler_flag("-Werror ${flag}" "C_SUPPORTS_${name}")
  check_cxx_compiler_flag("-Werror ${flag}" "CXX_SUPPORTS_${name}")
  if(C_SUPPORTS_${name} AND CXX_SUPPORTS_${name})
    message(STATUS "Building with ${flag}")
    set(CMAKE_CXX_FLAGS
        "${CMAKE_CXX_FLAGS} ${flag}"
        PARENT_SCOPE)
    set(CMAKE_C_FLAGS
        "${CMAKE_C_FLAGS} ${flag}"
        PARENT_SCOPE)
    set(CMAKE_ASM_FLAGS
        "${CMAKE_ASM_FLAGS} ${flag}"
        PARENT_SCOPE)
  else()
    message(WARNING "${flag} is not supported.")
  endif()
endfunction()

# ##############################################################################
# ################         SANITIZER: For Tests         ##################
# ##############################################################################
if(NOT DEFINED SANITIZER)
  set(SANITIZER "none")
elseif(NOT ${SANITIZER} STREQUAL "address_cmake")
  include(sanitizer)
endif()

unset(SANITIZER_ENVS) # Common sanitizer envs
unset(SANITIZER_ENVS_EXPORT) # Common sanitizer export commands, Add for all
                             # add_*_test
unset(SANITIZER_ENVS_PY) # Sanitizer envs for pytest
unset(SANITIZER_ENVS_PY_EXPORT) # Sanitizer export commands only for pytest
unset(SANITIZER_ENVS_CC) # Sanitizer envs for pytest
unset(SANITIZER_ENVS_CC_EXPORT) # Sanitizer export commands only for gtest
set(_SANITIZER_ENVS_LIST_ SANITIZER_ENVS SANITIZER_ENVS_PY SANITIZER_ENVS_CC)

set(suppression_dir ${CMAKE_SOURCE_DIR}/infra/cmake/module/sanitizer)

# LD_PRELOAD lib for asan and tsan
unset(_san_lib)
if(${SANITIZER} STREQUAL "address")
  set(_san_lib "libclang_rt.asan-x86_64.so")
elseif(${SANITIZER} STREQUAL "thread")
  set(_san_lib "libclang_rt.tsan-x86_64.so")
endif()

if(_san_lib)
  set(_san_lib_path ${CMAKE_SOURCE_DIR}/3rdparty/clang_binary/${_san_lib})
  if(NOT EXISTS ${_san_lib_path})
    message(FATAL_ERROR "[sanitizer] ${_san_lib_path} NOT EXIST!")
  endif()
  list(APPEND SANITIZER_ENVS_PY "LD_PRELOAD=`pwd`/lib/${_san_lib}")
endif()

if("${SANITIZER}" STREQUAL "address")
  if(NOT
     (EXISTS ${suppression_dir}/asan_suppressions.txt
      AND EXISTS ${suppression_dir}/lsan_cc_suppressions.txt
      AND EXISTS ${suppression_dir}/lsan_py_suppressions.txt))
    message(FATAL_ERROR "asan or lsan suppressions file not found")
  else()
    execute_process(
      COMMAND
        ${CMAKE_COMMAND} -E create_symlink
        ${suppression_dir}/asan_suppressions.txt
        ${CMAKE_BINARY_DIR}/asan_suppressions.txt
      COMMAND
        ${CMAKE_COMMAND} -E create_symlink
        ${suppression_dir}/lsan_cc_suppressions.txt
        ${CMAKE_BINARY_DIR}/lsan_cc_suppressions.txt
      COMMAND
        ${CMAKE_COMMAND} -E create_symlink
        ${suppression_dir}/lsan_py_suppressions.txt
        ${CMAKE_BINARY_DIR}/lsan_py_suppressions.txt)
    install(FILES ${suppression_dir}/asan_suppressions.txt DESTINATION .)
    install(FILES ${suppression_dir}/lsan_cc_suppressions.txt DESTINATION .)
    install(FILES ${suppression_dir}/lsan_py_suppressions.txt DESTINATION .)
    list(APPEND SANITIZER_ENVS
         "ASAN_OPTIONS=\"suppressions=`pwd`/asan_suppressions.txt\"")
    list(
      APPEND SANITIZER_ENVS_CC
      "LSAN_OPTIONS=\"suppressions=`pwd`/lsan_cc_suppressions.txt\":use_tls=0")
    list(
      APPEND SANITIZER_ENVS_PY
      "LSAN_OPTIONS=\"suppressions=`pwd`/lsan_py_suppressions.txt\":use_tls=0")
  endif()

  list(APPEND SANITIZER_ENVS "ASAN_SYMBOLIZER_PATH=`pwd`/bin/llvm-symbolizer")
elseif("${SANITIZER}" STREQUAL "thread")
  if(NOT EXISTS ${suppression_dir}/thread_suppressions.txt)
    message(FATAL_ERROR "tsan suppressions file not found")
  else()
    execute_process(
      COMMAND
        ${CMAKE_COMMAND} -E create_symlink
        ${suppression_dir}/thread_suppressions.txt
        ${CMAKE_BINARY_DIR}/thread_suppressions.txt)
    install(FILES ${suppression_dir}/thread_suppressions.txt DESTINATION .)
    list(APPEND SANITIZER_ENVS
         "TSAN_OPTIONS=\"suppressions=`pwd`/thread_suppressions.txt\"")
  endif()
endif()

set(RUN_WITHOUT_LEAKCHECK ${CMAKE_COMMAND} -E env
                          "ASAN_OPTIONS=detect_leaks=0:detect_odr_violation=0")

macro(fresh_sanitizer)
  foreach(envs IN LISTS _SANITIZER_ENVS_LIST_)
    if(DEFINED ${envs} AND NOT ("${${envs}}" STREQUAL ""))
      set(_sanitizer_export_ ${${envs}})
      list(TRANSFORM _sanitizer_export_ PREPEND "export ")
      list(JOIN _sanitizer_export_ " && " ${envs}_EXPORT)
      string(APPEND ${envs}_EXPORT " &&")
      set(${envs}_EXPORT
          "${${envs}_EXPORT}"
          CACHE STRING "")
      unset(_sanitizer_export_)
    endif()
  endforeach()
endmacro()
fresh_sanitizer()

set(EXPECTED_ARCH_LIST
    default x86 aarch64
    CACHE INTERNAL "" FORCE)
set(EXPECTED_CATEGORY_LIST
    func perf convergence stability
    CACHE INTERNAL "" FORCE)
set(EXPECTED_PLATFORM_LIST
    vdk
    vdk1x
    edk
    silicon
    distrib
    cpu
    cpu
    null
    kvm
    s6
    s30
    s60
    s90
    virtual
    sanity
    CACHE INTERNAL "" FORCE)
set(EXPECTED_PROJECT_LIST
    leo
    vela
    pavo
    dorado
    scorpio
    libra
    galaxy
    pavo_galaxy
    CACHE INTERNAL "" FORCE)
set(EXPECTED_REGRESSION_LIST
    ci
    daily
    weekly
    biweekly
    triweekly
    release
    sanity
    preci
    modelzoo
    null
    CACHE INTERNAL "" FORCE)
set(EXPECTED_OS_LIST
    ubuntu
    tlinux
    redhat
    centos
    ubuntuhost
    ubuntu1604
    ubuntu1804
    ubuntu2004
    ubuntu2204
    kylin
    uos
    euler
    anolis
    redhat9
    CACHE INTERNAL "" FORCE)

function(is_subset superset subset result)
  cmake_parse_arguments("" "WARNING;QUIET" "" "" ${ARGN})
  if(${superset} AND ${subset})
    foreach(_s IN LISTS ${subset})
      if(NOT ("${_s}" IN_LIST ${superset}))
        if(NOT _QUIET)
          if(_WARNING)
            message(WARNING "[${_s}] is not in LIST:${superset}")
          else()
            message(FATAL_ERROR "[${_s}] is not in LIST:${superset}")
          endif()
        endif()
        set(${result}
            FALSE
            PARENT_SCOPE)
        return()
      endif()
    endforeach()
  endif()
  set(${result}
      TRUE
      PARENT_SCOPE)
endfunction()

set(ICEST_SUBS CATEGORY PROJECT PLATFORM REGRESSION OS ARCH)

function(__add_test_)
  set(options
      OPTIONAL
      OVERRIDE
      WILLFAIL
      NO_MONITOR
      NO_TIMEOUT_CHECK
      TEST_W_LABEL
      DISABLE_HOST_COMPILE)
  set(oneValueArgs
      ID
      NAME
      TIMEOUT
      VDK_TIMEOUT
      EDK_TIMEOUT
      ARM_TIMEOUT
      PASS_EXPRESSION
      SETUP
      COMMAND
      CLEANUP
      WORKING_DIRECTORY
      ZEBU_DATABASE
      ENABLE_PROFILER)
  set(multiValueArgs ${ICEST_SUBS} PY_VER ENVIRONMENT MODULE)
  cmake_parse_arguments(X_TEST "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN})

  string(TOLOWER "${X_TEST_MODULE}" X_TEST_MODULE)
  string(TOLOWER "${X_TEST_NAME}" X_TEST_NAME)

  # Default Values.
  if(NOT X_TEST_PROJECT OR "${X_TEST_PROJECT}" STREQUAL "all")
    set(X_TEST_PROJECT ${EXPECTED_PROJECT_LIST})
  endif()
  if(NOT X_TEST_PLATFORM OR "${X_TEST_PLATFORM}" STREQUAL "dtu")
    set(X_TEST_PLATFORM vdk vdk1x edk silicon)
  endif()
  if(NOT X_TEST_REGRESSION)
    set(X_TEST_REGRESSION null)
  elseif(CI_TEST_ONLY)
    if("ci" IN_LIST X_TEST_REGRESSION)
      set(X_TEST_REGRESSION "ci")
    else()
      return()
    endif()
  endif()
  if(NOT X_TEST_CATEGORY)
    set(X_TEST_CATEGORY func)
  endif()
  if(NOT X_TEST_OS)
    set(X_TEST_OS ubuntu)
  endif()
  if(NOT X_TEST_ARCH)
    set(X_TEST_ARCH default)
  endif()

  foreach(subs ${ICEST_SUBS})
    is_subset(EXPECTED_${subs}_LIST X_TEST_${subs} _is_sub)
  endforeach()

  # for runtime 3.0
  list(APPEND X_TEST_ENVIRONMENT
       "LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/opt/tops/lib")

  if(NOT X_TEST_COMMAND)
    set(X_TEST_COMMAND test)
  endif()
  if(X_TEST_TIMEOUT)
    set(_TIME_OUT_DEFAULT ${X_TEST_TIMEOUT})
  endif()
  if(NOT X_TEST_EDK_TIMEOUT)
    set(X_TEST_EDK_TIMEOUT 240000)
  endif()
  if(NOT X_TEST_VDK_TIMEOUT)
    set(X_TEST_VDK_TIMEOUT 240000)
  endif()
  if(NOT X_TEST_ARM_TIMEOUT)
    set(X_TEST_ARM_TIMEOUT 3600)
  endif()
  # WORKING_DIRECTORY select condition branch TODO : Need to know how to set it
  # up.
  if(X_TEST_WORKING_DIRECTORY)
    set(workdir ./${X_TEST_WORKING_DIRECTORY})
  else()
    set(workdir ./)
  endif()

  # joint Zebu database upload fixture
  if(X_TEST_ZEBU_DATABASE)
    set(X_TEST_COMMAND
        "pmon_start ${X_TEST_ZEBU_DATABASE}\
                            ; ${X_TEST_COMMAND}\
                            ; ret_value=\$?\
                            ; pmon_stop\
                            ; sync\
                            ; pmon_report ${X_TEST_ZEBU_DATABASE}")
    set(bash_cmd /bin/bash -c -i)
  else()
    set(bash_cmd /bin/bash -c)
  endif()

  unset(ENV)
  # support runtime 3.0 list(APPEND X_TEST_ENVIRONMENT
  # "LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/opt/tops/lib")
  foreach(env IN LISTS X_TEST_ENVIRONMENT)
    string(APPEND ENV "${env};")
  endforeach()

  if(X_TEST_SETUP AND X_TEST_CLEANUP)
    set(_FIXTURES_REQUIRED "${X_TEST_SETUP};${X_TEST_CLEANUP}")
  elseif(X_TEST_SETUP)
    set(_FIXTURES_REQUIRED "${X_TEST_SETUP}")
  elseif(X_TEST_CLEANUP)
    set(_FIXTURES_REQUIRED "${X_TEST_CLEANUP}")
  endif()

  unset(X_TEST_ORGIN_COMMAND)
  set(X_TEST_ORGIN_COMMAND "${X_TEST_COMMAND}")

  foreach(proj IN LISTS X_TEST_PROJECT)
    # enable profiler
    if(X_TEST_ENABLE_PROFILER)
      if("${proj}" STREQUAL "pavo")
        set(enable_engine "cqm")
      else()
        set(enable_engine "cdma,cqm,sdma,odma,ts,sip")
      endif()
      if("${proj}" STREQUAL "scorpio")
        foreach(module IN LISTS X_TEST_MODULE)
          if("${module}" STREQUAL "vllm_whole_network")
            set(X_TEST_COMMAND
                "topsprof --reset\
                                ; topsprof --force-overwrite --print-app-log --enable-activities operator --trace all --topstx-domain-include all --buffer host --export-rawdata ${X_TEST_NAME}.data --export-visual-profiler ./${X_TEST_NAME} bash -c '${X_TEST_ORGIN_COMMAND}'"
            )
          else()
            set(X_TEST_COMMAND
                "topsprof --reset\
                                ; topsprof --force-overwrite --print-app-log --enable-activities all --buffer device --export-rawdata ${X_TEST_NAME}.data --export-visual-profiler ./${X_TEST_NAME} bash -c '${X_TEST_ORGIN_COMMAND}'"
            )
          endif()
        endforeach()
      else()
        set(X_TEST_COMMAND
            "export EFRT_PROFILING_ENABLE=true\
                                ; efsmt -dpm level=80\
                                ; efsmi --ppo off\
                                ; efsmi --ppo status\
                                ; efsmt -clock list \
                                ; topsprof --reset\
                                ; topsprof --force-overwrite --print-app-log --enable-activities '*/general/operator' --host-ringbuffer-size 64 --watermark-threshold 63 --buffer host --export-rawdata ${X_TEST_NAME}.data --export-csv ${X_TEST_NAME}.csv --export-visual-profiler ./${X_TEST_NAME} bash -c '${X_TEST_ORGIN_COMMAND}'\
                                ; ret_value=\$?\
                                ; /bin/bash -c \"for bdf in `lspci -d 1ea0:* | awk '{print $1}'`; do echo 1 > /sys/bus/pci/devices/0000:\\\${bdf}/gcu_reload; done\"\
                                ; /bin/bash -c \"for bdf in `lspci -d 1e36:* | awk '{print $1}'`; do echo 1 > /sys/bus/pci/devices/0000:\\\${bdf}/gcu_reload; done\""
        )
      endif()
    endif()
    # return saved return value if profiler or zebu databse enabled
    if(X_TEST_ENABLE_PROFILER OR X_TEST_ZEBU_DATABASE)
      set(X_TEST_COMMAND "${X_TEST_COMMAND}\
                                ; exit \$ret_value")
    endif()

    foreach(plat IN LISTS X_TEST_PLATFORM)
      if("${plat}" STREQUAL "edk")
        set(_timeout_plat ${X_TEST_EDK_TIMEOUT})
      elseif("${plat}" MATCHES "vdk")
        set(_timeout_plat ${X_TEST_VDK_TIMEOUT})
      else()
        set(_timeout_plat ${_TIME_OUT_DEFAULT})
      endif()

      foreach(regress IN LISTS X_TEST_REGRESSION)
        foreach(arch IN LISTS X_TEST_ARCH)
          if("${arch}" STREQUAL "aarch64")
            set(_timeout ${X_TEST_ARM_TIMEOUT})
          else()
            set(_timeout ${_timeout_plat})
          endif()

          foreach(os IN LISTS X_TEST_OS)
            foreach(category IN LISTS X_TEST_CATEGORY)
              foreach(module IN LISTS X_TEST_MODULE)
                if("${arch}" STREQUAL "default")
                  # default name not contain ${arch} for CI
                  set(TEST_NAME
                      ${proj}_${plat}_${regress}_${category}_${os}_${module}_${X_TEST_ID}_${X_TEST_NAME}
                  )
                  set(_label
                      "${proj},${plat},${regress},${category},${os},,${module},${X_TEST_ID},${X_TEST_NAME}"
                  )
                else()
                  set(TEST_NAME
                      ${proj}_${plat}_${regress}_${category}_${os}_${arch}_${module}_${X_TEST_ID}_${X_TEST_NAME}
                  )
                  set(_label
                      "${proj},${plat},${regress},${category},${os},${arch},${module},${X_TEST_ID},${X_TEST_NAME}"
                  )
                endif()
                add_test(
                  NAME ${TEST_NAME}
                  COMMAND ${bash_cmd} "${X_TEST_COMMAND}"
                  WORKING_DIRECTORY ${workdir})
                file(
                  APPEND ${CMAKE_BINARY_DIR}/sqlite.txt
                  "\"${proj}\",\"${plat}\", \"${regress}\", \",${category}\", \"${os}\", \"${arch}\", \"${module}\", \"${X_TEST_ID}\", \"${X_TEST_NAME}\", \"${TEST_NAME}\", \"${X_TEST_COMMAND}\", \"${workdir}\", \"${_timeout}\"\n"
                )
                if(ENV)
                  set_tests_properties(${TEST_NAME} PROPERTIES ENVIRONMENT
                                                               "${ENV}")
                endif()
                if(X_TEST_PASS_EXPRESSION)
                  set_tests_properties(
                    ${TEST_NAME} PROPERTIES PASS_REGULAR_EXPRESSION
                                            "${X_TEST_PASS_EXPRESSION}")
                  # else () set_tests_properties (${TEST_NAME} PROPERTIES
                  # SKIP_RETURN_CODE 2)
                endif()
                if(_timeout)
                  set_tests_properties(${TEST_NAME} PROPERTIES TIMEOUT
                                                               ${_timeout})
                endif()
                if(X_TEST_WILLFAIL)
                  set_tests_properties(${TEST_NAME} PROPERTIES WILL_FAIL true)
                endif()
                if(_FIXTURES_REQUIRED)
                  set_tests_properties(
                    ${proj}${suite}_${X_TEST_ID}${X_TEST_NAME}
                    PROPERTIES FIXTURES_REQUIRED "${_FIXTURES_REQUIRED}")
                endif()

                if(X_TEST_UNPARSED_ARGUMENTS)
                  message(
                    FATAL_ERROR
                      "${X_TEST_UNPARSED_ARGUMENTS} - parameters are not supported!"
                  )
                endif()
                set_tests_properties(${TEST_NAME} PROPERTIES LABELS "${_label}")
              endforeach()
            endforeach()
          endforeach()
        endforeach()
      endforeach()
    endforeach()
  endforeach()
endfunction()

#
# Add CTest Function
#
# ADD_SW_TEST PARAM: add_sw_test( ID                 test_ID, e.g. GDMA.1.1 NAME
# test_name [PROJECT]          leo|vela|pavo|dorado|scorpio|all, test could run
# on which project, default is all [PLATFORM]
# vdk|vdk1x|edk|silicon|distri|cpu|dtu, test could run on which platform,
# dtu=vdkedksilicon, default is dtu [REGRESSION]
# ci|daily|weekly|release|preci|modelzoo, test will run in which regression set,
# default is daily [CATEGORY]         func|robust|perf, test belongs to which
# category, default is func [OS]               u16|u20|c7, test will run with
# which OS, u16=ubuntu16.04, c7=centOS 7 COMMAND            test command line
# [PY_VER]           py27|py35|py36|py37, test will run with which version of
# python, py27=python2.7 [EXTRA_PARAM]      additional parameter for efvs e.g.
# "-v debug" [TIMEOUT]          time_in_second, set timeout to override default
# value(1200s) [VDK_TIMEOUT]      time_in_second, set timeout to override
# default value(240000s) [EDK_TIMEOUT]      time_in_second, set timeout to
# override default value(240000s) [PASS_EXPRESSION]  "pass regular expression in
# log", change the pass/fail criteria to parse log [SETUP]            set up
# fixtures [CLEANUP]          clean up fixtures [OVERRIDE]         override
# default parameter with EXTRA_PARAM [WILLFAIL]         test ctest expect result
# is fail (return code > 0) [NO_MONITOR]       per test request do not enable
# any system monitor in parallel [NO_TIMEOUT_CHECK] normally slt test need to be
# less than 120s, this option is used for special case which over 120s )
#

function(add_py_test)
  cmake_parse_arguments(X_TEST "" "COMMAND" "" ${ARGN})
  # generating ctest with label instead of combo if (X_TEST_TEST_W_LABEL OR
  # TEST_W_LABEL) add_py_test_n(${ARGN}) return() endif ()

  set(py_module_reg [[python[0-9\.]*[ \t\r\n\]+-m[ \t\r\n]+pytest]])
  set(pytest_reg [[pytest]])

  string(REGEX MATCH ${py_module_reg} _match_cmd "${X_TEST_COMMAND}")
  if(NOT _match_cmd)
    string(REGEX MATCH ${pytest_reg} _match_cmd "${X_TEST_COMMAND}")
  endif()
  if(_match_cmd)
    set(_sanitizer_env "${SANITIZER_ENVS_EXPORT} ${SANITIZER_ENVS_PY_EXPORT}")
    string(REPLACE "`pwd`" "$_test_prefix" _sanitizer_env "${_sanitizer_env}")
    string(REPLACE "${_match_cmd}" "${_sanitizer_env} ${_match_cmd}"
                   X_TEST_COMMAND "${X_TEST_COMMAND}")
    set(X_TEST_COMMAND "_test_prefix=`pwd` && ${X_TEST_COMMAND}")
  else()
    set(X_TEST_COMMAND "${SANITIZER_ENVS_PY_EXPORT} ${X_TEST_COMMAND}")
    set(X_TEST_COMMAND "${SANITIZER_ENVS_EXPORT} ${X_TEST_COMMAND}")
  endif()

  __add_test_(${ARGN} COMMAND ${X_TEST_COMMAND})
endfunction()

# function (add_cc_test) cmake_parse_arguments(X_TEST "" "COMMAND" "" ${ARGN})
#
# # generating ctest with label instead of combo if (X_TEST_TEST_W_LABEL OR
# TEST_W_LABEL) add_cc_test_n(${ARGN}) return() endif ()
#
# set (X_TEST_COMMAND "${SANITIZER_ENVS_CC_EXPORT} ${X_TEST_COMMAND}") set
# (X_TEST_COMMAND "${SANITIZER_ENVS_EXPORT} ${X_TEST_COMMAND}")
#
# __add_test_(${ARGN} COMMAND ${X_TEST_COMMAND}) endfunction ()

#
# This function is to add a submodule folder into CMake Build System. It will
# check if submodule init or not before calling add_subdirectory() Param1:
# submodule foler name Param2: tagfile to check if submodule exists
#
function(add_submodule_directory dir)
  if(${ARGC} GREATER 1)
    set(file_to_check ${ARGV1})
  else()
    set(file_to_check CMakeLists.txt)
  endif()

  set(tagfile ${CMAKE_CURRENT_SOURCE_DIR}/${dir}/${file_to_check})
  if(NOT EXISTS ${tagfile})
    message(
      WARNING
        "tagfile ${tagfile} doesn't exist, it seems submodule not init yet")
    message(
      FATAL_ERROR
        "Seems submodule '${dir}' not init, you MAY run following command to init:\n 'cd ${CMAKE_SOURCE_DIR} && git submodule update --init --recursive'\n"
    )
  else()
    add_subdirectory(${dir})
  endif()
endfunction()

option(DISABLE_PIP_CONFIG_CHECK "disable the ~/.pip/pip.conf file check" OFF)

macro(python_pip_config)
  if(NOT DISABLE_PIP_CONFIG_CHECK)
    if(NOT EXISTS $ENV{HOME}/.pip/pip.conf)
      message(
        WARNING
          "There is no local pip config file for better build performance, please setup as following:
######################################################################################
$ENV{HOME}/.pip/pip.conf:
\[global\]
  index-url = http://artifact.enflame.cn/artifactory/api/pypi/pypi-remote/simple
\[install\]
  trusted-host = artifact.enflame.cn
######################################################################################"
      )
      message(FATAL_ERROR "please setup python pip.conf for build performance")
    endif()
  endif()
endmacro()

#
# This function is to get current cmake args for a build tree and write the
# cmake args to .config file in the build tree There are 2 purpose: a. to record
# the config for the build tree in case need to know b. to re-config the build
# tree when the build tree break sometimes
#
function(write_cmake_args_to_build_tree)
  if(${CMAKE_SYSTEM_NAME} STREQUAL Linux)
    file(STRINGS /proc/self/status _cmake_process_status)

    # Grab the PID of the parent process
    string(REGEX MATCH "PPid:[ \t]*([0-9]*)" _ ${_cmake_process_status})
    set(pexec /proc/${CMAKE_MATCH_1}/exe)

    if(EXISTS ${pexec})
      # Grab the absolute path of the parent process
      file(READ_SYMLINK ${pexec} _cmake_parent_process_path)

      # Compute CMake arguments only if CMake was not invoked by the native
      # build system, to avoid dropping user specified options on re-triggers.
      if(NOT ${_cmake_parent_process_path} STREQUAL ${CMAKE_MAKE_PROGRAM})
        execute_process(COMMAND bash -c "tr '\\0' ' ' < /proc/$PPID/cmdline"
                        OUTPUT_VARIABLE _cmake_args)
        string(STRIP "${_cmake_args}" CMAKE_ARGS)
        file(READ_SYMLINK "/proc/self/cwd" CMAKE_CWD)
        if(NOT "${CMAKE_ARGS}" MATCHES "regenerate-during-build")
          file(WRITE ${CMAKE_BINARY_DIR}/.config
               "cd ${CMAKE_CWD} && ${CMAKE_ARGS}")
        endif()
      endif()
    endif()
  endif()
endfunction()
