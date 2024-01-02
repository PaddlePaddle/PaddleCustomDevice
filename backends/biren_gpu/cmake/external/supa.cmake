if(DEFINED ENV{SUPA_CUSTOM_PATH})
  set(SUPA_DIR $ENV{SUPA_CUSTOM_PATH})
else()
  set(SUPA_DIR /usr/local/supa)
endif()

set(SUPA_CL_DIR ${SUPA_DIR})
set(supa_cl_lib ${SUPA_CL_DIR}/lib/sucl/libsupa_cl.so)

set(SUPA_CL_INC_DIR ${SUPA_CL_DIR}/include/ ${SUPA_CL_DIR}/include/sucl/)

message(STATUS "SUPA_CL_INC_DIR ${SUPA_CL_INC_DIR}")
message(STATUS "SUPA_CL_DIR ${SUPA_CL_DIR}")

include_directories(${SUPA_CL_INC_DIR})
