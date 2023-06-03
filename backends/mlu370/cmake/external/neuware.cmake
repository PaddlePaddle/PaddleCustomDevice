if(NOT ENV{NEUWARE_HOME})
  set(NEUWARE_HOME "/usr/local/neuware")
else()
  set(NEUWARE_HOME $ENV{NEUWARE_HOME})
endif()
message(STATUS "NEUWARE_HOME: " ${NEUWARE_HOME})

set(NEUWARE_INCLUDE_DIR ${NEUWARE_HOME}/include)
set(NEUWARE_LIB_DIR ${NEUWARE_HOME}/lib64)

include_directories(${NEUWARE_INCLUDE_DIR})

set(CNNL_LIB ${NEUWARE_LIB_DIR}/libcnnl.so)
set(CNRT_LIB ${NEUWARE_LIB_DIR}/libcnrt.so)
set(CNPAPI_LIB ${NEUWARE_LIB_DIR}/libcnpapi.so)
set(CNCL_LIB ${NEUWARE_LIB_DIR}/libcncl.so)

set(NEUWARE_LIBS ${CNNL_LIB} ${CNRT_LIB} ${CNPAPI_LIB} ${CNCL_LIB})

