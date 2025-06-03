## Set package generator,
set ( CPACK_GENERATOR ${PACKAGE_GENERATOR} CACHE STRING "Package types to build")

## Only pack the "binary" and "dev" components, post install script will add the directory link.
set ( CPACK_COMPONENTS_ALL_IN_ONE_PACKAGE 1 )
set ( CPACK_COMPONENTS_ALL binary dev )

set ( CPACK_PACKAGE_NAME ${LIBRARY_NAME}_${MACA_VERSION} )
set ( CPACK_PACKAGE_VERSION ${MACA_VERSION} )
set ( CPACK_PACKAGE_CONTACT "https://www.metax-tech.com" )
set ( CPACK_PACKAGE_VENDOR "MetaX Integrated Circuits (Shanghai) Co., Ltd" )
set ( CPACK_PACKAGE_DESCRIPTION_SUMMARY "maca ${LIBRARY_NAME} lib package." )
set ( CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.md" )
set ( CPACK_PACKAGE_DIRECTORY  "${CMAKE_PACKAGE_DIR}")
set ( CPACK_SET_DESTDIR ON )
set ( CPACK_INSTALL_PREFIX /opt/maca-${MACA_VERSION}/ )
set ( CPACK_OUTPUT_FILE_PREFIX "${CMAKE_INSTALL_PREFIX}/${PACKAGE_EXT}" CACHE STRING "Default output file prefix.")
set ( CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}${DISTRO_CODE}.${PACKAGE_ARCH}" )

if ( ${CPACK_GENERATOR} STREQUAL "DEB" )
  set ( CPACK_DEBIAN_PACKAGE_ARCHITECTURE ${PACKAGE_ARCH})
elseif ( ${CPACK_GENERATOR} STREQUAL "RPM" )
  set ( CPACK_RPM_PACKAGE_ARCHITECTURE ${PACKAGE_ARCH})
endif()

set ( CPACK_DEBIAN_PACKAGE_HOMEPAGE "https://www.metax-tech.com" )
set ( CPACK_DEBIAN_PACKAGE_MAINTAINER "mxmaca@metax-tech.com")
#set ( CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA "DEBIAN/postinst;DEBIAN/prerm" )

## Process the install scripts to update the CPACK variables
#configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/DEBIAN/post_install DEBIAN/postinst @ONLY)
#configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/DEBIAN/pre_remove DEBIAN/prerm @ONLY)

set ( CPACK_RPM_PACKAGE_NAME ${LIBRARY_NAME})
set ( CPACK_RPM_PACKAGE_RELEASE ${BUILD_NUMBER})

# 'dist' breaks manual builds on debian systems due to empty Provides
execute_process( COMMAND rpm --eval %{?dist}
                 RESULT_VARIABLE PROC_RESULT
                 OUTPUT_VARIABLE EVAL_RESULT
                 OUTPUT_STRIP_TRAILING_WHITESPACE )
message("RESULT_VARIABLE ${PROC_RESULT} OUTPUT_VARIABLE: ${EVAL_RESULT}")

if ( PROC_RESULT EQUAL "0" AND NOT EVAL_RESULT STREQUAL "" )
  string ( APPEND CPACK_RPM_PACKAGE_RELEASE "%{?dist}" )
endif()

#set ( CPACK_RPM_POST_INSTALL_SCRIPT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/cmake/RPM/rpm_post" )
#set ( CPACK_RPM_POST_UNINSTALL_SCRIPT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/cmake/RPM/rpm_postun" )

set( CPACK_INSTALL_CMAKE_PROJECTS
  "${CMAKE_CURRENT_BINARY_DIR};${CMAKE_PROJECT_NAME};Devel;/"
)

## Include packaging
include ( CPack )