PROJECT( libtnl )

cmake_minimum_required( VERSION 2.8 )

SET( tnlVersion "0.1" )



#####
# Check for bzip2
#
FIND_PACKAGE(BZip2)
if( BZIP2_FOUND )
    set( HAVE_LIBBZ2 "" )    
else( BZIP2_FOUND )
    set( HAVE_LIBBZ2 "//" )
endif( BZIP2_FOUND )

#####
# Check for CUDA
#
FIND_PACKAGE( CUDA )
if( CUDA_FOUND )
    set( BUILD_CUDA TRUE)
    set( CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-DHAVE_CUDA )
else( CUDA_FOUND )
    
endif( CUDA_FOUND )

####
# Check for MPI
#
FIND_PACKAGE( MPI )
IF( MPI_FOUND )
    SET( BUILD_MPI TRUE )
ENDIF()
####
# Check for cppunit
#
FIND_PATH(CPPUNIT_INCLUDE_DIR cppunit/TestCase.h
  /usr/local/include
  /usr/include
)

# With Win32, important to have both
IF(WIN32)
  FIND_LIBRARY(CPPUNIT_LIBRARY cppunit
               ${CPPUNIT_INCLUDE_DIR}/../lib
               /usr/local/lib
               /usr/lib)
  FIND_LIBRARY(CPPUNIT_DEBUG_LIBRARY cppunitd
               ${CPPUNIT_INCLUDE_DIR}/../lib
               /usr/local/lib
               /usr/lib)
ELSE(WIN32)
  # On unix system, debug and release have the same name
  FIND_LIBRARY(CPPUNIT_LIBRARY cppunit
               ${CPPUNIT_INCLUDE_DIR}/../lib
               /usr/local/lib
               /usr/lib)
  FIND_LIBRARY(CPPUNIT_DEBUG_LIBRARY cppunit
               ${CPPUNIT_INCLUDE_DIR}/../lib
               /usr/local/lib
               /usr/lib)
ENDIF(WIN32)

IF(CPPUNIT_INCLUDE_DIR)
  IF(CPPUNIT_LIBRARY)
    SET(CPPUNIT_FOUND "YES")
    SET(CPPUNIT_LIBRARIES ${CPPUNIT_LIBRARY} ${CMAKE_DL_LIBS})
    SET(CPPUNIT_DEBUG_LIBRARIES ${CPPUNIT_DEBUG_LIBRARY}
${CMAKE_DL_LIBS})
  ENDIF(CPPUNIT_LIBRARY)
ENDIF(CPPUNIT_INCLUDE_DIR)


ENABLE_TESTING()
INCLUDE( Dart )

#Nastaveni kam davat vysledne soubory pro debug/release verzi
IF( CMAKE_BUILD_TYPE STREQUAL "Debug")
    set( PROJECT_BUILD_PATH ${PROJECT_SOURCE_DIR}/Debug/src )
	set( LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/Debug/lib )
	set( EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/Debug/bin )
	set( debugExt -dbg )
else()
    set( PROJECT_BUILD_PATH ${PROJECT_SOURCE_DIR}/Release/src )
	set( LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/Release/lib)
	set( EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/Release/bin)
	set( addAlsoSubdirs TRUE)
endif()

SET( configDirectory \"${CMAKE_INSTALL_PREFIX}/share/tnl-${tnlVersion}\")
CONFIGURE_FILE( "tnlConfig.h.in" "${PROJECT_BUILD_PATH}/tnlConfig.h" )

#Nastavime cesty k hlavickovym souborum a knihovnam
INCLUDE_DIRECTORIES( src )
INCLUDE_DIRECTORIES( ${PROJECT_BUILD_PATH} )
INCLUDE_DIRECTORIES( ${PROJECT_BINARY_DIR}/headers )
LINK_DIRECTORIES( ${LIBRARY_OUTPUT_PATH} )

#Pokracujeme dalsimi podadresari
add_subdirectory( src )
add_subdirectory( tests )
add_subdirectory( tools )

SET(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Template Numerical Library")
SET(CPACK_PACKAGE_VENDOR "MMG")
#SET(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/ReadMe.txt")
#SET(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/Copyright.txt")
SET(CPACK_PACKAGE_VERSION_MAJOR "0")
SET(CPACK_PACKAGE_VERSION_MINOR "1")
SET(CPACK_PACKAGE_VERSION_PATCH "0")
SET(CPACK_PACKAGE_INSTALL_DIRECTORY "CMake ${CMake_VERSION_MAJOR}.${CMake_VERSION_MINOR}")
IF(WIN32 AND NOT UNIX)
  # There is a bug in NSI that does not handle full unix paths properly. Make
  # sure there is at least one set of four (4) backlasshes.
  #SET(CPACK_PACKAGE_ICON "${CMake_SOURCE_DIR}/Utilities/Release\\\\InstallIcon.bmp")
  #SET(CPACK_NSIS_INSTALLED_ICON_NAME "bin\\\\MyExecutable.exe")
  SET(CPACK_NSIS_DISPLAY_NAME "${CPACK_PACKAGE_INSTALL_DIRECTORY} Template Numerical Library")
  SET(CPACK_NSIS_HELP_LINK "http:\\\\\\\\geraldine.fjfi.cvut.cz/~oberhuber/doku-wiki-tnl")
  SET(CPACK_NSIS_URL_INFO_ABOUT "http:\\\\\\\\geraldine.fjfi.cvut.cz/~oberhuber/doku-wiki-tnl")
  SET(CPACK_NSIS_CONTACT "tomas.oberhuber@fjfi.cvut.cz")
  SET(CPACK_NSIS_MODIFY_PATH ON)
ELSE(WIN32 AND NOT UNIX)
#  SET(CPACK_STRIP_FILES "bin/MyExecutable")
  SET(CPACK_SOURCE_STRIP_FILES "")
ENDIF(WIN32 AND NOT UNIX)
#SET(CPACK_PACKAGE_EXECUTABLES "MyExecutable" "My Executable")
INCLUDE( CPack )
