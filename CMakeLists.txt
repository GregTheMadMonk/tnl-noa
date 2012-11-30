cmake_minimum_required( VERSION 2.8 )

project( tnl )

set( tnlVersion "0.1" )

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

include( OptimizeForArchitecture )

####
# Settings for debug/release version
#
if( CMAKE_BUILD_TYPE STREQUAL "Debug")
    set( PROJECT_BUILD_PATH ${PROJECT_SOURCE_DIR}/Debug/src )
    set( PROJECT_TESTS_PATH ${PROJECT_SOURCE_DIR}/Debug/tests )
    set( LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/Debug/lib )
    set( EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/Debug/bin )
    set( debugExt -dbg )
    AddCompilerFlag( "-std=gnu++0x" )
else()
    set( PROJECT_BUILD_PATH ${PROJECT_SOURCE_DIR}/Release/src )
    set( PROJECT_TESTS_PATH ${PROJECT_SOURCE_DIR}/Release/tests )
    set( LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/Release/lib)
    set( EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/Release/bin)
    #set( CXX_OPTIMIZE_FLAGS "-O3")
    OptimizeForArchitecture()
    AddCompilerFlag( "-std=gnu++0x -O3 -DNDEBUG" )
endif()

#####
# Check for bzip2
#
find_package(BZip2)
if( BZIP2_FOUND )
    set( HAVE_LIBBZ2 "#define HAVE_BZIP2" )    
else( BZIP2_FOUND )
    set( HAVE_LIBBZ2 "// #define HAVE_BZIP2" )
endif( BZIP2_FOUND )

#####
# Check for CUDA
#
if( NOT WITH_CUDA STREQUAL "no" )
    find_package( CUDA )
    if( CUDA_FOUND )
        set( BUILD_CUDA TRUE)
        set( CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-DHAVE_CUDA )
        if( CUDA_ARCHITECTURE STREQUAL "1.0" )
            set( CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=sm_10;-DCUDA_ARCH=10)
        endif()
        if( CUDA_ARCHITECTURE STREQUAL "1.1" )
            set( CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=sm_11;-DCUDA_ARCH=11)
        endif()    
        if( CUDA_ARCHITECTURE STREQUAL "1.2" )
            set( CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=sm_12;-DCUDA_ARCH=12)
        endif()    
        if( CUDA_ARCHITECTURE STREQUAL "1.3" )
            set( CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=sm_13;-DCUDA_ARCH=13)
        endif()    
        if( CUDA_ARCHITECTURE STREQUAL "2.0" )
            set( CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=sm_20;-DCUDA_ARCH=20)
        endif()    
        if( CUDA_ARCHITECTURE STREQUAL "2.1" )
            set( CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=sm_21;-DCUDA_ARCH=21)
        endif()    
        if( CUDA_ARCHITECTURE STREQUAL "2.2" )
            set( CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=sm_22;-DCUDA_ARCH=22)
        endif()    
        if( CUDA_ARCHITECTURE STREQUAL "2.3" )
            set( CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=sm_23;-DCUDA_ARCH=23)
        endif()    
        if( CUDA_ARCHITECTURE STREQUAL "3.0" )
            set( CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=sm_30;-DCUDA_ARCH=30)
        endif()    
    
        ####
        # Check for CUSP
        #
        find_path( CUSP_INCLUDE_DIR cusp/hyb_matrix.h  
                   DOC "CUSP headers." )
        if( ${CUSP_INCLUDE_DIR} STREQUAL "CUSP_INCLUDE_DIR-NOTFOUND" )
            message( "CUSP not found." )
            set( HAVE_CUSP "//#define HAVE_CUSP" )
        else()      
            message( "CUSP found. -- ${CUSP_INCLUDE_DIR}" )
            set( HAVE_CUSP "#define HAVE_CUSP" )
            cuda_include_directories( ${CUSP_INCLUDE_DIR} )
        endif()
        
        ####
        # Check for CUSPARSE
        #
        if( NOT WITH_CUSPARSE STREQUAL "no" )
           find_path( CUSPARSE_INCLUDE_DIR cusparse.h
                      /usr/local/cuda/include                   
                      ${CUDA_INCLUDE_DIR}  
                      DOC "CUSPARSE headers." )
           if( ${CUSPARSE_INCLUDE_DIR} STREQUAL "CUSPARSE_INCLUDE_DIR-NOTFOUND" )
               message( "CUSPARSE not found." )
               set( HAVE_CUSPARSE "//#define HAVE_CUSPARSE" )
           else()      
               message( "CUSPARSE found. -- ${CUSPARSE_INCLUDE_DIR}" )
               set( HAVE_CUSPARSE "#define HAVE_CUSPARSE" )
               cuda_include_directories( ${CUSPARSE_INCLUDE_DIR} )
           endif() 
           
           string( REPLACE "/include" "" CUSPARSE_LINK_DIR_HINT ${CUSPARSE_INCLUDE_DIR} )           
           find_library( CUSPARSE_LIBRARY
                         NAME cusparse 
                         HINTS ${CUSPARSE_LINK_DIR_HINT}/lib
                         DOC "CUSPARSE library" )                                                                  
           if( ${CUSPARSE_LIBRARY} STREQUAL "CUSPARSE_LIBRARY-NOTFOUND" )
               message( "Cannot find CUSPARSE library." )
           endif()    
           string( REPLACE "/libcusparse.so" "" CUSPARSE_LINK_DIRECTORY ${CUSPARSE_LIBRARY} )
           link_directories( ${CUSPARSE_LINK_DIRECTORY} )
        endif( NOT WITH_CUSPARSE STREQUAL "no" )
            
    endif( CUDA_FOUND )
endif( NOT WITH_CUDA STREQUAL "no" )    

####
# Check for OpenMP
#
find_package( OpenMP ) 
if( OPENMP_FOUND )
# TODO: finish this
endif()

####
# Check for MPI
#
find_package( MPI )
if( MPI_FOUND )
    set( BUILD_MPI TRUE )
endif()

####
# Check for some system header
#
find_path( SYS_TIME_INCLUDE_DIR sys/time.h  
           /usr/include/x86_64-linux-gnu
           /usr/include
           DOC "System timer headers." )
if( ${SYS_TIME_INCLUDE_DIR} STREQUAL "SYS_TIME_INCLUDE_DIR-NOTFOUND" )
    message(  "Missing header file sys/time.h" )
    set( HAVE_SYS_TIME_H "//#define HAVE_SYS_TIME_H" )
else()
    include_directories( ${SYS_TIME_INCLUDE_DIR}/tnl-${tnlVersion} )
    set( HAVE_SYS_TIME_H "#define HAVE_SYS_TIME_H" )
endif()

find_path( SYS_RESOURCE_INCLUDE_DIR sys/resource.h
           /usr/include/x86_64-linux-gnu
           /usr/include  
           DOC "System resources headers." )
if( ${SYS_RESOURCE_INCLUDE_DIR} STREQUAL "SYS_RESOURCE_INCLUDE_DIR-NOTFOUND" )
    message( "Missing header file sys/time.h" )
    set( HAVE_SYS_RESOURCE_H "//#define HAVE_SYS_RESOURCE_H" )
else()
    include_directories( ${SYS_RESOURCE_INCLUDE_DIR}/tnl-${tnlVersion} )
    set( HAVE_SYS_RESOURCE_H "#define HAVE_SYS_RESOURCE_H" )
endif()

  


####
# Check for cppunit
#
FIND_PATH(CPPUNIT_INCLUDE_DIR cppunit/TestCase.h
  /usr/local/include
  /usr/include
  DOC "CppUnit headers."
)

####
# With Win32, important to have both
#
if(WIN32)
  FIND_LIBRARY(CPPUNIT_LIBRARY cppunit
               ${CPPUNIT_INCLUDE_DIR}/../lib
               /usr/local/lib
               /usr/lib)
  FIND_LIBRARY(CPPUNIT_DEBUG_LIBRARY cppunitd
               ${CPPUNIT_INCLUDE_DIR}/../lib
               /usr/local/lib
               /usr/lib)
else(WIN32)
  # On unix system, debug and release have the same name
  FIND_LIBRARY(CPPUNIT_LIBRARY cppunit
               ${CPPUNIT_INCLUDE_DIR}/../lib
               /usr/local/lib
               /usr/lib)
  FIND_LIBRARY(CPPUNIT_DEBUG_LIBRARY cppunit
               ${CPPUNIT_INCLUDE_DIR}/../lib
               /usr/local/lib
               /usr/lib)
endif(WIN32)


if( ${CPPUNIT_INCLUDE_DIR} STREQUAL "CPPUNIT_INCLUDE_DIR-NOTFOUND" )
      message( "CPPUNIT not found." )
      set( HAVE_CPPUNIT "//#define HAVE_CPPUNIT" )
else( ${CPPUNIT_INCLUDE_DIR} STREQUAL "CPPUNIT_INCLUDE_DIR-NOTFOUND" )
  if(CPPUNIT_LIBRARY)
    message( "CPPUNIT found. -- ${CPPUNIT_INCLUDE_DIR}" )
    set(CPPUNIT_FOUND "YES")
    set(CPPUNIT_LIBRARIES ${CPPUNIT_LIBRARY} ${CMAKE_DL_LIBS})
    set(CPPUNIT_DEBUG_LIBRARIES ${CPPUNIT_DEBUG_LIBRARY}
                                ${CMAKE_DL_LIBS})
   set( HAVE_CPPUNIT "#define HAVE_CPPUNIT" )
  endif(CPPUNIT_LIBRARY)
endif( ${CPPUNIT_INCLUDE_DIR} STREQUAL "CPPUNIT_INCLUDE_DIR-NOTFOUND" )

ENABLE_TESTING()
INCLUDE( Dart )

set( configDirectory \"${CMAKE_INSTALL_PREFIX}/share/tnl-${tnlVersion}/\")
set( sourceDirectory \"${PROJECT_SOURCE_DIR}/\" )
set( testsDirectory \"${PROJECT_TESTS_PATH}/\" )
CONFIGURE_FILE( "tnlConfig.h.in" "${PROJECT_BUILD_PATH}/tnlConfig.h" )
INSTALL( FILES ${PROJECT_BUILD_PATH}/tnlConfig.h DESTINATION include/tnl-${tnlVersion} )

#Nastavime cesty k hlavickovym souborum a knihovnam
INCLUDE_DIRECTORIES( src )
INCLUDE_DIRECTORIES( ${PROJECT_BUILD_PATH} )
LINK_DIRECTORIES( ${LIBRARY_OUTPUT_PATH} )

#Pokracujeme dalsimi podadresari
add_subdirectory( src )
add_subdirectory( tests )
add_subdirectory( tools )

set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Template Numerical Library")
set(CPACK_PACKAGE_VENDOR "MMG")
#set(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/ReadMe.txt")
#set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/Copyright.txt")
set(CPACK_PACKAGE_VERSION_MAJOR "0")
set(CPACK_PACKAGE_VERSION_MINOR "1")
set(CPACK_PACKAGE_VERSION_PATCH "0")
set(CPACK_PACKAGE_INSTALL_DIRECTORY "CMake ${CMake_VERSION_MAJOR}.${CMake_VERSION_MINOR}")
if(WIN32 AND NOT UNIX)
  # There is a bug in NSI that does not handle full unix paths properly. Make
  # sure there is at least one set of four (4) backlasshes.
  #set(CPACK_PACKAGE_ICON "${CMake_SOURCE_DIR}/Utilities/Release\\\\InstallIcon.bmp")
  #set(CPACK_NSIS_INSTALLED_ICON_NAME "bin\\\\MyExecutable.exe")
  set(CPACK_NSIS_DISPLAY_NAME "${CPACK_PACKAGE_INSTALL_DIRECTORY} Template Numerical Library")
  set(CPACK_NSIS_HELP_LINK "http:\\\\\\\\geraldine.fjfi.cvut.cz/~oberhuber/doku-wiki-tnl")
  set(CPACK_NSIS_URL_INFO_ABOUT "http:\\\\\\\\geraldine.fjfi.cvut.cz/~oberhuber/doku-wiki-tnl")
  set(CPACK_NSIS_CONTACT "tomas.oberhuber@fjfi.cvut.cz")
  set(CPACK_NSIS_MODIFY_PATH ON)
else(WIN32 AND NOT UNIX)
#  set(CPACK_STRIP_FILES "bin/MyExecutable")
  set(CPACK_SOURCE_STRIP_FILES "")
endif(WIN32 AND NOT UNIX)
#set(CPACK_PACKAGE_EXECUTABLES "MyExecutable" "My Executable")
INCLUDE( CPack )
