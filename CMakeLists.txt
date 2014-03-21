cmake_minimum_required( VERSION 2.8.10 )

project( tnl )

set( tnlVersion "0.1" )

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

include( OptimizeForArchitecture )
include( UseCodeCoverage )

####
# Settings for debug/release version
#
if( CMAKE_BUILD_TYPE STREQUAL "Debug")
    set( PROJECT_BUILD_PATH ${PROJECT_SOURCE_DIR}/Debug/src )
    set( PROJECT_TESTS_PATH ${PROJECT_SOURCE_DIR}/Debug/tests )
    set( LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/Debug/lib )
    set( EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/Debug/bin )
    set( debugExt -dbg )
    AddCompilerFlag( "-g" )
else()
    set( PROJECT_BUILD_PATH ${PROJECT_SOURCE_DIR}/Release/src )
    set( PROJECT_TESTS_PATH ${PROJECT_SOURCE_DIR}/Release/tests )
    set( LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/Release/lib)
    set( EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/Release/bin)
    OptimizeForArchitecture()
    AddCompilerFlag( "-O3 -DNDEBUG " )
endif()

if( WITH_PROFILING STREQUAL "yes" )
   AddCompilerFlag( "-pg" )
endif()   

if( WITH_TEMPLATE_EXPLICIT_INSTANTIATION STREQUAL "yes" )
   AddCompilerFlag( "-DTEMPLATE_EXPLICIT_INSTANTIATION " )
endif()   

if( WITH_CUDA STREQUAL "yes" )
   AddCompilerFlag( "-DHAVE_NOT_CXX11 -U_GLIBCXX_ATOMIC_BUILTINS -U_GLIBCXX_USE_INT128" )
else()
   AddCompilerFlag( "-std=gnu++0x" )
endif()      

#####
# Check for CUDA
#
if( NOT WITH_CUDA STREQUAL "no" )
    find_package( CUDA )
    if( CUDA_FOUND )
        set( BUILD_CUDA TRUE)
        set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF)
        set(BUILD_SHARED_LIBS ON)
        set(CUDA_SEPARABLE_COMPILATION ON)
        
        set( CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-DHAVE_CUDA)
        if( CUDA_ARCHITECTURE STREQUAL "2.0" )
            set( CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-DCUDA_ARCH=20)            
        endif()    
        if( CUDA_ARCHITECTURE STREQUAL "2.1" )
            set( CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=compute_20;-code=sm_21;-DCUDA_ARCH=21)
        endif()    
        if( CUDA_ARCHITECTURE STREQUAL "2.2" )
            set( CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=sm_22;-DCUDA_ARCH=22)
        endif()    
        if( CUDA_ARCHITECTURE STREQUAL "2.3" )
            set( CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=sm_23;-DCUDA_ARCH=23)
        endif()    
        if( CUDA_ARCHITECTURE STREQUAL "3.0" )
            set( CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=compute_30;-code=sm_30;-DCUDA_ARCH=30)
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
if( MPI_CXX_FOUND )
    set( BUILD_MPI TRUE )
    message( "MPI headers found -- ${MPI_CXX_INCLUDE_PATH}")
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
  message( "CPPUNIT headers found -- ${CPPUNIT_INCLUDE_DIR}" )
  if(CPPUNIT_LIBRARY)
    message( "CPPUNIT library found -- ${CPPUNIT_LIBRARY}" )
    set(CPPUNIT_FOUND "YES")
    set(CPPUNIT_LIBRARIES ${CPPUNIT_LIBRARY} ${CMAKE_DL_LIBS})
    set(CPPUNIT_DEBUG_LIBRARIES ${CPPUNIT_DEBUG_LIBRARY}
                                ${CMAKE_DL_LIBS})
   set( HAVE_CPPUNIT "#define HAVE_CPPUNIT" )
  endif(CPPUNIT_LIBRARY)
endif( ${CPPUNIT_INCLUDE_DIR} STREQUAL "CPPUNIT_INCLUDE_DIR-NOTFOUND" )

#if( BUILD_MPI )
#   FIND_PATH( PETSC_INCLUDE_DIR petsc.h
#     /usr/include/petsc
#     ${PETSC_DIR}/${PETSC_ARCH}/include
#     ${PETSC_DIR}/include
#     DOC "PETSC headers."
#   )
#   if( ${PETSC_INCLUDE_DIR} STREQUAL "PETSC_INCLUDE_DIR-NOTFOUND" )
#      message( "PETSC not found." ) 
#   else()
#      message( "PETSC headers found -- ${PETSC_INCLUDE_DIR}" )
#      FIND_LIBRARY(PETSC_LIBRARY petsc
#                  ${PETSC_INCLUDE_DIR}/../lib
#                  /usr/local/lib
#                  /usr/lib)
#      if( PETSC_LIBRARY )
#         #string( REPLACE ";" " " MPI_LIBRARIES "${MPI_CXX_LIBRARIES}" )
#         #set( PETSC_LIBRARY "${MPI_LIBRARIES} ${PETSC_LIBRARY}")
#         message( "PETSC library found -- ${PETSC_LIBRARY}")
#         list( GET MPI_CXX_INCLUDE_PATH 0 MPI_CXX_PATH )
#         set(PETSC_CXX_FLAGS "-DHAVE_PETSC -I${PETSC_INCLUDE_DIR} -DHAVE_MPI -I${MPI_CXX_PATH}")                     
#      endif()
#   endif()
#endif()


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
add_subdirectory( examples )

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
endif(WIN32 AND NOT UNIX)

set(CPACK_SOURCE_STRIP_FILES "Debug")
set(CPACK_SOURCE_STRIP_FILES "Release")

#set(CPACK_PACKAGE_EXECUTABLES "MyExecutable" "My Executable")
INCLUDE( CPack )
