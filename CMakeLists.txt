###############################################################################
#                      Cmake project script for TNL
#                             -------------------
#    begin               : Dec 8, 2010
#    copyright           : (C) 2010 by Tomas Oberhuber et al.
#    email               : tomas.oberhuber@fjfi.cvut.cz
#
###############################################################################
#
# Authors:
# Tomas Oberhuber
# Vladimir Klement
# Jakub Klinkovsky

cmake_minimum_required( VERSION 3.5.1 )

project( tnl )

set( tnlVersion "0.1" )

# declare all custom build options
option(OFFLINE_BUILD "Offline build (i.e. without downloading libraries such as pybind11)" OFF)
option(WITH_MIC "Build with MIC support" OFF)
option(WITH_CUDA "Build with CUDA support" ON)
set(WITH_CUDA_ARCH "auto" CACHE STRING "Build for these CUDA architectures")
option(WITH_OPENMP "Build with OpenMP support" ON)
option(WITH_GMP "Build with GMP support" OFF)
option(WITH_TESTS "Build tests" ON)
option(WITH_COVERAGE "Enable code coverage reports from unit tests" OFF)
option(WITH_EXAMPLES "Compile the 'examples' directory" ON)
option(WITH_TOOLS "Compile the 'src/Tools' directory" ON)
option(WITH_PYTHON "Compile the Python bindings" ON)
option(WITH_TEMPLATES_INSTANTIATION "Enable explicit template instantiation" OFF)

# install paths relative to the cmake's prefix
set( TNL_TARGET_INCLUDE_DIRECTORY "include/TNL" )
set( TNL_TARGET_DATA_DIRECTORY "share/TNL" )

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

include( OptimizeForArchitecture )

# Note that in cmake 3.10 the FindOpenMP module is broken - it does not work when
# CMAKE_EXECUTABLE_SUFFIX is not empty, see https://www.mail-archive.com/cmake@cmake.org/msg56886.html
# Hence, we find OpenMP before setting CMAKE_EXECUTABLE_SUFFIX.
find_package( OpenMP )

####
# Settings for debug/release version
#
if( CMAKE_BUILD_TYPE STREQUAL "Debug")
    set( PROJECT_BUILD_PATH ${PROJECT_SOURCE_DIR}/Debug/src )
    set( PROJECT_TOOLS_PATH ${PROJECT_SOURCE_DIR}/Debug/bin )
    set( LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/Debug/lib )
    set( EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/Debug/bin )
    set( CMAKE_EXECUTABLE_SUFFIX "-dbg${CMAKE_EXECUTABLE_SUFFIX}" )  # suffix for executables
    set( CMAKE_DEBUG_POSTFIX "-dbg" )  # suffix for libraries
else()
    set( PROJECT_BUILD_PATH ${PROJECT_SOURCE_DIR}/Release/src )
    set( PROJECT_TOOLS_PATH ${PROJECT_SOURCE_DIR}/Release/bin )
    set( LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/Release/lib )
    set( EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/Release/bin )
endif()

# check if the compiler is good enough
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
   # GCC 5.0 is the first release with full C++11 support (due to libstdc++)
   # https://gcc.gnu.org/gcc-5/changes.html
   if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "5.0")
      message(FATAL_ERROR "Insufficient GCC version")
   endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
   # Clang 3.4 has full C++14 support: http://clang.llvm.org/cxx_status.html
   if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "3.4")
      message(FATAL_ERROR "Insufficient Clang version")
   endif()
endif()

# set Debug/Release options
set( CMAKE_CXX_FLAGS "-std=c++11 -pthread -Wall -Wno-unused-local-typedefs -Wno-unused-variable" )
set( CMAKE_CXX_FLAGS_DEBUG "-g" )
set( CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native -DNDEBUG" )
#set( CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native -DNDEBUG -ftree-vectorizer-verbose=1 -ftree-vectorize -fopt-info-vec-missed -funroll-loops" )
# pass -rdynamic only in Debug mode
set( CMAKE_SHARED_LIBRARY_LINK_C_FLAGS "" )
set( CMAKE_SHARED_LIBRARY_LINK_C_FLAGS_DEBUG "-rdynamic" )
set( CMAKE_SHARED_LIBRARY_LINK_C_FLAGS_RELEASE "" )
set( CMAKE_EXE_LINKER_FLAGS "" )
set( CMAKE_EXE_LINKER_FLAGS_DEBUG "-rdynamic" )
set( CMAKE_EXE_LINKER_FLAGS_RELEASE "" )
set( CMAKE_SHARED_LINKER_FLAGS "" )
set( CMAKE_SHARED_LINKER_FLAGS_DEBUG "-rdynamic" )
set( CMAKE_SHARED_LINKER_FLAGS_RELEASE "" )

get_filename_component( CXX_COMPILER_NAME ${CMAKE_CXX_COMPILER} NAME )
if( CXX_COMPILER_NAME MATCHES "icpc" )
   message( "Intel compiler detected..."    )
   set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_ICPC -wd2568 -wd2571 -wd2570")
   #####
   #  Check for MIC
   #
   if( ${WITH_MIC} )
      message( "Enabled MIC support." )
      set( MIC_CXX_FLAGS "-DHAVE_MIC")
      # build all tests with MIC support
      set( CXX_TESTS_FLAGS ${CXX_TESTS_FLAGS} -DHAVE_MIC )
      set( WITH_CUDA OFF CACHE BOOL "Build with CUDA support" )
   else()
      set( MIC_CXX_FLAGS "")
   endif()
endif()

# force colorized output in continuous integration
if( DEFINED ENV{CI_JOB_NAME} OR ${CMAKE_GENERATOR} STREQUAL "Ninja" )
   message(STATUS "Continuous integration or Ninja detected -- forcing compilers to produce colorized output.")
   if( CXX_COMPILER_NAME MATCHES "clang" )
      set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fcolor-diagnostics" )
   else()
      set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdiagnostics-color" )
   endif()
endif()

#####
# Check for MPI -- poznej podle vraperu compileru -- da se testovat preklad bez MPI
#
if( ${CXX_COMPILER_NAME} STREQUAL "mpic++" )
   message( "MPI compiler detected."    )
   set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_MPI" )
   set( CUDA_HOST_COMPILER "mpic++" )
endif()

####
# Check for MPI -- not working
#
#find_package( MPI )
#if( MPI_CXX_FOUND )
   # set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_MPI" )
   # message( "MPI headers found -- ${MPI_CXX_INCLUDE_PATH}")
   # message( "MPI link flags  -- ${MPI_CXX_LINK_FLAGS}")
   # message( "MPI libraries-- ${MPI_CXX_LIBRARIES}")
#endif()

#####
# Check for CUDA
#
if( ${WITH_CUDA} )
    find_package( CUDA )
    if( CUDA_FOUND )
        set( BUILD_CUDA TRUE)
        set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF)
        set(BUILD_SHARED_LIBS ON)
        set(CUDA_SEPARABLE_COMPILATION ON)
        # Use the CUDA_HOST_COMPILER environment variable if the user specified it.
        if( NOT $ENV{CUDA_HOST_COMPILER} STREQUAL "" )
            message( "-- Setting CUDA_HOST_COMPILER to '$ENV{CUDA_HOST_COMPILER}'" )
            set( CUDA_HOST_COMPILER $ENV{CUDA_HOST_COMPILER} )
        else()
            if( EXISTS "${CUDA_TOOLKIT_ROOT_DIR}/bin/g++" )
               message( "-- Setting CUDA_HOST_COMPILER to '${CUDA_TOOLKIT_ROOT_DIR}/bin/g++'" )
               set( CUDA_HOST_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/g++" )
            else()
               message( "-- Setting CUDA_HOST_COMPILER to '${CMAKE_CXX_COMPILER}'" )
               set( CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER} )
            endif()
        endif()
        # An extra CUDA_ARCH_HOST_COMPILER variable for compiling tnl-cuda-arch alone,
        # because it SHOULD NOT be compiled using mpic++, which would cause weird
        # RPATH_CHANGE error in cmake.
        # FIXME: find better solution to switch between MPI-enabled and MPI-disabled binaries in cmake
        if( NOT $ENV{CUDA_ARCH_HOST_COMPILER} STREQUAL "" )
           message( "-- Setting CUDA_ARCH_HOST_COMPILER to '$ENV{CUDA_ARCH_HOST_COMPILER}'" )
           set( CUDA_ARCH_HOST_COMPILER $ENV{CUDA_ARCH_HOST_COMPILER} )
        else()
            if( EXISTS "${CUDA_TOOLKIT_ROOT_DIR}/bin/g++" )
               message( "-- Setting CUDA_ARCH_HOST_COMPILER to '${CUDA_TOOLKIT_ROOT_DIR}/bin/g++'" )
               set( CUDA_ARCH_HOST_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/g++" )
            else()
               message( "-- Setting CUDA_ARCH_HOST_COMPILER to '${CMAKE_CXX_COMPILER}'" )
               set( CUDA_ARCH_HOST_COMPILER ${CMAKE_CXX_COMPILER} )
            endif()
        endif()
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ;-DHAVE_CUDA)
        # disable false compiler warnings
        #   reference for the -Xcudafe flag: http://stackoverflow.com/questions/14831051/how-to-disable-compiler-warnings-with-nvcc/17095910#17095910
        #   list of possible tokens: http://www.ssl.berkeley.edu/~jimm/grizzly_docs/SSL/opt/intel/cc/9.0/lib/locale/en_US/mcpcom.msg
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ; -Wno-deprecated-gpu-targets --expt-relaxed-constexpr --expt-extended-lambda -Xcudafe "\"--diag_suppress=code_is_unreachable --diag_suppress=implicit_return_from_non_void_function\"")
        set( ALL_CUDA_ARCHS -gencode arch=compute_20,code=sm_20
                            -gencode arch=compute_30,code=sm_30
                            -gencode arch=compute_32,code=sm_32 
                            -gencode arch=compute_35,code=sm_35 
                            -gencode arch=compute_37,code=sm_37 
                            -gencode arch=compute_50,code=sm_50 
                            -gencode arch=compute_52,code=sm_52 )
        if( WITH_CUDA_ARCH STREQUAL "all" )
           set( CUDA_ARCH ${ALL_CUDA_ARCHS} )   
        else()
            if( WITH_CUDA_ARCH STREQUAL "auto")
                ####
                # Select GPU architecture
                #
                set( CUDA_ARCH_EXECUTABLE ${EXECUTABLE_OUTPUT_PATH}/tnl-cuda-arch)
                set( CUDA_ARCH_SOURCE ${PROJECT_SOURCE_DIR}/src/Tools/tnl-cuda-arch.cu)
                message( "Compiling tnl-cuda-arch ..." )
                file( MAKE_DIRECTORY ${EXECUTABLE_OUTPUT_PATH} )
                execute_process( COMMAND nvcc --compiler-bindir ${CUDA_ARCH_HOST_COMPILER} --std=c++11 ${CUDA_ARCH_SOURCE} -o ${CUDA_ARCH_EXECUTABLE}
                                 RESULT_VARIABLE CUDA_ARCH_RESULT
                                 OUTPUT_VARIABLE CUDA_ARCH_OUTPUT
                                 ERROR_VARIABLE CUDA_ARCH_OUTPUT )
                execute_process( COMMAND ${CUDA_ARCH_EXECUTABLE}
                                 OUTPUT_VARIABLE CUDA_ARCH )
                if( NOT CUDA_ARCH_RESULT )
                    # strip linebreaks and convert to list delimited with ';'
                    string( REGEX REPLACE "[\n ]" ";" CUDA_ARCH ${CUDA_ARCH} )
                    # cache the result
                    set( CUDA_ARCH ${CUDA_ARCH} CACHE LIST "GPU architecture options" )
                else()
                    message( "Failed to detect GPU architecture:\n${CUDA_ARCH_OUTPUT}" )
                    message( "Using (almost) all GPU architectures as fallback." )
                    set( CUDA_ARCH ${ALL_CUDA_ARCHS} )
                endif()
                message( "GPU architecture options:  ${CUDA_ARCH}" )
            else()
                if( NOT WITH_CUDA_ARCH STREQUAL "" )
                    set( CUDA_ARCH -gencode arch=compute_${WITH_CUDA_ARCH},code=sm_${WITH_CUDA_ARCH} )
                else()
                    message( FATAL_ERROR "\$WITH_CUDA_ARCH cannot be empty." )
                endif()
            endif()
        endif()
        set( CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ; ${CUDA_ARCH} -D_FORCE_INLINES )
        # TODO: this is necessary only due to a bug in cmake
        set( CUDA_ADD_LIBRARY_OPTIONS -shared )
    endif()
endif()


####
# Check for OpenMP
#
if( OPENMP_FOUND AND ${WITH_OPENMP} )
   message( "Enabled OpenMP support." )
   set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_OPENMP ${OpenMP_CXX_FLAGS}" )
endif()

find_package( DCMTK )
if( DCMTK_FOUND )
   set( HAVE_DCMTK_H "#define HAVE_DCMTK_H 1" )
   include_directories( ${DCMTK_INCLUDE_DIRS} )
   set( HAVE_DCMTK TRUE)
   set( DCMTK_LD_FLAGS "" )
   foreach( arg ${DCMTK_LIBRARIES} )
      set( DCMTK_LD_FLAGS "${DCMTK_LD_FLAGS} ${arg}" )
   endforeach( arg ${DCMTK_LIBRARIES} )
else()
   set( HAVE_DCMTK_H "//#define HAVE_DCMTK_H 1" )
endif()

find_package( PNG )
if( PNG_FOUND )      
   set( HAVE_PNG_H "#define HAVE_PNG_H 1" )
   include_directories( ${PNG_INCLUDE_DIRS} )
else()
   set( HAVE_PNG_H "//#define HAVE_PNG_H 1" )
   set( PNG_LIBRARIES "" )
endif()

find_package( JPEG )
if( JPEG_FOUND )      
   set( HAVE_JPEG_H "#define HAVE_JPEG_H 1" )
   include_directories( ${JPEG_INCLUDE_DIRS} )
else()
   set( HAVE_JPEG_H "//#define HAVE_JPEG_H 1" )
   set( JPEG_LIBRARIES "" )
endif()

####
# Test for GMP 
#
if( ${WITH_GMP} )
   if (GMP_INCLUDES AND GMP_LIBRARIES)
      set(GMP_FIND_QUIETLY TRUE)
   endif (GMP_INCLUDES AND GMP_LIBRARIES)

   find_path(GMP_INCLUDES
      NAMES
      gmp.h
      PATHS
      $ENV{GMPDIR}
      ${INCLUDE_INSTALL_DIR}
   )

   find_library(GMP_LIBRARIES gmp PATHS $ENV{GMPDIR} ${LIB_INSTALL_DIR})

   include(FindPackageHandleStandardArgs)
   find_package_handle_standard_args(GMP DEFAULT_MSG
                                      GMP_INCLUDES GMP_LIBRARIES)
   if( ${GMP_INCLUDES} STREQUAL "GMP_INCLUDES-NOTFOUND" OR ${GMP_LIBRARIES} STREQUAL "GMP_LIBRARIES-NOTFOUND" )
      message( "GMP was not found. Some tests for higher precision arithmetics will not be passed." )
   else()
      set( HAVE_GMP )
      set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${GMP_INCLUDES} -DHAVE_GMP" )
      set( CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${GMP_LIBRARIES}" )
      mark_as_advanced(GMP_INCLUDES GMP_LIBRARIES)
   endif()
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
    set( HAVE_SYS_TIME_H "//#define HAVE_SYS_TIME_H 1" )
else()
    #include_directories( ${SYS_TIME_INCLUDE_DIR} )
    set( HAVE_SYS_TIME_H "#define HAVE_SYS_TIME_H 1" )
endif()

find_path( SYS_RESOURCE_INCLUDE_DIR sys/resource.h
           /usr/include/x86_64-linux-gnu
           /usr/include  
           DOC "System resources headers." )
if( ${SYS_RESOURCE_INCLUDE_DIR} STREQUAL "SYS_RESOURCE_INCLUDE_DIR-NOTFOUND" )
    message( "Missing header file sys/time.h" )
    set( HAVE_SYS_RESOURCE_H "//#define HAVE_SYS_RESOURCE_H 1" )
else()
    #include_directories( ${SYS_RESOURCE_INCLUDE_DIR} )
    set( HAVE_SYS_RESOURCE_H "#define HAVE_SYS_RESOURCE_H 1" )
endif()

find_path( SYS_IOCTL_INCLUDE_DIR sys/ioctl.h
           /usr/include/x86_64-linux-gnu
           /usr/include  
           DOC "System ioctl headers." )
if( ${SYS_IOCTL_INCLUDE_DIR} STREQUAL "SYS_IOCTL_INCLUDE_DIR-NOTFOUND" )
    message( "Missing header file sys/time.h" )
    set( HAVE_SYS_IOCTL_H "//#define HAVE_SYS_IOCTL_H 1" )
else()
    #include_directories( ${SYS_IOCTL_INCLUDE_DIR} )
    set( HAVE_SYS_IOCTL_H "#define HAVE_SYS_IOCTL_H 1" )
endif()

if( ${WITH_TESTS} )
   enable_testing()

   # build gtest libs
   include( BuildGtest )

   if( ${WITH_COVERAGE} AND CMAKE_BUILD_TYPE STREQUAL "Debug" )
      # enable code coverage reports
      include( UseCodeCoverage )
   endif()
endif()

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

####
# Explicit template instantiation
#
#if( ${WITH_TEMPLATE_INSTANTIATION} )
#   AddCompilerFlag( "-DTEMPLATE_EXPLICIT_INSTANTIATION " )
#
#   if( INSTANTIATE_INT STREQUAL "yes" )
#      AddCompilerFlag( "-DINSTANTIATE_INT " )
#   endif()
#
#   if( INSTANTIATE_LONG_INT STREQUAL "yes" )
#      AddCompilerFlag( "-DINSTANTIATE_LONG_INT " )
#   endif()
#
#   if( INSTANTIATE_FLOAT STREQUAL "yes" )
#      AddCompilerFlag( "-DINSTANTIATE_FLOAT " )
#   endif()
#
#   if( INSTANTIATE_DOUBLE STREQUAL "yes" )
#      AddCompilerFlag( "-DINSTANTIATE_DOUBLE " )
#   endif()
#
#   if( INSTANTIATE_LONG_DOUBLE STREQUAL "yes" )
#      AddCompilerFlag( "-DINSTANTIATE_LONG_DOUBLE " )
#   endif()
#endif()

if( OPTIMIZED_VECTOR_HOST_OPERATIONS STREQUAL "yes" )
   AddCompilerFlag( "-DOPTIMIZED_VECTOR_HOST_OPERATIONS " )
endif()

CONFIGURE_FILE( "tnlConfig.h.in" "${PROJECT_BUILD_PATH}/TNL/tnlConfig.h" )
INSTALL( FILES ${PROJECT_BUILD_PATH}/TNL/tnlConfig.h DESTINATION ${TNL_TARGET_INCLUDE_DIRECTORY} )

#Nastavime cesty k hlavickovym souborum a knihovnam
INCLUDE_DIRECTORIES( src )
INCLUDE_DIRECTORIES( ${PROJECT_BUILD_PATH} )
LINK_DIRECTORIES( ${LIBRARY_OUTPUT_PATH} )

# Add all subdirectories
# Note that it is important to start building examples as soon as possible,
# because they take the longest time and other stuff can be pipelined before
# they are finished (at least with Ninja).
if( WITH_EXAMPLES STREQUAL "yes" )
   add_subdirectory( examples )
endif( WITH_EXAMPLES STREQUAL "yes" )
if( WITH_TESTS STREQUAL "yes" )
    add_subdirectory( tests )
endif( WITH_TESTS STREQUAL "yes" )
add_subdirectory( src )
add_subdirectory( share )

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

# Print custom build options
message( "-- Build options:" )
message( "   OFFLINE_BUILD=${OFFLINE_BUILD}" )
message( "   WITH_MIC=${WITH_MIC}" )
message( "   WITH_CUDA=${WITH_CUDA}" )
message( "   WITH_CUDA_ARCH=${WITH_CUDA_ARCH}" )
message( "   WITH_OPENMP=${WITH_OPENMP}" )
message( "   WITH_GMP=${WITH_GMP}" )
message( "   WITH_TESTS=${WITH_TESTS}" )
message( "   WITH_COVERAGE=${WITH_COVERAGE}" )
message( "   WITH_EXAMPLES=${WITH_EXAMPLES}" )
message( "   WITH_TOOLS=${WITH_TOOLS}" )
message( "   WITH_PYTHON=${WITH_PYTHON}" )
message( "   WITH_TEMPLATES_INSTANTIATION=${WITH_TEMPLATES_INSTANTIATION}" )
# Print compiler options
message( "-- Compiler options:" )
message( "   CMAKE_CXX_FLAGS = ${CMAKE_CXX_FLAGS}" )
message( "   CMAKE_CXX_FLAGS_DEBUG = ${CMAKE_CXX_FLAGS_DEBUG}" )
message( "   CMAKE_CXX_FLAGS_RELEASE = ${CMAKE_CXX_FLAGS_RELEASE}" )
message( "   CMAKE_SHARED_LIBRARY_LINK_C_FLAGS = ${CMAKE_SHARED_LIBRARY_LINK_C_FLAGS}" )
message( "   CMAKE_SHARED_LIBRARY_LINK_C_FLAGS_DEBUG = ${CMAKE_SHARED_LIBRARY_LINK_C_FLAGS_DEBUG}" )
message( "   CMAKE_SHARED_LIBRARY_LINK_C_FLAGS_RELEASE = ${CMAKE_SHARED_LIBRARY_LINK_C_FLAGS_RELEASE}" )
message( "   CMAKE_EXE_LINKER_FLAGS = ${CMAKE_EXE_LINKER_FLAGS}" )
message( "   CMAKE_EXE_LINKER_FLAGS_DEBUG = ${CMAKE_EXE_LINKER_FLAGS_DEBUG}" )
message( "   CMAKE_EXE_LINKER_FLAGS_RELEASE = ${CMAKE_EXE_LINKER_FLAGS_RELEASE}" )
message( "   CMAKE_SHARED_LINKER_FLAGS = ${CMAKE_SHARED_LINKER_FLAGS}" )
message( "   CMAKE_SHARED_LINKER_FLAGS_DEBUG = ${CMAKE_SHARED_LINKER_FLAGS_DEBUG}" )
message( "   CMAKE_SHARED_LINKER_FLAGS_RELEASE = ${CMAKE_SHARED_LINKER_FLAGS_RELEASE}" )
message( "   CUDA_NVCC_FLAGS = ${CUDA_NVCC_FLAGS}" )
message( "   GMP_LIBRARIES = ${GMP_LIBRARIES}" )
