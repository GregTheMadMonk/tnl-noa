#Libtnl

#Je potreba prenest vsechny Cmakelisty
#PrelozVse
#configDirectory.h.in - podle nej se vytvori hlavickovy soubor s informacemi o umisteni konfiguracnich souboru
#src/dummy.cpp
#src/core/config.h
#src/core/tnlFieldCUDA2D.h
#src/debug/parse.cc - zmena na include tnlDebugParser.h misto
#src/debug/tnlDebugScanner.l	a jeho projeti skrz sflex++

#tests/matrix-formats-test.cpp
#tests/sparse-matrix-benchmark.cpp
#tools/src/mdiff-err-norms.cpp
#tools/src/tnlcurve2gnuplot.cpp
#tools/src/tnl-grid-view.cpp
#tools/src/tnl/matrix-convert.cpp

project(libtnl)

cmake_minimum_required( VERSION 2.8 )

#####
# Check for bzip2
#
FIND_PACKAGE(BZip2)
if( BZIP2_FOUND )
    set( HAVE_LIBBZ2 "" )
    set( CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-DHAVE_CUDA )
else( BZIP2_FOUND )
    set( HAVE_LIBBZ2 "//" )
endif( BZIP2_FOUND )

#####
# Check for CUDA
#
FIND_PACKAGE( CUDA )
if( CUDA_FOUND )
    set( BUILD_CUDA TRUE)
else( CUDA_FOUND )
    
endif( CUDA_FOUND )

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



#-------------------------Skutecny zacatek-----------------------------

ENABLE_TESTING()	#zapneme moznost testovani
# set (BUILD_CUDA)
# set (BUILD_MPI)

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

if (BUILD_MPI)		#Pokud to ma byt za pouziti MPI nastavime si koncovku -mpi, a spravne prekladace
  set(mpiExt -mpi)
  SET(CMAKE_C_COMPILER mpicc)
  SET(CMAKE_CXX_COMPILER mpicxx)
endif()


set (Config_Directory \"${EXECUTABLE_OUTPUT_PATH}/config/\")
configure_file( "configDirectory.h.in"  "${PROJECT_BINARY_DIR}/headers/configDirectory.h" )
configure_file( "tnlConfig.h.in" "${PROJECT_BUILD_PATH}/tnlConfig.h" )

#Nastavime cesty k hlavickovym souborum a knihovnam
INCLUDE_DIRECTORIES( src )
INCLUDE_DIRECTORIES( ${PROJECT_BUILD_PATH} )
INCLUDE_DIRECTORIES( ${PROJECT_BINARY_DIR}/headers )
LINK_DIRECTORIES( ${LIBRARY_OUTPUT_PATH} )

#Pokracujeme dalsimi podadresari
add_subdirectory (src)
add_subdirectory (tests)
add_subdirectory (tools)

