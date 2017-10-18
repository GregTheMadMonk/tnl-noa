ADD_EXECUTABLE( BoundaryTagsTest${mpiExt}${debugExt} BoundaryTagsTest.cpp )
TARGET_COMPILE_OPTIONS( BoundaryTagsTest${mpiExt}${debugExt} PRIVATE ${CXX_TESTS_FLAGS} )
TARGET_LINK_LIBRARIES( BoundaryTagsTest${mpiExt}${debugExt}
                           ${GTEST_BOTH_LIBRARIES}
                           tnl${mpiExt}${debugExt}-${tnlVersion} )

if( BUILD_CUDA AND ${CMAKE_CXX_COMPILER} MATCHES ".*clang\\+\\+" )
   CUDA_ADD_EXECUTABLE( MeshTest${mpiExt}${debugExt} MeshTest.cu
                        OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( MeshTest${mpiExt}${debugExt}
                           ${GTEST_BOTH_LIBRARIES}
                           tnl${mpiExt}${debugExt}-${tnlVersion} )

   CUDA_ADD_EXECUTABLE( MeshOrderingTest${mpiExt}${debugExt} MeshOrderingTest.cu
                        OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( MeshOrderingTest${mpiExt}${debugExt}
                           ${GTEST_BOTH_LIBRARIES}
                           tnl${mpiExt}${debugExt}-${tnlVersion} )
else()
   ADD_EXECUTABLE( MeshTest${mpiExt}${debugExt} MeshTest.cpp )
   TARGET_COMPILE_OPTIONS( MeshTest${mpiExt}${debugExt} PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( MeshTest${mpiExt}${debugExt}
                           ${GTEST_BOTH_LIBRARIES}
                           tnl${mpiExt}${debugExt}-${tnlVersion} )

   ADD_EXECUTABLE( MeshOrderingTest${mpiExt}${debugExt} MeshOrderingTest.cpp )
   TARGET_COMPILE_OPTIONS( MeshOrderingTest${mpiExt}${debugExt} PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( MeshOrderingTest${mpiExt}${debugExt}
                           ${GTEST_BOTH_LIBRARIES}
                           tnl${mpiExt}${debugExt}-${tnlVersion} )
endif()

ADD_EXECUTABLE( MeshEntityTest${mpiExt}${debugExt} MeshEntityTest.cpp )
TARGET_COMPILE_OPTIONS( MeshEntityTest${mpiExt}${debugExt} PRIVATE ${CXX_TESTS_FLAGS} )
TARGET_LINK_LIBRARIES( MeshEntityTest${mpiExt}${debugExt}
                           ${GTEST_BOTH_LIBRARIES}
                           tnl${mpiExt}${debugExt}-${tnlVersion} )

ADD_EXECUTABLE( MultimapTest${mpiExt}${debugExt} MultimapTest.cpp )
TARGET_COMPILE_OPTIONS( MultimapTest${mpiExt}${debugExt} PRIVATE ${CXX_TESTS_FLAGS} )
TARGET_LINK_LIBRARIES( MultimapTest${mpiExt}${debugExt}
                           ${GTEST_BOTH_LIBRARIES}
                           tnl${mpiExt}${debugExt}-${tnlVersion} )

ADD_EXECUTABLE( StaticMultimapTest${mpiExt}${debugExt} StaticMultimapTest.cpp )
TARGET_COMPILE_OPTIONS( StaticMultimapTest${mpiExt}${debugExt} PRIVATE ${CXX_TESTS_FLAGS} )
TARGET_LINK_LIBRARIES( StaticMultimapTest${mpiExt}${debugExt}
                           ${GTEST_BOTH_LIBRARIES}
                           tnl${mpiExt}${debugExt}-${tnlVersion} )


ADD_TEST( BoundaryTagsTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/BoundaryTagsTest${mpiExt}${debugExt} )
ADD_TEST( MeshTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/MeshTest${mpiExt}${debugExt} )
ADD_TEST( MeshOrderingTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/MeshOrderingTest${mpiExt}${debugExt} )
ADD_TEST( MeshEntityTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/MeshEntityTest${mpiExt}${debugExt} )
ADD_TEST( MultimapTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/MultimapTest${mpiExt}${debugExt} )
ADD_TEST( StaticMultimapTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/MultimapTest${mpiExt}${debugExt} )



##
## Tests with VTK
##

find_package( VTK )
if( VTK_FOUND )
   include(${VTK_USE_FILE})

   AddCompilerFlag( "-DHAVE_VTK " )
   SET( VTK_COMMON_LIBRARIES vtkCommonCore ; vtkIOLegacy )
endif( VTK_FOUND )

if( BUILD_CUDA AND ${CMAKE_CXX_COMPILER} MATCHES ".*clang\\+\\+" )
   CUDA_ADD_EXECUTABLE( MeshReaderTest${mpiExt}${debugExt} MeshReaderTest.cu
                        OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( MeshReaderTest${mpiExt}${debugExt}
                           ${GTEST_BOTH_LIBRARIES}
                           ${VTK_COMMON_LIBRARIES}
                           tnl${mpiExt}${debugExt}-${tnlVersion} )
else()
   ADD_EXECUTABLE( MeshReaderTest${mpiExt}${debugExt} MeshReaderTest.cpp )
   TARGET_COMPILE_OPTIONS( MeshReaderTest${mpiExt}${debugExt} PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( MeshReaderTest${mpiExt}${debugExt}
                           ${GTEST_BOTH_LIBRARIES}
                           ${VTK_COMMON_LIBRARIES}
                           tnl${debugExt}-${tnlVersion} )
endif()
