ADD_SUBDIRECTORY( DistributedMeshes )

ADD_EXECUTABLE( BoundaryTagsTest BoundaryTagsTest.cpp )
TARGET_COMPILE_OPTIONS( BoundaryTagsTest PRIVATE ${CXX_TESTS_FLAGS} )
TARGET_LINK_LIBRARIES( BoundaryTagsTest
                           ${GTEST_BOTH_LIBRARIES}
                           tnl )

# Mesh cannot be compiled by nvcc < 9 due to bugs in the compiler
if( ${BUILD_CUDA} AND ${CUDA_VERSION_MAJOR} GREATER_EQUAL 9 )
   CUDA_ADD_EXECUTABLE( MeshTest MeshTest.cu
                        OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( MeshTest
                           ${GTEST_BOTH_LIBRARIES}
                           tnl )

   CUDA_ADD_EXECUTABLE( MeshOrderingTest MeshOrderingTest.cu
                        OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( MeshOrderingTest
                           ${GTEST_BOTH_LIBRARIES}
                           tnl )
else()
   ADD_EXECUTABLE( MeshTest MeshTest.cpp )
   TARGET_COMPILE_OPTIONS( MeshTest PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( MeshTest
                           ${GTEST_BOTH_LIBRARIES}
                           tnl )

   ADD_EXECUTABLE( MeshOrderingTest MeshOrderingTest.cpp )
   TARGET_COMPILE_OPTIONS( MeshOrderingTest PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( MeshOrderingTest
                           ${GTEST_BOTH_LIBRARIES}
                           tnl )
endif()

ADD_EXECUTABLE( MeshEntityTest MeshEntityTest.cpp )
TARGET_COMPILE_OPTIONS( MeshEntityTest PRIVATE ${CXX_TESTS_FLAGS} )
TARGET_LINK_LIBRARIES( MeshEntityTest
                           ${GTEST_BOTH_LIBRARIES}
                           tnl )


ADD_TEST( BoundaryTagsTest ${EXECUTABLE_OUTPUT_PATH}/BoundaryTagsTest${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( MeshTest ${EXECUTABLE_OUTPUT_PATH}/MeshTest${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( MeshOrderingTest ${EXECUTABLE_OUTPUT_PATH}/MeshOrderingTest${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( MeshEntityTest ${EXECUTABLE_OUTPUT_PATH}/MeshEntityTest${CMAKE_EXECUTABLE_SUFFIX} )



##
## Tests with VTK
##

#find_package( VTK )
#if( VTK_FOUND )
#   include(${VTK_USE_FILE})
#
#   AddCompilerFlag( "-DHAVE_VTK " )
#   SET( VTK_COMMON_LIBRARIES vtkCommonCore ; vtkIOLegacy )
#endif( VTK_FOUND )

# Mesh cannot be compiled by nvcc < 9 due to bugs in the compiler
if( ${BUILD_CUDA} AND ${CUDA_VERSION_MAJOR} GREATER_EQUAL 9 )
   CUDA_ADD_EXECUTABLE( MeshReaderTest MeshReaderTest.cu
                        OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( MeshReaderTest
                           ${GTEST_BOTH_LIBRARIES}
                           ${VTK_COMMON_LIBRARIES}
                           tnl )
else()
   ADD_EXECUTABLE( MeshReaderTest MeshReaderTest.cpp )
   TARGET_COMPILE_OPTIONS( MeshReaderTest PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( MeshReaderTest
                           ${GTEST_BOTH_LIBRARIES}
                           ${VTK_COMMON_LIBRARIES}
                           tnl )
endif()
