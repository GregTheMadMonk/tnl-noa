ADD_EXECUTABLE( BoundaryTagsTest BoundaryTagsTest.cpp )
TARGET_COMPILE_OPTIONS( BoundaryTagsTest PRIVATE ${CXX_TESTS_FLAGS} )
TARGET_LINK_LIBRARIES( BoundaryTagsTest
                           ${GTEST_BOTH_LIBRARIES}
                           tnl )

if( BUILD_CUDA AND ${CMAKE_CXX_COMPILER} MATCHES ".*clang\\+\\+" )
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

ADD_EXECUTABLE( MultimapTest MultimapTest.cpp )
TARGET_COMPILE_OPTIONS( MultimapTest PRIVATE ${CXX_TESTS_FLAGS} )
TARGET_LINK_LIBRARIES( MultimapTest
                           ${GTEST_BOTH_LIBRARIES}
                           tnl )

ADD_EXECUTABLE( StaticMultimapTest StaticMultimapTest.cpp )
TARGET_COMPILE_OPTIONS( StaticMultimapTest PRIVATE ${CXX_TESTS_FLAGS} )
TARGET_LINK_LIBRARIES( StaticMultimapTest
                           ${GTEST_BOTH_LIBRARIES}
                           tnl )


ADD_TEST( BoundaryTagsTest ${EXECUTABLE_OUTPUT_PATH}/BoundaryTagsTest${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( MeshTest ${EXECUTABLE_OUTPUT_PATH}/MeshTest${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( MeshOrderingTest ${EXECUTABLE_OUTPUT_PATH}/MeshOrderingTest${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( MeshEntityTest ${EXECUTABLE_OUTPUT_PATH}/MeshEntityTest${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( MultimapTest ${EXECUTABLE_OUTPUT_PATH}/MultimapTest${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( StaticMultimapTest ${EXECUTABLE_OUTPUT_PATH}/MultimapTest${CMAKE_EXECUTABLE_SUFFIX} )



##
## Tests with VTK
##

find_package( VTK )
if( VTK_FOUND )
   include(${VTK_USE_FILE})

   AddCompilerFlag( "-DHAVE_VTK " )
   SET( VTK_COMMON_LIBRARIES vtkCommonCore ; vtkIOLegacy )
endif( VTK_FOUND )

# FIXME: compilation fails with nvcc (CUDA 9.0)
#if( BUILD_CUDA AND ${CMAKE_CXX_COMPILER} MATCHES ".*clang\\+\\+" )
#   CUDA_ADD_EXECUTABLE( MeshReaderTest MeshReaderTest.cu
#                        OPTIONS ${CXX_TESTS_FLAGS} )
#   TARGET_LINK_LIBRARIES( MeshReaderTest
#                           ${GTEST_BOTH_LIBRARIES}
#                           ${VTK_COMMON_LIBRARIES}
#                           tnl )
#else()
   ADD_EXECUTABLE( MeshReaderTest MeshReaderTest.cpp )
   TARGET_COMPILE_OPTIONS( MeshReaderTest PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( MeshReaderTest
                           ${GTEST_BOTH_LIBRARIES}
                           ${VTK_COMMON_LIBRARIES}
                           tnl )
#endif()
