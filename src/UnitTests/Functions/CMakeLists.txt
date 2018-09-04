IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( BoundaryMeshFunctionTest BoundaryMeshFunctionTest.h BoundaryMeshFunctionTest.cu OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( BoundaryMeshFunctionTest ${GTEST_BOTH_LIBRARIES}
                                                           tnl )
ELSE(  BUILD_CUDA )
   ADD_EXECUTABLE( BoundaryMeshFunctionTest BoundaryMeshFunctionTest.h BoundaryMeshFunctionTest.cpp )
   TARGET_COMPILE_OPTIONS( BoundaryMeshFunctionTest PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( BoundaryMeshFunctionTest ${GTEST_BOTH_LIBRARIES}
                                                           tnl )
ENDIF( BUILD_CUDA )


ADD_TEST( BoundaryMeshFunctionTest ${EXECUTABLE_OUTPUT_PATH}/BoundaryMeshFunctionTest${CMAKE_EXECUTABLE_SUFFIX} )