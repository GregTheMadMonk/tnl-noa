IF( BUILD_CUDA )
#   CUDA_ADD_EXECUTABLE( SegmentsTest_AdEllpack SegmentsTest_AdEllpack.cu OPTIONS ${CXX_TESTS_FLAGS} )
#   TARGET_LINK_LIBRARIES( SegmentsTest_AdEllpack ${GTEST_BOTH_LIBRARIES} )

#   CUDA_ADD_EXECUTABLE( SegmentsTest_BiEllpack SegmentsTest_BiEllpack.cu OPTIONS ${CXX_TESTS_FLAGS} )
#   TARGET_LINK_LIBRARIES( SegmentsTest_BiEllpack ${GTEST_BOTH_LIBRARIES} )

#   CUDA_ADD_EXECUTABLE( SegmentsTest_ChunkedEllpack SegmentsTest_ChunkedEllpack.cu OPTIONS ${CXX_TESTS_FLAGS} )
#   TARGET_LINK_LIBRARIES( SegmentsTest_ChunkedEllpack ${GTEST_BOTH_LIBRARIES} )

   CUDA_ADD_EXECUTABLE( SegmentsTest_CSR SegmentsTest_CSR.cu OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( SegmentsTest_CSR ${GTEST_BOTH_LIBRARIES} )

   CUDA_ADD_EXECUTABLE( SegmentsTest_Ellpack SegmentsTest_Ellpack.cu OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( SegmentsTest_Ellpack ${GTEST_BOTH_LIBRARIES} )

#   CUDA_ADD_EXECUTABLE( SegmentsTest_SlicedEllpack SegmentsTest_SlicedEllpack.cu OPTIONS ${CXX_TESTS_FLAGS} )
#   TARGET_LINK_LIBRARIES( SegmentsTest_SlicedEllpack ${GTEST_BOTH_LIBRARIES} )

ELSE(  BUILD_CUDA )
#   ADD_EXECUTABLE( SegmentsTest_AdEllpack SegmentsTest_AdEllpack.cpp )
#   TARGET_COMPILE_OPTIONS( SegmentsTest_AdEllpack PRIVATE ${CXX_TESTS_FLAGS} )
#   TARGET_LINK_LIBRARIES( SegmentsTest_AdEllpack ${GTEST_BOTH_LIBRARIES} )

#   ADD_EXECUTABLE( SegmentsTest_BiEllpack SegmentsTest_BiEllpack.cpp )
#   TARGET_COMPILE_OPTIONS( SegmentsTest_BiEllpack PRIVATE ${CXX_TESTS_FLAGS} )
#   TARGET_LINK_LIBRARIES( SegmentsTest_BiEllpack ${GTEST_BOTH_LIBRARIES} )

#   ADD_EXECUTABLE( SegmentsTest_ChunkedEllpack SegmentsTest_ChunkedEllpack.cpp )
#   TARGET_COMPILE_OPTIONS( SegmentsTest_ChunkedEllpack PRIVATE ${CXX_TESTS_FLAGS} )
#   TARGET_LINK_LIBRARIES( SegmentsTest_ChunkedEllpack ${GTEST_BOTH_LIBRARIES} )

   ADD_EXECUTABLE( SegmentsTest_CSR SegmentsTest_CSR.cpp )
   TARGET_COMPILE_OPTIONS( SegmentsTest_CSR PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( SegmentsTest_CSR ${GTEST_BOTH_LIBRARIES} )

   ADD_EXECUTABLE( SegmentsTest_Ellpack SegmentsTest_Ellpack.cpp )
   TARGET_COMPILE_OPTIONS( SegmentsTest_Ellpack PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( SegmentsTest_Ellpack ${GTEST_BOTH_LIBRARIES} )

#   ADD_EXECUTABLE( SegmentsTest_SlicedEllpack SegmentsTest_SlicedEllpack.cpp )
#   TARGET_COMPILE_OPTIONS( SegmentsTest_SlicedEllpack PRIVATE ${CXX_TESTS_FLAGS} )
#   TARGET_LINK_LIBRARIES( SegmentsTest_SlicedEllpack ${GTEST_BOTH_LIBRARIES} )
ENDIF( BUILD_CUDA )


#ADD_TEST( SegmentsTest_BiEllpack ${EXECUTABLE_OUTPUT_PATH}/SegmentsTest_BiEllpack${CMAKE_EXECUTABLE_SUFFIX} )
#ADD_TEST( SegmentsTest_ChunkedEllpack ${EXECUTABLE_OUTPUT_PATH}/SegmentsTest_ChunkedEllpack${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( SegmentsTest_CSR ${EXECUTABLE_OUTPUT_PATH}/SegmentsTest_CSR${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( SegmentsTest_Ellpack ${EXECUTABLE_OUTPUT_PATH}/SegmentsTest_Ellpack${CMAKE_EXECUTABLE_SUFFIX} )
#ADD_TEST( SegmentsTest_SlicedEllpack ${EXECUTABLE_OUTPUT_PATH}/SegmentsTest_SlicedEllpack${CMAKE_EXECUTABLE_SUFFIX} )

