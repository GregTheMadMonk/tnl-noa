IF( BUILD_CUDA )
   #CUDA_ADD_EXECUTABLE( Legacy_SparseMatrixCopyTest SparseMatrixCopyTest.cu OPTIONS ${CXX_TESTS_FLAGS} )
   #TARGET_LINK_LIBRARIES( Legacy_SparseMatrixCopyTest ${GTEST_BOTH_LIBRARIES} )

   CUDA_ADD_EXECUTABLE( Legacy_SparseMatrixTest SparseMatrixTest.cu OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( Legacy_SparseMatrixTest ${GTEST_BOTH_LIBRARIES} )

   CUDA_ADD_EXECUTABLE( Legacy_SparseMatrixTest_AdEllpack SparseMatrixTest_AdEllpack.cu OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( Legacy_SparseMatrixTest_AdEllpack ${GTEST_BOTH_LIBRARIES} )

   CUDA_ADD_EXECUTABLE( Legacy_SparseMatrixTest_BiEllpack SparseMatrixTest_BiEllpack.cu OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( Legacy_SparseMatrixTest_BiEllpack ${GTEST_BOTH_LIBRARIES} )

   CUDA_ADD_EXECUTABLE( Legacy_SparseMatrixTest_ChunkedEllpack SparseMatrixTest_ChunkedEllpack.cu OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( Legacy_SparseMatrixTest_ChunkedEllpack ${GTEST_BOTH_LIBRARIES} )

   CUDA_ADD_EXECUTABLE( Legacy_SparseMatrixTest_CSR SparseMatrixTest_CSR.cu OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( Legacy_SparseMatrixTest_CSR ${GTEST_BOTH_LIBRARIES} )

   CUDA_ADD_EXECUTABLE( Legacy_SparseMatrixTest_Ellpack SparseMatrixTest_Ellpack.cu OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( Legacy_SparseMatrixTest_Ellpack ${GTEST_BOTH_LIBRARIES} )

   CUDA_ADD_EXECUTABLE( Legacy_SparseMatrixTest_SlicedEllpack SparseMatrixTest_SlicedEllpack.cu OPTIONS ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( Legacy_SparseMatrixTest_SlicedEllpack ${GTEST_BOTH_LIBRARIES} )

ELSE(  BUILD_CUDA )
   #ADD_EXECUTABLE( Legacy_SparseMatrixCopyTest SparseMatrixCopyTest.cpp )
   #TARGET_COMPILE_OPTIONS( Legacy_SparseMatrixCopyTest PRIVATE ${CXX_TESTS_FLAGS} )
   #TARGET_LINK_LIBRARIES( Legacy_SparseMatrixCopyTest ${GTEST_BOTH_LIBRARIES} )

   ADD_EXECUTABLE( Legacy_SparseMatrixTest SparseMatrixTest.cpp )
   TARGET_COMPILE_OPTIONS( Legacy_SparseMatrixTest PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( Legacy_SparseMatrixTest ${GTEST_BOTH_LIBRARIES} )

   ADD_EXECUTABLE( Legacy_SparseMatrixTest_AdEllpack SparseMatrixTest_AdEllpack.cpp )
   TARGET_COMPILE_OPTIONS( Legacy_SparseMatrixTest_AdEllpack PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( Legacy_SparseMatrixTest_AdEllpack ${GTEST_BOTH_LIBRARIES} )

   ADD_EXECUTABLE( Legacy_SparseMatrixTest_BiEllpack SparseMatrixTest_BiEllpack.cpp )
   TARGET_COMPILE_OPTIONS( Legacy_SparseMatrixTest_BiEllpack PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( Legacy_SparseMatrixTest_BiEllpack ${GTEST_BOTH_LIBRARIES} )

   ADD_EXECUTABLE( Legacy_SparseMatrixTest_ChunkedEllpack SparseMatrixTest_ChunkedEllpack.cpp )
   TARGET_COMPILE_OPTIONS( Legacy_SparseMatrixTest_ChunkedEllpack PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( Legacy_SparseMatrixTest_ChunkedEllpack ${GTEST_BOTH_LIBRARIES} )

   ADD_EXECUTABLE( Legacy_SparseMatrixTest_CSR SparseMatrixTest_CSR.cpp )
   TARGET_COMPILE_OPTIONS( Legacy_SparseMatrixTest_CSR PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( Legacy_SparseMatrixTest_CSR ${GTEST_BOTH_LIBRARIES} )

   ADD_EXECUTABLE( Legacy_SparseMatrixTest_Ellpack SparseMatrixTest_Ellpack.cpp )
   TARGET_COMPILE_OPTIONS( Legacy_SparseMatrixTest_Ellpack PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( Legacy_SparseMatrixTest_Ellpack ${GTEST_BOTH_LIBRARIES} )

   ADD_EXECUTABLE( Legacy_SparseMatrixTest_SlicedEllpack SparseMatrixTest_SlicedEllpack.cpp )
   TARGET_COMPILE_OPTIONS( Legacy_SparseMatrixTest_SlicedEllpack PRIVATE ${CXX_TESTS_FLAGS} )
   TARGET_LINK_LIBRARIES( Legacy_SparseMatrixTest_SlicedEllpack ${GTEST_BOTH_LIBRARIES} )

ENDIF( BUILD_CUDA )


#ADD_TEST( Legacy_SparseMatrixCopyTest ${EXECUTABLE_OUTPUT_PATH}/Legacy_SparseMatrixCopyTest${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( Legacy_SparseMatrixTest ${EXECUTABLE_OUTPUT_PATH}/Legacy_SparseMatrixTest${CMAKE_EXECUTABLE_SUFFIX} )
# TODO: Uncomment the following when AdEllpack works
#ADD_TEST( SparseMatrixTest_AdEllpack ${EXECUTABLE_OUTPUT_PATH}/SparseMatrixTest_AdEllpack${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( Legacy_SparseMatrixTest_BiEllpack ${EXECUTABLE_OUTPUT_PATH}/Legacy_SparseMatrixTest_BiEllpack${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( Legacy_SparseMatrixTest_ChunkedEllpack ${EXECUTABLE_OUTPUT_PATH}/Legacy_SparseMatrixTest_ChunkedEllpack${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( Legacy_SparseMatrixTest_CSR ${EXECUTABLE_OUTPUT_PATH}/Legacy_SparseMatrixTest_CSR${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( Legacy_SparseMatrixTest_Ellpack ${EXECUTABLE_OUTPUT_PATH}/Legacy_SparseMatrixTest_Ellpack${CMAKE_EXECUTABLE_SUFFIX} )
ADD_TEST( Legacy_SparseMatrixTest_SlicedEllpack ${EXECUTABLE_OUTPUT_PATH}/Legacy_SparseMatrixTest_SlicedEllpack${CMAKE_EXECUTABLE_SUFFIX} )
# TODO: DenseMatrixTest is not finished
#ADD_TEST( DenseMatrixTest ${EXECUTABLE_OUTPUT_PATH}/DenseMatrixTest${CMAKE_EXECUTABLE_SUFFIX} )
