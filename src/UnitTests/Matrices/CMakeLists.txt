IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( SparseMatrixCopyTest${mpiExt}${debugExt} SparseMatrixCopyTest.h SparseMatrixCopyTest.cu OPTIONS "${CXX_TESTS_FLAGS}" )
   TARGET_LINK_LIBRARIES( SparseMatrixCopyTest${mpiExt}${debugExt} ${GTEST_BOTH_LIBRARIES}
                                                           tnl${mpiExt}${debugExt}-${tnlVersion} )
ELSE(  BUILD_CUDA )
   ADD_EXECUTABLE( SparseMatrixCopyTest${mpiExt}${debugExt} SparseMatrixCopyTest.h SparseMatrixCopyTest.cpp )
   SET_TARGET_PROPERTIES( SparseMatrixCopyTest${mpiExt}${debugExt} PROPERTIES COMPILE_FLAGS "${CXX_TESTS_FLAGS}" )
   TARGET_LINK_LIBRARIES( SparseMatrixCopyTest${mpiExt}${debugExt} ${GTEST_BOTH_LIBRARIES}
                                                           tnl${mpiExt}${debugExt}-${tnlVersion} )
ENDIF( BUILD_CUDA )


ADD_TEST( SparseMatrixCopyTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/SparseMatrixCopyTest${mpiExt}${debugExt} )
