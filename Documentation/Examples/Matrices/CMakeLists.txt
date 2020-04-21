IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( DenseMatrixExample_Constructor_init_list_cuda DenseMatrixExample_Constructor_init_list.cu )
   ADD_CUSTOM_COMMAND( COMMAND DenseMatrixExample_Constructor_init_list_cuda > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/DenseMatrixExample_Constructor_init_list.out 
                       OUTPUT DenseMatrixExample_Constructor_init_list.out )

   CUDA_ADD_EXECUTABLE( DenseMatrixExample_setElements_cuda DenseMatrixExample_setElements.cu )
   ADD_CUSTOM_COMMAND( COMMAND DenseMatrixExample_setElements_cuda > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/DenseMatrixExample_setElements.out 
                       OUTPUT DenseMatrixExample_setElements.out )

   CUDA_ADD_EXECUTABLE( DenseMatrixExample_getCompressedRowLengths_cuda DenseMatrixExample_getCompressedRowLengths.cu )
   ADD_CUSTOM_COMMAND( COMMAND DenseMatrixExample_getCompressedRowLengths_cuda > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/DenseMatrixExample_getCompressedRowLengths.out 
                       OUTPUT DenseMatrixExample_getCompressedRowLengths.out )

ELSE()
   ADD_EXECUTABLE( DenseMatrixExample_Constructor_init_list DenseMatrixExample_Constructor_init_list.cpp )
   ADD_CUSTOM_COMMAND( COMMAND DenseMatrixExample_Constructor_init_list > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/DenseMatrixExample_Constructor_init_list.out 
                       OUTPUT DenseMatrixExample_Constructor_init_list.out )

   ADD_EXECUTABLE( DenseMatrixExample_setElements DenseMatrixExample_setElements.cpp )
   ADD_CUSTOM_COMMAND( COMMAND DenseMatrixExample_setElements > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/DenseMatrixExample_setElements.out 
                       OUTPUT DenseMatrixExample_setElements.out )

   ADD_EXECUTABLE( DenseMatrixExample_getCompressedRowLengths DenseMatrixExample_getCompressedRowLengths.cpp )
   ADD_CUSTOM_COMMAND( COMMAND DenseMatrixExample_getCompressedRowLengths > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/DenseMatrixExample_getCompressedRowLengths.out 
                       OUTPUT DenseMatrixExample_getCompressedRowLengths.out )
ENDIF()

IF( BUILD_CUDA )
ADD_CUSTOM_TARGET( RunMatricesExamples-cuda ALL DEPENDS
   DenseMatrixExample_Constructor_init_list.out
   DenseMatrixExample_setElements.out
   DenseMatrixExample_getCompressedRowLengths.out
   )
ELSE()
ADD_CUSTOM_TARGET( RunMatricesExamples ALL DEPENDS
   DenseMatrixExample_Constructor_init_list.out
   DenseMatrixExample_setElements.out
   DenseMatrixExample_getCompressedRowLengths.out
   )
ENDIF()
