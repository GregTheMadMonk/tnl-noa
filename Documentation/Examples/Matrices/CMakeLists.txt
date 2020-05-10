IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( DenseMatrixExample_Constructor_init_list_cuda DenseMatrixExample_Constructor_init_list.cu )
   ADD_CUSTOM_COMMAND( COMMAND DenseMatrixExample_Constructor_init_list_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/DenseMatrixExample_Constructor_init_list.out
                       OUTPUT DenseMatrixExample_Constructor_init_list.out )

   CUDA_ADD_EXECUTABLE( DenseMatrixExample_setElements_cuda DenseMatrixExample_setElements.cu )
   ADD_CUSTOM_COMMAND( COMMAND DenseMatrixExample_setElements_cuda > 
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/DenseMatrixExample_setElements.out
                       OUTPUT DenseMatrixExample_setElements.out )

   CUDA_ADD_EXECUTABLE( DenseMatrixExample_getCompressedRowLengths_cuda DenseMatrixExample_getCompressedRowLengths.cu )
   ADD_CUSTOM_COMMAND( COMMAND DenseMatrixExample_getCompressedRowLengths_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/DenseMatrixExample_getCompressedRowLengths.out
                       OUTPUT DenseMatrixExample_getCompressedRowLengths.out )

   CUDA_ADD_EXECUTABLE( DenseMatrixExample_getElementsCount_cuda DenseMatrixExample_getElementsCount.cu )
   ADD_CUSTOM_COMMAND( COMMAND DenseMatrixExample_getElementsCount_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/DenseMatrixExample_getElementsCount.out
                       OUTPUT DenseMatrixExample_getElementsCount.out )

   CUDA_ADD_EXECUTABLE( DenseMatrixExample_getConstRow_cuda DenseMatrixExample_getConstRow.cu )
   ADD_CUSTOM_COMMAND( COMMAND DenseMatrixExample_getConstRow_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/DenseMatrixExample_getConstRow.out
                       OUTPUT DenseMatrixExample_getConstRow.out )

   CUDA_ADD_EXECUTABLE( DenseMatrixExample_getRow_cuda DenseMatrixExample_getRow.cu )
   ADD_CUSTOM_COMMAND( COMMAND DenseMatrixExample_getRow_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/DenseMatrixExample_getRow.out
                       OUTPUT DenseMatrixExample_getRow.out )

   CUDA_ADD_EXECUTABLE( DenseMatrixExample_setElement_cuda DenseMatrixExample_setElement.cu )
   ADD_CUSTOM_COMMAND( COMMAND DenseMatrixExample_setElement_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/DenseMatrixExample_setElement.out
                       OUTPUT DenseMatrixExample_setElement.out )

   CUDA_ADD_EXECUTABLE( DenseMatrixExample_addElement_cuda DenseMatrixExample_addElement.cu )
   ADD_CUSTOM_COMMAND( COMMAND DenseMatrixExample_addElement_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/DenseMatrixExample_addElement.out
                       OUTPUT DenseMatrixExample_addElement.out )

   CUDA_ADD_EXECUTABLE( DenseMatrixExample_getElement_cuda DenseMatrixExample_getElement.cu )
   ADD_CUSTOM_COMMAND( COMMAND DenseMatrixExample_getElement_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/DenseMatrixExample_getElement.out
                       OUTPUT DenseMatrixExample_getElement.out )

   CUDA_ADD_EXECUTABLE( DenseMatrixExample_rowsReduction_cuda DenseMatrixExample_rowsReduction.cu )
   ADD_CUSTOM_COMMAND( COMMAND DenseMatrixExample_rowsReduction_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/DenseMatrixExample_rowsReduction.out
                       OUTPUT DenseMatrixExample_rowsReduction.out )

   CUDA_ADD_EXECUTABLE( DenseMatrixExample_allRowsReduction_cuda DenseMatrixExample_allRowsReduction.cu )
   ADD_CUSTOM_COMMAND( COMMAND DenseMatrixExample_allRowsReduction_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/DenseMatrixExample_allRowsReduction.out
                       OUTPUT DenseMatrixExample_allRowsReduction.out )

   CUDA_ADD_EXECUTABLE( DenseMatrixExample_forRows_cuda DenseMatrixExample_forRows.cu )
   ADD_CUSTOM_COMMAND( COMMAND DenseMatrixExample_forRows_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/DenseMatrixExample_forRows.out
                       OUTPUT DenseMatrixExample_forRows.out )

   CUDA_ADD_EXECUTABLE( DenseMatrixExample_forAllRows_cuda DenseMatrixExample_forAllRows.cu )
   ADD_CUSTOM_COMMAND( COMMAND DenseMatrixExample_forAllRows_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/DenseMatrixExample_forAllRows.out
                       OUTPUT DenseMatrixExample_forAllRows.out )

   CUDA_ADD_EXECUTABLE( DenseMatrixViewExample_constructor_cuda DenseMatrixViewExample_constructor.cu )
   ADD_CUSTOM_COMMAND( COMMAND DenseMatrixViewExample_constructor_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/DenseMatrixViewExample_constructor.out
                       OUTPUT DenseMatrixViewExample_constructor.out )

   CUDA_ADD_EXECUTABLE( DenseMatrixViewExample_getCompressedRowLengths_cuda DenseMatrixViewExample_getCompressedRowLengths.cu )
   ADD_CUSTOM_COMMAND( COMMAND DenseMatrixViewExample_getCompressedRowLengths_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/DenseMatrixViewExample_getCompressedRowLengths.out
                       OUTPUT DenseMatrixViewExample_getCompressedRowLengths.out )

   CUDA_ADD_EXECUTABLE( DenseMatrixViewExample_getElementsCount_cuda DenseMatrixViewExample_getElementsCount.cu )
   ADD_CUSTOM_COMMAND( COMMAND DenseMatrixViewExample_getElementsCount_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/DenseMatrixViewExample_getElementsCount.out
                       OUTPUT DenseMatrixViewExample_getElementsCount.out )


ELSE()
   ADD_EXECUTABLE( DenseMatrixExample_Constructor_init_list DenseMatrixExample_Constructor_init_list.cpp )
   ADD_CUSTOM_COMMAND( COMMAND DenseMatrixExample_Constructor_init_list >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/DenseMatrixExample_Constructor_init_list.out
                       OUTPUT DenseMatrixExample_Constructor_init_list.out )

   ADD_EXECUTABLE( DenseMatrixExample_setElements DenseMatrixExample_setElements.cpp )
   ADD_CUSTOM_COMMAND( COMMAND DenseMatrixExample_setElements > 
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/DenseMatrixExample_setElements.out
                       OUTPUT DenseMatrixExample_setElements.out )

   ADD_EXECUTABLE( DenseMatrixExample_getCompressedRowLengths DenseMatrixExample_getCompressedRowLengths.cpp )
   ADD_CUSTOM_COMMAND( COMMAND DenseMatrixExample_getCompressedRowLengths >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/DenseMatrixExample_getCompressedRowLengths.out
                       OUTPUT DenseMatrixExample_getCompressedRowLengths.out )

   ADD_EXECUTABLE( DenseMatrixExample_getElementsCount DenseMatrixExample_getElementsCount.cpp )
   ADD_CUSTOM_COMMAND( COMMAND DenseMatrixExample_getElementsCount >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/DenseMatrixExample_getElementsCount.out
                       OUTPUT DenseMatrixExample_getElementsCount.out )

   ADD_EXECUTABLE( DenseMatrixExample_getConstRow DenseMatrixExample_getConstRow.cpp )
   ADD_CUSTOM_COMMAND( COMMAND DenseMatrixExample_getConstRow >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/DenseMatrixExample_getConstRow.out
                       OUTPUT DenseMatrixExample_getConstRow.out )

   ADD_EXECUTABLE( DenseMatrixExample_getRow DenseMatrixExample_getRow.cpp )
   ADD_CUSTOM_COMMAND( COMMAND DenseMatrixExample_getRow >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/DenseMatrixExample_getRow.out
                       OUTPUT DenseMatrixExample_getRow.out )

   ADD_EXECUTABLE( DenseMatrixExample_setElement DenseMatrixExample_setElement.cpp )
   ADD_CUSTOM_COMMAND( COMMAND DenseMatrixExample_setElement >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/DenseMatrixExample_setElement.out
                       OUTPUT DenseMatrixExample_setElement.out )

   ADD_EXECUTABLE( DenseMatrixExample_addElement DenseMatrixExample_addElement.cpp )
   ADD_CUSTOM_COMMAND( COMMAND DenseMatrixExample_addElement >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/DenseMatrixExample_addElement.out
                       OUTPUT DenseMatrixExample_addElement.out )

   ADD_EXECUTABLE( DenseMatrixExample_getElement DenseMatrixExample_getElement.cpp )
   ADD_CUSTOM_COMMAND( COMMAND DenseMatrixExample_getElement >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/DenseMatrixExample_getElement.out
                       OUTPUT DenseMatrixExample_getElement.out )

   ADD_EXECUTABLE( DenseMatrixExample_rowsReduction DenseMatrixExample_rowsReduction.cpp )
   ADD_CUSTOM_COMMAND( COMMAND DenseMatrixExample_rowsReduction >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/DenseMatrixExample_rowsReduction.out
                       OUTPUT DenseMatrixExample_rowsReduction.out )

   ADD_EXECUTABLE( DenseMatrixExample_allRowsReduction DenseMatrixExample_allRowsReduction.cpp )
   ADD_CUSTOM_COMMAND( COMMAND DenseMatrixExample_allRowsReduction >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/DenseMatrixExample_allRowsReduction.out
                       OUTPUT DenseMatrixExample_allRowsReduction.out )

   ADD_EXECUTABLE( DenseMatrixExample_forRows DenseMatrixExample_forRows.cpp )
   ADD_CUSTOM_COMMAND( COMMAND DenseMatrixExample_forRows >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/DenseMatrixExample_forRows.out
                       OUTPUT DenseMatrixExample_forRows.out )

   ADD_EXECUTABLE( DenseMatrixExample_forAllRows DenseMatrixExample_forAllRows.cpp )
   ADD_CUSTOM_COMMAND( COMMAND DenseMatrixExample_forAllRows >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/DenseMatrixExample_forAllRows.out
                       OUTPUT DenseMatrixExample_forAllRows.out )

   ADD_EXECUTABLE( DenseMatrixViewExample_constructor DenseMatrixViewExample_constructor.cpp )
   ADD_CUSTOM_COMMAND( COMMAND DenseMatrixViewExample_constructor >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/DenseMatrixViewExample_constructor.out
                       OUTPUT DenseMatrixViewExample_constructor.out )

   ADD_EXECUTABLE( DenseMatrixViewExample_getCompressedRowLengths DenseMatrixViewExample_getCompressedRowLengths.cpp )
   ADD_CUSTOM_COMMAND( COMMAND DenseMatrixViewExample_getCompressedRowLengths >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/DenseMatrixViewExample_getCompressedRowLengths.out
                       OUTPUT DenseMatrixViewExample_getCompressedRowLengths.out )

   ADD_EXECUTABLE( DenseMatrixViewExample_getElementsCount DenseMatrixViewExample_getElementsCount.cpp )
   ADD_CUSTOM_COMMAND( COMMAND DenseMatrixViewExample_getElementsCount >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/DenseMatrixViewExample_getElementsCount.out
                       OUTPUT DenseMatrixViewExample_getElementsCount.out )

ENDIF()

ADD_CUSTOM_TARGET( RunMatricesExamples ALL DEPENDS
   DenseMatrixExample_Constructor_init_list.out
   DenseMatrixExample_setElements.out
   DenseMatrixExample_getCompressedRowLengths.out
   DenseMatrixExample_getElementsCount.out
   DenseMatrixExample_getConstRow.out
   DenseMatrixExample_getRow.out
   DenseMatrixExample_setElement.out
   DenseMatrixExample_addElement.out
   DenseMatrixExample_getElement.out
   DenseMatrixExample_rowsReduction.out
   DenseMatrixExample_allRowsReduction.out
   DenseMatrixExample_forRows.out
   DenseMatrixExample_forAllRows.out
   DenseMatrixViewExample_constructor.out
   DenseMatrixViewExample_getCompressedRowLengths.out
   DenseMatrixViewExample_getElementsCount.out
)

