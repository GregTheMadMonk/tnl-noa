IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( TridiagonalMatrixExample_Constructor_init_list_1_cuda TridiagonalMatrixExample_Constructor_init_list_1.cu )
   ADD_CUSTOM_COMMAND( COMMAND TridiagonalMatrixExample_Constructor_init_list_1_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/TridiagonalMatrixExample_Constructor_init_list_1.out
                       OUTPUT TridiagonalMatrixExample_Constructor_init_list_1.out )

   CUDA_ADD_EXECUTABLE( TridiagonalMatrixExample_getSerializationType_cuda TridiagonalMatrixExample_getSerializationType.cu )
   ADD_CUSTOM_COMMAND( COMMAND TridiagonalMatrixExample_getSerializationType_cuda > 
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/TridiagonalMatrixExample_getSerializationType.out
                       OUTPUT TridiagonalMatrixExample_getSerializationType.out )

   CUDA_ADD_EXECUTABLE( TridiagonalMatrixExample_setElements_cuda TridiagonalMatrixExample_setElements.cu )
   ADD_CUSTOM_COMMAND( COMMAND TridiagonalMatrixExample_setElements_cuda > 
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/TridiagonalMatrixExample_setElements.out
                       OUTPUT TridiagonalMatrixExample_setElements.out )

   CUDA_ADD_EXECUTABLE( TridiagonalMatrixExample_getCompressedRowLengths_cuda TridiagonalMatrixExample_getCompressedRowLengths.cu )
   ADD_CUSTOM_COMMAND( COMMAND TridiagonalMatrixExample_getCompressedRowLengths_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/TridiagonalMatrixExample_getCompressedRowLengths.out
                       OUTPUT TridiagonalMatrixExample_getCompressedRowLengths.out )

   CUDA_ADD_EXECUTABLE( TridiagonalMatrixExample_getConstRow_cuda TridiagonalMatrixExample_getConstRow.cu )
   ADD_CUSTOM_COMMAND( COMMAND TridiagonalMatrixExample_getConstRow_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/TridiagonalMatrixExample_getConstRow.out
                       OUTPUT TridiagonalMatrixExample_getConstRow.out )

   CUDA_ADD_EXECUTABLE( TridiagonalMatrixExample_getRow_cuda TridiagonalMatrixExample_getRow.cu )
   ADD_CUSTOM_COMMAND( COMMAND TridiagonalMatrixExample_getRow_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/TridiagonalMatrixExample_getRow.out
                       OUTPUT TridiagonalMatrixExample_getRow.out )

# This example does not work with nvcc 10.1. Restore it here when it works.
#   CUDA_ADD_EXECUTABLE( TridiagonalMatrixExample_setElement_cuda TridiagonalMatrixExample_setElement.cu )
#   ADD_CUSTOM_COMMAND( COMMAND TridiagonalMatrixExample_setElement_cuda >
#                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/TridiagonalMatrixExample_setElement.out
#                       OUTPUT TridiagonalMatrixExample_setElement.out )

   CUDA_ADD_EXECUTABLE( TridiagonalMatrixExample_addElement_cuda TridiagonalMatrixExample_addElement.cu )
   ADD_CUSTOM_COMMAND( COMMAND TridiagonalMatrixExample_addElement_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/TridiagonalMatrixExample_addElement.out
                       OUTPUT TridiagonalMatrixExample_addElement.out )

   CUDA_ADD_EXECUTABLE( TridiagonalMatrixExample_getElement_cuda TridiagonalMatrixExample_getElement.cu )
   ADD_CUSTOM_COMMAND( COMMAND TridiagonalMatrixExample_getElement_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/TridiagonalMatrixExample_getElement.out
                       OUTPUT TridiagonalMatrixExample_getElement.out )

   CUDA_ADD_EXECUTABLE( TridiagonalMatrixExample_rowsReduction_cuda TridiagonalMatrixExample_rowsReduction.cu )
   ADD_CUSTOM_COMMAND( COMMAND TridiagonalMatrixExample_rowsReduction_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/TridiagonalMatrixExample_rowsReduction.out
                       OUTPUT TridiagonalMatrixExample_rowsReduction.out )

   CUDA_ADD_EXECUTABLE( TridiagonalMatrixExample_allRowsReduction_cuda TridiagonalMatrixExample_allRowsReduction.cu )
   ADD_CUSTOM_COMMAND( COMMAND TridiagonalMatrixExample_allRowsReduction_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/TridiagonalMatrixExample_allRowsReduction.out
                       OUTPUT TridiagonalMatrixExample_allRowsReduction.out )

   CUDA_ADD_EXECUTABLE( TridiagonalMatrixExample_forElements_cuda TridiagonalMatrixExample_forElements.cu )
   ADD_CUSTOM_COMMAND( COMMAND TridiagonalMatrixExample_forElements_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/TridiagonalMatrixExample_forElements.out
                       OUTPUT TridiagonalMatrixExample_forElements.out )

   CUDA_ADD_EXECUTABLE( TridiagonalMatrixExample_forEachElement_cuda TridiagonalMatrixExample_forEachElement.cu )
   ADD_CUSTOM_COMMAND( COMMAND TridiagonalMatrixExample_forEachElement_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/TridiagonalMatrixExample_forEachElement.out
                       OUTPUT TridiagonalMatrixExample_forEachElement.out )

   CUDA_ADD_EXECUTABLE( TridiagonalMatrixViewExample_getCompressedRowLengths_cuda TridiagonalMatrixViewExample_getCompressedRowLengths.cu )
   ADD_CUSTOM_COMMAND( COMMAND TridiagonalMatrixViewExample_getCompressedRowLengths_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/TridiagonalMatrixViewExample_getCompressedRowLengths.out
                       OUTPUT TridiagonalMatrixViewExample_getCompressedRowLengths.out )

   CUDA_ADD_EXECUTABLE( TridiagonalMatrixViewExample_getConstRow_cuda TridiagonalMatrixViewExample_getConstRow.cu )
   ADD_CUSTOM_COMMAND( COMMAND TridiagonalMatrixViewExample_getConstRow_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/TridiagonalMatrixViewExample_getConstRow.out
                       OUTPUT TridiagonalMatrixViewExample_getConstRow.out )

   CUDA_ADD_EXECUTABLE( TridiagonalMatrixViewExample_getRow_cuda TridiagonalMatrixViewExample_getRow.cu )
   ADD_CUSTOM_COMMAND( COMMAND TridiagonalMatrixViewExample_getRow_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/TridiagonalMatrixViewExample_getRow.out
                       OUTPUT TridiagonalMatrixViewExample_getRow.out )

   CUDA_ADD_EXECUTABLE( TridiagonalMatrixViewExample_setElement_cuda TridiagonalMatrixViewExample_setElement.cu )
   ADD_CUSTOM_COMMAND( COMMAND TridiagonalMatrixViewExample_setElement_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/TridiagonalMatrixViewExample_setElement.out
                       OUTPUT TridiagonalMatrixViewExample_setElement.out )

   CUDA_ADD_EXECUTABLE( TridiagonalMatrixViewExample_addElement_cuda TridiagonalMatrixViewExample_addElement.cu )
   ADD_CUSTOM_COMMAND( COMMAND TridiagonalMatrixViewExample_addElement_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/TridiagonalMatrixViewExample_addElement.out
                       OUTPUT TridiagonalMatrixViewExample_addElement.out )

   CUDA_ADD_EXECUTABLE( TridiagonalMatrixViewExample_getElement_cuda TridiagonalMatrixViewExample_getElement.cu )
   ADD_CUSTOM_COMMAND( COMMAND TridiagonalMatrixViewExample_getElement_cuda >
                       ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/TridiagonalMatrixViewExample_getElement.out
                       OUTPUT TridiagonalMatrixViewExample_getElement.out )

   CUDA_ADD_EXECUTABLE( TridiagonalMatrixViewExample_rowsReduction_cuda TridiagonalMatrixViewExample_rowsReduction.cu )
   ADD_CUSTOM_COMMAND( COMMAND TridiagonalMatrixViewExample_rowsReduction_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/TridiagonalMatrixViewExample_rowsReduction.out
                       OUTPUT TridiagonalMatrixViewExample_rowsReduction.out )

   CUDA_ADD_EXECUTABLE( TridiagonalMatrixViewExample_allRowsReduction_cuda TridiagonalMatrixViewExample_allRowsReduction.cu )
   ADD_CUSTOM_COMMAND( COMMAND TridiagonalMatrixViewExample_allRowsReduction_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/TridiagonalMatrixViewExample_allRowsReduction.out
                       OUTPUT TridiagonalMatrixViewExample_allRowsReduction.out )

   CUDA_ADD_EXECUTABLE( TridiagonalMatrixViewExample_forElements_cuda TridiagonalMatrixViewExample_forElements.cu )
   ADD_CUSTOM_COMMAND( COMMAND TridiagonalMatrixViewExample_forElements_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/TridiagonalMatrixViewExample_forElements.out
                       OUTPUT TridiagonalMatrixViewExample_forElements.out )

   CUDA_ADD_EXECUTABLE( TridiagonalMatrixViewExample_forEachElement_cuda TridiagonalMatrixViewExample_forEachElement.cu )
   ADD_CUSTOM_COMMAND( COMMAND TridiagonalMatrixViewExample_forEachElement_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/TridiagonalMatrixViewExample_forEachElement.out
                       OUTPUT TridiagonalMatrixViewExample_forEachElement.out )

ELSE()

   ADD_EXECUTABLE( TridiagonalMatrixExample_Constructor_init_list_1 TridiagonalMatrixExample_Constructor_init_list_1.cpp )
   ADD_CUSTOM_COMMAND( COMMAND TridiagonalMatrixExample_Constructor_init_list_1 >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/TridiagonalMatrixExample_Constructor_init_list_1.out
                       OUTPUT TridiagonalMatrixExample_Constructor_init_list_1.out )

   ADD_EXECUTABLE( TridiagonalMatrixExample_getSerializationType TridiagonalMatrixExample_getSerializationType.cpp )
   ADD_CUSTOM_COMMAND( COMMAND TridiagonalMatrixExample_getSerializationType >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/TridiagonalMatrixExample_getSerializationType.out
                       OUTPUT TridiagonalMatrixExample_getSerializationType.out )

   ADD_EXECUTABLE( TridiagonalMatrixExample_setElements TridiagonalMatrixExample_setElements.cpp )
   ADD_CUSTOM_COMMAND( COMMAND TridiagonalMatrixExample_setElements >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/TridiagonalMatrixExample_setElements.out
                       OUTPUT TridiagonalMatrixExample_setElements.out )

   ADD_EXECUTABLE( TridiagonalMatrixExample_getCompressedRowLengths TridiagonalMatrixExample_getCompressedRowLengths.cpp )
   ADD_CUSTOM_COMMAND( COMMAND TridiagonalMatrixExample_getCompressedRowLengths >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/TridiagonalMatrixExample_getCompressedRowLengths.out
                       OUTPUT TridiagonalMatrixExample_getCompressedRowLengths.out )

   ADD_EXECUTABLE( TridiagonalMatrixExample_getConstRow TridiagonalMatrixExample_getConstRow.cpp )
   ADD_CUSTOM_COMMAND( COMMAND TridiagonalMatrixExample_getConstRow >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/TridiagonalMatrixExample_getConstRow.out
                       OUTPUT TridiagonalMatrixExample_getConstRow.out )

   ADD_EXECUTABLE( TridiagonalMatrixExample_getRow TridiagonalMatrixExample_getRow.cpp )
   ADD_CUSTOM_COMMAND( COMMAND TridiagonalMatrixExample_getRow >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/TridiagonalMatrixExample_getRow.out
                       OUTPUT TridiagonalMatrixExample_getRow.out )

#  This example does not work with nvcc 10.1. Restore it here when it works.
#   ADD_EXECUTABLE( TridiagonalMatrixExample_setElement TridiagonalMatrixExample_setElement.cpp )
#   ADD_CUSTOM_COMMAND( COMMAND TridiagonalMatrixExample_setElement >
#                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/TridiagonalMatrixExample_setElement.out
#                       OUTPUT TridiagonalMatrixExample_setElement.out )

   ADD_EXECUTABLE( TridiagonalMatrixExample_addElement TridiagonalMatrixExample_addElement.cpp )
   ADD_CUSTOM_COMMAND( COMMAND TridiagonalMatrixExample_addElement >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/TridiagonalMatrixExample_addElement.out
                       OUTPUT TridiagonalMatrixExample_addElement.out )

   ADD_EXECUTABLE( TridiagonalMatrixExample_getElement TridiagonalMatrixExample_getElement.cpp )
   ADD_CUSTOM_COMMAND( COMMAND TridiagonalMatrixExample_getElement >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/TridiagonalMatrixExample_getElement.out
                       OUTPUT TridiagonalMatrixExample_getElement.out )

   ADD_EXECUTABLE( TridiagonalMatrixExample_rowsReduction TridiagonalMatrixExample_rowsReduction.cpp )
   ADD_CUSTOM_COMMAND( COMMAND TridiagonalMatrixExample_rowsReduction >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/TridiagonalMatrixExample_rowsReduction.out
                       OUTPUT TridiagonalMatrixExample_rowsReduction.out )

   ADD_EXECUTABLE( TridiagonalMatrixExample_allRowsReduction TridiagonalMatrixExample_allRowsReduction.cpp )
   ADD_CUSTOM_COMMAND( COMMAND TridiagonalMatrixExample_allRowsReduction >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/TridiagonalMatrixExample_allRowsReduction.out
                       OUTPUT TridiagonalMatrixExample_allRowsReduction.out )

   ADD_EXECUTABLE( TridiagonalMatrixExample_forElements TridiagonalMatrixExample_forElements.cpp )
   ADD_CUSTOM_COMMAND( COMMAND TridiagonalMatrixExample_forElements >
                       ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/TridiagonalMatrixExample_forElements.out
                       OUTPUT TridiagonalMatrixExample_forElements.out )

   ADD_EXECUTABLE( TridiagonalMatrixExample_forEachElement TridiagonalMatrixExample_forEachElement.cpp )
   ADD_CUSTOM_COMMAND( COMMAND TridiagonalMatrixExample_forEachElement >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/TridiagonalMatrixExample_forEachElement.out
                       OUTPUT TridiagonalMatrixExample_forEachElement.out )

   ADD_EXECUTABLE( TridiagonalMatrixViewExample_getCompressedRowLengths TridiagonalMatrixViewExample_getCompressedRowLengths.cpp )
   ADD_CUSTOM_COMMAND( COMMAND TridiagonalMatrixViewExample_getCompressedRowLengths >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/TridiagonalMatrixViewExample_getCompressedRowLengths.out
                       OUTPUT TridiagonalMatrixViewExample_getCompressedRowLengths.out )

   ADD_EXECUTABLE( TridiagonalMatrixViewExample_getConstRow TridiagonalMatrixViewExample_getConstRow.cpp )
   ADD_CUSTOM_COMMAND( COMMAND TridiagonalMatrixViewExample_getConstRow >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/TridiagonalMatrixViewExample_getConstRow.out
                       OUTPUT TridiagonalMatrixViewExample_getConstRow.out )

   ADD_EXECUTABLE( TridiagonalMatrixViewExample_getRow TridiagonalMatrixViewExample_getRow.cpp )
   ADD_CUSTOM_COMMAND( COMMAND TridiagonalMatrixViewExample_getRow >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/TridiagonalMatrixViewExample_getRow.out
                       OUTPUT TridiagonalMatrixViewExample_getRow.out )

   ADD_EXECUTABLE( TridiagonalMatrixViewExample_setElement TridiagonalMatrixViewExample_setElement.cpp )
   ADD_CUSTOM_COMMAND( COMMAND TridiagonalMatrixViewExample_setElement >
                       ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/TridiagonalMatrixViewExample_setElement.out
                       OUTPUT TridiagonalMatrixViewExample_setElement.out )

   ADD_EXECUTABLE( TridiagonalMatrixViewExample_addElement TridiagonalMatrixViewExample_addElement.cpp )
   ADD_CUSTOM_COMMAND( COMMAND TridiagonalMatrixViewExample_addElement >
                       ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/TridiagonalMatrixViewExample_addElement.out
                       OUTPUT TridiagonalMatrixViewExample_addElement.out )

   ADD_EXECUTABLE( TridiagonalMatrixViewExample_getElement TridiagonalMatrixViewExample_getElement.cpp )
   ADD_CUSTOM_COMMAND( COMMAND TridiagonalMatrixViewExample_getElement >
                       ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/TridiagonalMatrixViewExample_getElement.out
                       OUTPUT TridiagonalMatrixViewExample_getElement.out )

   ADD_EXECUTABLE( TridiagonalMatrixViewExample_rowsReduction TridiagonalMatrixViewExample_rowsReduction.cpp )
   ADD_CUSTOM_COMMAND( COMMAND TridiagonalMatrixViewExample_rowsReduction >
                       ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/TridiagonalMatrixViewExample_rowsReduction.out
                       OUTPUT TridiagonalMatrixViewExample_rowsReduction.out )

   ADD_EXECUTABLE( TridiagonalMatrixViewExample_allRowsReduction TridiagonalMatrixViewExample_allRowsReduction.cpp )
   ADD_CUSTOM_COMMAND( COMMAND TridiagonalMatrixViewExample_allRowsReduction >
                       ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/TridiagonalMatrixViewExample_allRowsReduction.out
                       OUTPUT TridiagonalMatrixViewExample_allRowsReduction.out )

   ADD_EXECUTABLE( TridiagonalMatrixViewExample_forElements TridiagonalMatrixViewExample_forElements.cpp )
   ADD_CUSTOM_COMMAND( COMMAND TridiagonalMatrixViewExample_forElements >
                       ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/TridiagonalMatrixViewExample_forElements.out
                       OUTPUT TridiagonalMatrixViewExample_forElements.out )

   ADD_EXECUTABLE( TridiagonalMatrixViewExample_forEachElement TridiagonalMatrixViewExample_forEachElement.cpp )
   ADD_CUSTOM_COMMAND( COMMAND TridiagonalMatrixViewExample_forEachElement >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/TridiagonalMatrixViewExample_forEachElement.out
                       OUTPUT TridiagonalMatrixViewExample_forEachElement.out )

ENDIF()

   ADD_EXECUTABLE( TridiagonalMatrixExample_setElement TridiagonalMatrixExample_setElement.cpp )
   ADD_CUSTOM_COMMAND( COMMAND TridiagonalMatrixExample_setElement >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/TridiagonalMatrixExample_setElement.out
                       OUTPUT TridiagonalMatrixExample_setElement.out )


ADD_CUSTOM_TARGET( RunTridiagonalMatricesExamples ALL DEPENDS
   TridiagonalMatrixExample_Constructor_init_list_1.out
   TridiagonalMatrixExample_getSerializationType.out
   TridiagonalMatrixExample_setElements.out
   TridiagonalMatrixExample_getCompressedRowLengths.out
   TridiagonalMatrixExample_getConstRow.out
   TridiagonalMatrixExample_getRow.out
   TridiagonalMatrixExample_setElement.out
   TridiagonalMatrixExample_addElement.out
   TridiagonalMatrixExample_getElement.out
   TridiagonalMatrixExample_rowsReduction.out
   TridiagonalMatrixExample_allRowsReduction.out
   TridiagonalMatrixExample_forElements.out
   TridiagonalMatrixExample_forEachElement.out
   TridiagonalMatrixViewExample_getCompressedRowLengths.out
   TridiagonalMatrixViewExample_getConstRow.out
   TridiagonalMatrixViewExample_getRow.out
   TridiagonalMatrixViewExample_setElement.out
   TridiagonalMatrixViewExample_addElement.out
   TridiagonalMatrixViewExample_getElement.out
   TridiagonalMatrixViewExample_rowsReduction.out
   TridiagonalMatrixViewExample_allRowsReduction.out
   TridiagonalMatrixViewExample_forElements.out
   TridiagonalMatrixViewExample_forEachElement.out
)

