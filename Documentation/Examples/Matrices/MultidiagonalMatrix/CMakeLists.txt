IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( MultidiagonalMatrixExample_Constructor_cuda MultidiagonalMatrixExample_Constructor.cu )
   ADD_CUSTOM_COMMAND( COMMAND MultidiagonalMatrixExample_Constructor_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/MultidiagonalMatrixExample_Constructor.out
                       OUTPUT MultidiagonalMatrixExample_Constructor.out )

   CUDA_ADD_EXECUTABLE( MultidiagonalMatrixExample_Constructor_init_list_1_cuda MultidiagonalMatrixExample_Constructor_init_list_1.cu )
   ADD_CUSTOM_COMMAND( COMMAND MultidiagonalMatrixExample_Constructor_init_list_1_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/MultidiagonalMatrixExample_Constructor_init_list_1.out
                       OUTPUT MultidiagonalMatrixExample_Constructor_init_list_1.out )

   CUDA_ADD_EXECUTABLE( MultidiagonalMatrixExample_Constructor_init_list_2_cuda MultidiagonalMatrixExample_Constructor_init_list_2.cu )
   ADD_CUSTOM_COMMAND( COMMAND MultidiagonalMatrixExample_Constructor_init_list_2_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/MultidiagonalMatrixExample_Constructor_init_list_2.out
                       OUTPUT MultidiagonalMatrixExample_Constructor_init_list_2.out )

   CUDA_ADD_EXECUTABLE( MultidiagonalMatrixExample_getSerializationType_cuda MultidiagonalMatrixExample_getSerializationType.cu )
   ADD_CUSTOM_COMMAND( COMMAND MultidiagonalMatrixExample_getSerializationType_cuda > 
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/MultidiagonalMatrixExample_getSerializationType.out
                       OUTPUT MultidiagonalMatrixExample_getSerializationType.out )

   CUDA_ADD_EXECUTABLE( MultidiagonalMatrixExample_setElements_cuda MultidiagonalMatrixExample_setElements.cu )
   ADD_CUSTOM_COMMAND( COMMAND MultidiagonalMatrixExample_setElements_cuda > 
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/MultidiagonalMatrixExample_setElements.out
                       OUTPUT MultidiagonalMatrixExample_setElements.out )

   CUDA_ADD_EXECUTABLE( MultidiagonalMatrixExample_getCompressedRowLengths_cuda MultidiagonalMatrixExample_getCompressedRowLengths.cu )
   ADD_CUSTOM_COMMAND( COMMAND MultidiagonalMatrixExample_getCompressedRowLengths_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/MultidiagonalMatrixExample_getCompressedRowLengths.out
                       OUTPUT MultidiagonalMatrixExample_getCompressedRowLengths.out )

   CUDA_ADD_EXECUTABLE( MultidiagonalMatrixExample_getConstRow_cuda MultidiagonalMatrixExample_getConstRow.cu )
   ADD_CUSTOM_COMMAND( COMMAND MultidiagonalMatrixExample_getConstRow_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/MultidiagonalMatrixExample_getConstRow.out
                       OUTPUT MultidiagonalMatrixExample_getConstRow.out )

   CUDA_ADD_EXECUTABLE( MultidiagonalMatrixExample_getRow_cuda MultidiagonalMatrixExample_getRow.cu )
   ADD_CUSTOM_COMMAND( COMMAND MultidiagonalMatrixExample_getRow_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/MultidiagonalMatrixExample_getRow.out
                       OUTPUT MultidiagonalMatrixExample_getRow.out )

# This example does not work with nvcc 10.1. Restore it here when it works.
#   CUDA_ADD_EXECUTABLE( MultidiagonalMatrixExample_setElement_cuda MultidiagonalMatrixExample_setElement.cu )
#   ADD_CUSTOM_COMMAND( COMMAND MultidiagonalMatrixExample_setElement_cuda >
#                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/MultidiagonalMatrixExample_setElement.out
#                       OUTPUT MultidiagonalMatrixExample_setElement.out )

   CUDA_ADD_EXECUTABLE( MultidiagonalMatrixExample_addElement_cuda MultidiagonalMatrixExample_addElement.cu )
   ADD_CUSTOM_COMMAND( COMMAND MultidiagonalMatrixExample_addElement_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/MultidiagonalMatrixExample_addElement.out
                       OUTPUT MultidiagonalMatrixExample_addElement.out )

   CUDA_ADD_EXECUTABLE( MultidiagonalMatrixExample_getElement_cuda MultidiagonalMatrixExample_getElement.cu )
   ADD_CUSTOM_COMMAND( COMMAND MultidiagonalMatrixExample_getElement_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/MultidiagonalMatrixExample_getElement.out
                       OUTPUT MultidiagonalMatrixExample_getElement.out )

   CUDA_ADD_EXECUTABLE( MultidiagonalMatrixExample_rowsReduction_cuda MultidiagonalMatrixExample_rowsReduction.cu )
   ADD_CUSTOM_COMMAND( COMMAND MultidiagonalMatrixExample_rowsReduction_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/MultidiagonalMatrixExample_rowsReduction.out
                       OUTPUT MultidiagonalMatrixExample_rowsReduction.out )

   CUDA_ADD_EXECUTABLE( MultidiagonalMatrixExample_allRowsReduction_cuda MultidiagonalMatrixExample_allRowsReduction.cu )
   ADD_CUSTOM_COMMAND( COMMAND MultidiagonalMatrixExample_allRowsReduction_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/MultidiagonalMatrixExample_allRowsReduction.out
                       OUTPUT MultidiagonalMatrixExample_allRowsReduction.out )

   CUDA_ADD_EXECUTABLE( MultidiagonalMatrixExample_forElements_cuda MultidiagonalMatrixExample_forElements.cu )
   ADD_CUSTOM_COMMAND( COMMAND MultidiagonalMatrixExample_forElements_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/MultidiagonalMatrixExample_forElements.out
                       OUTPUT MultidiagonalMatrixExample_forElements.out )

   CUDA_ADD_EXECUTABLE( MultidiagonalMatrixExample_forEachElement_cuda MultidiagonalMatrixExample_forEachElement.cu )
   ADD_CUSTOM_COMMAND( COMMAND MultidiagonalMatrixExample_forEachElement_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/MultidiagonalMatrixExample_forEachElement.out
                       OUTPUT MultidiagonalMatrixExample_forEachElement.out )

   CUDA_ADD_EXECUTABLE( MultidiagonalMatrixViewExample_getCompressedRowLengths_cuda MultidiagonalMatrixViewExample_getCompressedRowLengths.cu )
   ADD_CUSTOM_COMMAND( COMMAND MultidiagonalMatrixViewExample_getCompressedRowLengths_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/MultidiagonalMatrixViewExample_getCompressedRowLengths.out
                       OUTPUT MultidiagonalMatrixViewExample_getCompressedRowLengths.out )

   CUDA_ADD_EXECUTABLE( MultidiagonalMatrixViewExample_getConstRow_cuda MultidiagonalMatrixViewExample_getConstRow.cu )
   ADD_CUSTOM_COMMAND( COMMAND MultidiagonalMatrixViewExample_getConstRow_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/MultidiagonalMatrixViewExample_getConstRow.out
                       OUTPUT MultidiagonalMatrixViewExample_getConstRow.out )

   CUDA_ADD_EXECUTABLE( MultidiagonalMatrixViewExample_getRow_cuda MultidiagonalMatrixViewExample_getRow.cu )
   ADD_CUSTOM_COMMAND( COMMAND MultidiagonalMatrixViewExample_getRow_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/MultidiagonalMatrixViewExample_getRow.out
                       OUTPUT MultidiagonalMatrixViewExample_getRow.out )

   CUDA_ADD_EXECUTABLE( MultidiagonalMatrixViewExample_setElement_cuda MultidiagonalMatrixViewExample_setElement.cu )
   ADD_CUSTOM_COMMAND( COMMAND MultidiagonalMatrixViewExample_setElement_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/MultidiagonalMatrixViewExample_setElement.out
                       OUTPUT MultidiagonalMatrixViewExample_setElement.out )

   CUDA_ADD_EXECUTABLE( MultidiagonalMatrixViewExample_addElement_cuda MultidiagonalMatrixViewExample_addElement.cu )
   ADD_CUSTOM_COMMAND( COMMAND MultidiagonalMatrixViewExample_addElement_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/MultidiagonalMatrixViewExample_addElement.out
                       OUTPUT MultidiagonalMatrixViewExample_addElement.out )

   CUDA_ADD_EXECUTABLE( MultidiagonalMatrixViewExample_getElement_cuda MultidiagonalMatrixViewExample_getElement.cu )
   ADD_CUSTOM_COMMAND( COMMAND MultidiagonalMatrixViewExample_getElement_cuda >
                       ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/MultidiagonalMatrixViewExample_getElement.out
                       OUTPUT MultidiagonalMatrixViewExample_getElement.out )

   CUDA_ADD_EXECUTABLE( MultidiagonalMatrixViewExample_rowsReduction_cuda MultidiagonalMatrixViewExample_rowsReduction.cu )
   ADD_CUSTOM_COMMAND( COMMAND MultidiagonalMatrixViewExample_rowsReduction_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/MultidiagonalMatrixViewExample_rowsReduction.out
                       OUTPUT MultidiagonalMatrixViewExample_rowsReduction.out )

   CUDA_ADD_EXECUTABLE( MultidiagonalMatrixViewExample_allRowsReduction_cuda MultidiagonalMatrixViewExample_allRowsReduction.cu )
   ADD_CUSTOM_COMMAND( COMMAND MultidiagonalMatrixViewExample_allRowsReduction_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/MultidiagonalMatrixViewExample_allRowsReduction.out
                       OUTPUT MultidiagonalMatrixViewExample_allRowsReduction.out )

   CUDA_ADD_EXECUTABLE( MultidiagonalMatrixViewExample_forElements_cuda MultidiagonalMatrixViewExample_forElements.cu )
   ADD_CUSTOM_COMMAND( COMMAND MultidiagonalMatrixViewExample_forElements_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/MultidiagonalMatrixViewExample_forElements.out
                       OUTPUT MultidiagonalMatrixViewExample_forElements.out )

   CUDA_ADD_EXECUTABLE( MultidiagonalMatrixViewExample_forEachElement_cuda MultidiagonalMatrixViewExample_forEachElement.cu )
   ADD_CUSTOM_COMMAND( COMMAND MultidiagonalMatrixViewExample_forEachElement_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/MultidiagonalMatrixViewExample_forEachElement.out
                       OUTPUT MultidiagonalMatrixViewExample_forEachElement.out )

ELSE()
   ADD_EXECUTABLE( MultidiagonalMatrixExample_Constructor MultidiagonalMatrixExample_Constructor.cpp )
   ADD_CUSTOM_COMMAND( COMMAND MultidiagonalMatrixExample_Constructor >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/MultidiagonalMatrixExample_Constructor.out
                       OUTPUT MultidiagonalMatrixExample_Constructor.out )

   ADD_EXECUTABLE( MultidiagonalMatrixExample_Constructor_init_list_1 MultidiagonalMatrixExample_Constructor_init_list_1.cpp )
   ADD_CUSTOM_COMMAND( COMMAND MultidiagonalMatrixExample_Constructor_init_list_1 >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/MultidiagonalMatrixExample_Constructor_init_list_1.out
                       OUTPUT MultidiagonalMatrixExample_Constructor_init_list_1.out )

   ADD_EXECUTABLE( MultidiagonalMatrixExample_Constructor_init_list_2 MultidiagonalMatrixExample_Constructor_init_list_2.cpp )
   ADD_CUSTOM_COMMAND( COMMAND MultidiagonalMatrixExample_Constructor_init_list_2 >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/MultidiagonalMatrixExample_Constructor_init_list_2.out
                       OUTPUT MultidiagonalMatrixExample_Constructor_init_list_2.out )

   ADD_EXECUTABLE( MultidiagonalMatrixExample_getSerializationType MultidiagonalMatrixExample_getSerializationType.cpp )
   ADD_CUSTOM_COMMAND( COMMAND MultidiagonalMatrixExample_getSerializationType >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/MultidiagonalMatrixExample_getSerializationType.out
                       OUTPUT MultidiagonalMatrixExample_getSerializationType.out )

   ADD_EXECUTABLE( MultidiagonalMatrixExample_setElements MultidiagonalMatrixExample_setElements.cpp )
   ADD_CUSTOM_COMMAND( COMMAND MultidiagonalMatrixExample_setElements >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/MultidiagonalMatrixExample_setElements.out
                       OUTPUT MultidiagonalMatrixExample_setElements.out )

   ADD_EXECUTABLE( MultidiagonalMatrixExample_getCompressedRowLengths MultidiagonalMatrixExample_getCompressedRowLengths.cpp )
   ADD_CUSTOM_COMMAND( COMMAND MultidiagonalMatrixExample_getCompressedRowLengths >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/MultidiagonalMatrixExample_getCompressedRowLengths.out
                       OUTPUT MultidiagonalMatrixExample_getCompressedRowLengths.out )

   ADD_EXECUTABLE( MultidiagonalMatrixExample_getConstRow MultidiagonalMatrixExample_getConstRow.cpp )
   ADD_CUSTOM_COMMAND( COMMAND MultidiagonalMatrixExample_getConstRow >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/MultidiagonalMatrixExample_getConstRow.out
                       OUTPUT MultidiagonalMatrixExample_getConstRow.out )

   ADD_EXECUTABLE( MultidiagonalMatrixExample_getRow MultidiagonalMatrixExample_getRow.cpp )
   ADD_CUSTOM_COMMAND( COMMAND MultidiagonalMatrixExample_getRow >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/MultidiagonalMatrixExample_getRow.out
                       OUTPUT MultidiagonalMatrixExample_getRow.out )

#  This example does not work with nvcc 10.1. Restore it here when it works.
#   ADD_EXECUTABLE( MultidiagonalMatrixExample_setElement MultidiagonalMatrixExample_setElement.cpp )
#   ADD_CUSTOM_COMMAND( COMMAND MultidiagonalMatrixExample_setElement >
#                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/MultidiagonalMatrixExample_setElement.out
#                       OUTPUT MultidiagonalMatrixExample_setElement.out )

   ADD_EXECUTABLE( MultidiagonalMatrixExample_addElement MultidiagonalMatrixExample_addElement.cpp )
   ADD_CUSTOM_COMMAND( COMMAND MultidiagonalMatrixExample_addElement >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/MultidiagonalMatrixExample_addElement.out
                       OUTPUT MultidiagonalMatrixExample_addElement.out )

   ADD_EXECUTABLE( MultidiagonalMatrixExample_getElement MultidiagonalMatrixExample_getElement.cpp )
   ADD_CUSTOM_COMMAND( COMMAND MultidiagonalMatrixExample_getElement >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/MultidiagonalMatrixExample_getElement.out
                       OUTPUT MultidiagonalMatrixExample_getElement.out )

   ADD_EXECUTABLE( MultidiagonalMatrixExample_rowsReduction MultidiagonalMatrixExample_rowsReduction.cpp )
   ADD_CUSTOM_COMMAND( COMMAND MultidiagonalMatrixExample_rowsReduction >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/MultidiagonalMatrixExample_rowsReduction.out
                       OUTPUT MultidiagonalMatrixExample_rowsReduction.out )

   ADD_EXECUTABLE( MultidiagonalMatrixExample_allRowsReduction MultidiagonalMatrixExample_allRowsReduction.cpp )
   ADD_CUSTOM_COMMAND( COMMAND MultidiagonalMatrixExample_allRowsReduction >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/MultidiagonalMatrixExample_allRowsReduction.out
                       OUTPUT MultidiagonalMatrixExample_allRowsReduction.out )

   ADD_EXECUTABLE( MultidiagonalMatrixExample_forElements MultidiagonalMatrixExample_forElements.cpp )
   ADD_CUSTOM_COMMAND( COMMAND MultidiagonalMatrixExample_forElements >
                       ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/MultidiagonalMatrixExample_forElements.out
                       OUTPUT MultidiagonalMatrixExample_forElements.out )

   ADD_EXECUTABLE( MultidiagonalMatrixExample_forEachElement MultidiagonalMatrixExample_forEachElement.cpp )
   ADD_CUSTOM_COMMAND( COMMAND MultidiagonalMatrixExample_forEachElement >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/MultidiagonalMatrixExample_forEachElement.out
                       OUTPUT MultidiagonalMatrixExample_forEachElement.out )

   ADD_EXECUTABLE( MultidiagonalMatrixViewExample_getCompressedRowLengths MultidiagonalMatrixViewExample_getCompressedRowLengths.cpp )
   ADD_CUSTOM_COMMAND( COMMAND MultidiagonalMatrixViewExample_getCompressedRowLengths >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/MultidiagonalMatrixViewExample_getCompressedRowLengths.out
                       OUTPUT MultidiagonalMatrixViewExample_getCompressedRowLengths.out )

   ADD_EXECUTABLE( MultidiagonalMatrixViewExample_getConstRow MultidiagonalMatrixViewExample_getConstRow.cpp )
   ADD_CUSTOM_COMMAND( COMMAND MultidiagonalMatrixViewExample_getConstRow >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/MultidiagonalMatrixViewExample_getConstRow.out
                       OUTPUT MultidiagonalMatrixViewExample_getConstRow.out )

   ADD_EXECUTABLE( MultidiagonalMatrixViewExample_getRow MultidiagonalMatrixViewExample_getRow.cpp )
   ADD_CUSTOM_COMMAND( COMMAND MultidiagonalMatrixViewExample_getRow >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/MultidiagonalMatrixViewExample_getRow.out
                       OUTPUT MultidiagonalMatrixViewExample_getRow.out )

   ADD_EXECUTABLE( MultidiagonalMatrixViewExample_setElement MultidiagonalMatrixViewExample_setElement.cpp )
   ADD_CUSTOM_COMMAND( COMMAND MultidiagonalMatrixViewExample_setElement >
                       ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/MultidiagonalMatrixViewExample_setElement.out
                       OUTPUT MultidiagonalMatrixViewExample_setElement.out )

   ADD_EXECUTABLE( MultidiagonalMatrixViewExample_addElement MultidiagonalMatrixViewExample_addElement.cpp )
   ADD_CUSTOM_COMMAND( COMMAND MultidiagonalMatrixViewExample_addElement >
                       ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/MultidiagonalMatrixViewExample_addElement.out
                       OUTPUT MultidiagonalMatrixViewExample_addElement.out )

   ADD_EXECUTABLE( MultidiagonalMatrixViewExample_getElement MultidiagonalMatrixViewExample_getElement.cpp )
   ADD_CUSTOM_COMMAND( COMMAND MultidiagonalMatrixViewExample_getElement >
                       ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/MultidiagonalMatrixViewExample_getElement.out
                       OUTPUT MultidiagonalMatrixViewExample_getElement.out )

   ADD_EXECUTABLE( MultidiagonalMatrixViewExample_rowsReduction MultidiagonalMatrixViewExample_rowsReduction.cpp )
   ADD_CUSTOM_COMMAND( COMMAND MultidiagonalMatrixViewExample_rowsReduction >
                       ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/MultidiagonalMatrixViewExample_rowsReduction.out
                       OUTPUT MultidiagonalMatrixViewExample_rowsReduction.out )

   ADD_EXECUTABLE( MultidiagonalMatrixViewExample_allRowsReduction MultidiagonalMatrixViewExample_allRowsReduction.cpp )
   ADD_CUSTOM_COMMAND( COMMAND MultidiagonalMatrixViewExample_allRowsReduction >
                       ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/MultidiagonalMatrixViewExample_allRowsReduction.out
                       OUTPUT MultidiagonalMatrixViewExample_allRowsReduction.out )

   ADD_EXECUTABLE( MultidiagonalMatrixViewExample_forElements MultidiagonalMatrixViewExample_forElements.cpp )
   ADD_CUSTOM_COMMAND( COMMAND MultidiagonalMatrixViewExample_forElements >
                       ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/MultidiagonalMatrixViewExample_forElements.out
                       OUTPUT MultidiagonalMatrixViewExample_forElements.out )

   ADD_EXECUTABLE( MultidiagonalMatrixViewExample_forEachElement MultidiagonalMatrixViewExample_forEachElement.cpp )
   ADD_CUSTOM_COMMAND( COMMAND MultidiagonalMatrixViewExample_forEachElement >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/MultidiagonalMatrixViewExample_forEachElement.out
                       OUTPUT MultidiagonalMatrixViewExample_forEachElement.out )

ENDIF()

   ADD_EXECUTABLE( MultidiagonalMatrixExample_setElement MultidiagonalMatrixExample_setElement.cpp )
   ADD_CUSTOM_COMMAND( COMMAND MultidiagonalMatrixExample_setElement >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/MultidiagonalMatrixExample_setElement.out
                       OUTPUT MultidiagonalMatrixExample_setElement.out )


ADD_CUSTOM_TARGET( RunMultidiagonalMatricesExamples ALL DEPENDS
   MultidiagonalMatrixExample_Constructor.out
   MultidiagonalMatrixExample_Constructor_init_list_1.out
   MultidiagonalMatrixExample_Constructor_init_list_2.out
   MultidiagonalMatrixExample_getSerializationType.out
   MultidiagonalMatrixExample_setElements.out
   MultidiagonalMatrixExample_getCompressedRowLengths.out
   MultidiagonalMatrixExample_getConstRow.out
   MultidiagonalMatrixExample_getRow.out
   MultidiagonalMatrixExample_setElement.out
   MultidiagonalMatrixExample_addElement.out
   MultidiagonalMatrixExample_getElement.out
   MultidiagonalMatrixExample_rowsReduction.out
   MultidiagonalMatrixExample_allRowsReduction.out
   MultidiagonalMatrixExample_forElements.out
   MultidiagonalMatrixExample_forEachElement.out
   MultidiagonalMatrixViewExample_getCompressedRowLengths.out
   MultidiagonalMatrixViewExample_getConstRow.out
   MultidiagonalMatrixViewExample_getRow.out
   MultidiagonalMatrixViewExample_setElement.out
   MultidiagonalMatrixViewExample_addElement.out
   MultidiagonalMatrixViewExample_getElement.out
   MultidiagonalMatrixViewExample_rowsReduction.out
   MultidiagonalMatrixViewExample_allRowsReduction.out
   MultidiagonalMatrixViewExample_forElements.out
   MultidiagonalMatrixViewExample_forEachElement.out
)

