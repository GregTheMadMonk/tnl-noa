IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( SparseMatrixExample_Constructor_init_list_1_cuda SparseMatrixExample_Constructor_init_list_1.cu )
   ADD_CUSTOM_COMMAND( COMMAND SparseMatrixExample_Constructor_init_list_1_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SparseMatrixExample_Constructor_init_list_1.out
                       OUTPUT SparseMatrixExample_Constructor_init_list_1.out )

   CUDA_ADD_EXECUTABLE( SparseMatrixExample_Constructor_init_list_2_cuda SparseMatrixExample_Constructor_init_list_2.cu )
   ADD_CUSTOM_COMMAND( COMMAND SparseMatrixExample_Constructor_init_list_2_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SparseMatrixExample_Constructor_init_list_2.out
                       OUTPUT SparseMatrixExample_Constructor_init_list_2.out )

   CUDA_ADD_EXECUTABLE( SparseMatrixExample_Constructor_rowCapacities_vector_cuda SparseMatrixExample_Constructor_rowCapacities_vector.cu )
   ADD_CUSTOM_COMMAND( COMMAND SparseMatrixExample_Constructor_rowCapacities_vector_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SparseMatrixExample_Constructor_rowCapacities_vector.out
                       OUTPUT SparseMatrixExample_Constructor_rowCapacities_vector.out )

   CUDA_ADD_EXECUTABLE( SparseMatrixExample_Constructor_std_map_cuda SparseMatrixExample_Constructor_std_map.cu )
   ADD_CUSTOM_COMMAND( COMMAND SparseMatrixExample_Constructor_std_map_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SparseMatrixExample_Constructor_std_map.out
                       OUTPUT SparseMatrixExample_Constructor_std_map.out )

   CUDA_ADD_EXECUTABLE( SparseMatrixExample_getSerializationType_cuda SparseMatrixExample_getSerializationType.cu )
   ADD_CUSTOM_COMMAND( COMMAND SparseMatrixExample_getSerializationType_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SparseMatrixExample_getSerializationType.out
                       OUTPUT SparseMatrixExample_getSerializationType.out )

   CUDA_ADD_EXECUTABLE( SparseMatrixExample_setRowCapacities_cuda SparseMatrixExample_setRowCapacities.cu )
   ADD_CUSTOM_COMMAND( COMMAND SparseMatrixExample_setRowCapacities_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SparseMatrixExample_setRowCapacities.out
                       OUTPUT SparseMatrixExample_setRowCapacities.out )

   CUDA_ADD_EXECUTABLE( SparseMatrixExample_setElements_cuda SparseMatrixExample_setElements.cu )
   ADD_CUSTOM_COMMAND( COMMAND SparseMatrixExample_setElements_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SparseMatrixExample_setElements.out
                       OUTPUT SparseMatrixExample_setElements.out )

   CUDA_ADD_EXECUTABLE( SparseMatrixExample_setElements_map_cuda SparseMatrixExample_setElements_map.cu )
   ADD_CUSTOM_COMMAND( COMMAND SparseMatrixExample_setElements_map_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SparseMatrixExample_setElements_map.out
                       OUTPUT SparseMatrixExample_setElements_map.out )

   CUDA_ADD_EXECUTABLE( SparseMatrixExample_getCompressedRowLengths_cuda SparseMatrixExample_getCompressedRowLengths.cu )
   ADD_CUSTOM_COMMAND( COMMAND SparseMatrixExample_getCompressedRowLengths_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SparseMatrixExample_getCompressedRowLengths.out
                       OUTPUT SparseMatrixExample_getCompressedRowLengths.out )

   CUDA_ADD_EXECUTABLE( SparseMatrixExample_getConstRow_cuda SparseMatrixExample_getConstRow.cu )
   ADD_CUSTOM_COMMAND( COMMAND SparseMatrixExample_getConstRow_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SparseMatrixExample_getConstRow.out
                       OUTPUT SparseMatrixExample_getConstRow.out )

   CUDA_ADD_EXECUTABLE( SparseMatrixExample_getRow_cuda SparseMatrixExample_getRow.cu )
   ADD_CUSTOM_COMMAND( COMMAND SparseMatrixExample_getRow_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SparseMatrixExample_getRow.out
                       OUTPUT SparseMatrixExample_getRow.out )

   CUDA_ADD_EXECUTABLE( SparseMatrixExample_setElement_cuda SparseMatrixExample_setElement.cu )
   ADD_CUSTOM_COMMAND( COMMAND SparseMatrixExample_setElement_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SparseMatrixExample_setElement.out
                       OUTPUT SparseMatrixExample_setElement.out )

   CUDA_ADD_EXECUTABLE( SparseMatrixExample_addElement_cuda SparseMatrixExample_addElement.cu )
   ADD_CUSTOM_COMMAND( COMMAND SparseMatrixExample_addElement_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SparseMatrixExample_addElement.out
                       OUTPUT SparseMatrixExample_addElement.out )

   CUDA_ADD_EXECUTABLE( SparseMatrixExample_getElement_cuda SparseMatrixExample_getElement.cu )
   ADD_CUSTOM_COMMAND( COMMAND SparseMatrixExample_getElement_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SparseMatrixExample_getElement.out
                       OUTPUT SparseMatrixExample_getElement.out )

   CUDA_ADD_EXECUTABLE( SparseMatrixExample_rowsReduction_cuda SparseMatrixExample_rowsReduction.cu )
   ADD_CUSTOM_COMMAND( COMMAND SparseMatrixExample_rowsReduction_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SparseMatrixExample_rowsReduction.out
                       OUTPUT SparseMatrixExample_rowsReduction.out )

   CUDA_ADD_EXECUTABLE( SparseMatrixExample_allRowsReduction_cuda SparseMatrixExample_allRowsReduction.cu )
   ADD_CUSTOM_COMMAND( COMMAND SparseMatrixExample_allRowsReduction_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SparseMatrixExample_allRowsReduction.out
                       OUTPUT SparseMatrixExample_allRowsReduction.out )

   CUDA_ADD_EXECUTABLE( SparseMatrixExample_forElements_cuda SparseMatrixExample_forElements.cu )
   ADD_CUSTOM_COMMAND( COMMAND SparseMatrixExample_forElements_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SparseMatrixExample_forElements.out
                       OUTPUT SparseMatrixExample_forElements.out )

   CUDA_ADD_EXECUTABLE( SparseMatrixExample_forEachElement_cuda SparseMatrixExample_forEachElement.cu )
   ADD_CUSTOM_COMMAND( COMMAND SparseMatrixExample_forEachElement_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SparseMatrixExample_forEachElement.out
                       OUTPUT SparseMatrixExample_forEachElement.out )

   CUDA_ADD_EXECUTABLE( SparseMatrixViewExample_getSerializationType_cuda SparseMatrixViewExample_getSerializationType.cu )
   ADD_CUSTOM_COMMAND( COMMAND SparseMatrixViewExample_getSerializationType_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SparseMatrixViewExample_getSerializationType.out
                       OUTPUT SparseMatrixViewExample_getSerializationType.out )

   CUDA_ADD_EXECUTABLE( SparseMatrixViewExample_getCompressedRowLengths_cuda SparseMatrixViewExample_getCompressedRowLengths.cu )
   ADD_CUSTOM_COMMAND( COMMAND SparseMatrixViewExample_getCompressedRowLengths_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SparseMatrixViewExample_getCompressedRowLengths.out
                       OUTPUT SparseMatrixViewExample_getCompressedRowLengths.out )

   CUDA_ADD_EXECUTABLE( SparseMatrixViewExample_getConstRow_cuda SparseMatrixViewExample_getConstRow.cu )
   ADD_CUSTOM_COMMAND( COMMAND SparseMatrixViewExample_getConstRow_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SparseMatrixViewExample_getConstRow.out
                       OUTPUT SparseMatrixViewExample_getConstRow.out )

   CUDA_ADD_EXECUTABLE( SparseMatrixViewExample_getRow_cuda SparseMatrixViewExample_getRow.cu )
   ADD_CUSTOM_COMMAND( COMMAND SparseMatrixViewExample_getRow_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SparseMatrixViewExample_getRow.out
                       OUTPUT SparseMatrixViewExample_getRow.out )

   CUDA_ADD_EXECUTABLE( SparseMatrixViewExample_setElement_cuda SparseMatrixViewExample_setElement.cu )
   ADD_CUSTOM_COMMAND( COMMAND SparseMatrixViewExample_setElement_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SparseMatrixViewExample_setElement.out
                       OUTPUT SparseMatrixViewExample_setElement.out )

   CUDA_ADD_EXECUTABLE( SparseMatrixViewExample_addElement_cuda SparseMatrixViewExample_addElement.cu )
   ADD_CUSTOM_COMMAND( COMMAND SparseMatrixViewExample_addElement_cuda >
                       ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SparseMatrixViewExample_addElement.out
                       OUTPUT SparseMatrixViewExample_addElement.out )

   CUDA_ADD_EXECUTABLE( SparseMatrixViewExample_getElement_cuda SparseMatrixViewExample_getElement.cu )
   ADD_CUSTOM_COMMAND( COMMAND SparseMatrixViewExample_getElement_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SparseMatrixViewExample_getElement.out
                       OUTPUT SparseMatrixViewExample_getElement.out )

   CUDA_ADD_EXECUTABLE( SparseMatrixViewExample_rowsReduction_cuda SparseMatrixViewExample_rowsReduction.cu )
   ADD_CUSTOM_COMMAND( COMMAND SparseMatrixViewExample_rowsReduction_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SparseMatrixViewExample_rowsReduction.out
                       OUTPUT SparseMatrixViewExample_rowsReduction.out )

   CUDA_ADD_EXECUTABLE( SparseMatrixViewExample_allRowsReduction_cuda SparseMatrixViewExample_allRowsReduction.cu )
   ADD_CUSTOM_COMMAND( COMMAND SparseMatrixViewExample_allRowsReduction_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SparseMatrixViewExample_allRowsReduction.out
                       OUTPUT SparseMatrixViewExample_allRowsReduction.out )

   CUDA_ADD_EXECUTABLE( SparseMatrixViewExample_forElements_cuda SparseMatrixViewExample_forElements.cu )
   ADD_CUSTOM_COMMAND( COMMAND SparseMatrixViewExample_forElements_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SparseMatrixViewExample_forElements.out
                       OUTPUT SparseMatrixViewExample_forElements.out )

   CUDA_ADD_EXECUTABLE( SparseMatrixViewExample_forEachElement_cuda SparseMatrixViewExample_forEachElement.cu )
   ADD_CUSTOM_COMMAND( COMMAND SparseMatrixViewExample_forEachElement_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SparseMatrixViewExample_forEachElement.out
                       OUTPUT SparseMatrixViewExample_forEachElement.out )

ELSE()
   ADD_EXECUTABLE( SparseMatrixExample_Constructor_init_list_1 SparseMatrixExample_Constructor_init_list_1.cpp )
   ADD_CUSTOM_COMMAND( COMMAND SparseMatrixExample_Constructor_init_list_1 >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SparseMatrixExample_Constructor_init_list_1.out
                       OUTPUT SparseMatrixExample_Constructor_init_list_1.out )

   ADD_EXECUTABLE( SparseMatrixExample_Constructor_init_list_2 SparseMatrixExample_Constructor_init_list_2.cpp )
   ADD_CUSTOM_COMMAND( COMMAND SparseMatrixExample_Constructor_init_list_2 >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SparseMatrixExample_Constructor_init_list_2.out
                       OUTPUT SparseMatrixExample_Constructor_init_list_2.out )

   ADD_EXECUTABLE( SparseMatrixExample_Constructor_rowCapacities_vector SparseMatrixExample_Constructor_rowCapacities_vector.cpp )
   ADD_CUSTOM_COMMAND( COMMAND SparseMatrixExample_Constructor_rowCapacities_vector >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SparseMatrixExample_Constructor_rowCapacities_vector.out
                       OUTPUT SparseMatrixExample_Constructor_rowCapacities_vector.out )

   ADD_EXECUTABLE( SparseMatrixExample_Constructor_std_map SparseMatrixExample_Constructor_std_map.cpp )
   ADD_CUSTOM_COMMAND( COMMAND SparseMatrixExample_Constructor_std_map >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SparseMatrixExample_Constructor_std_map.out
                       OUTPUT SparseMatrixExample_Constructor_std_map.out )

   ADD_EXECUTABLE( SparseMatrixExample_getSerializationType SparseMatrixExample_getSerializationType.cpp )
   ADD_CUSTOM_COMMAND( COMMAND SparseMatrixExample_getSerializationType >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SparseMatrixExample_getSerializationType.out
                       OUTPUT SparseMatrixExample_getSerializationType.out )

   ADD_EXECUTABLE( SparseMatrixExample_setRowCapacities SparseMatrixExample_setRowCapacities.cpp )
   ADD_CUSTOM_COMMAND( COMMAND SparseMatrixExample_setRowCapacities >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SparseMatrixExample_setRowCapacities.out
                       OUTPUT SparseMatrixExample_setRowCapacities.out )

   ADD_EXECUTABLE( SparseMatrixExample_setElements SparseMatrixExample_setElements.cpp )
   ADD_CUSTOM_COMMAND( COMMAND SparseMatrixExample_setElements >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SparseMatrixExample_setElements.out
                       OUTPUT SparseMatrixExample_setElements.out )

   ADD_EXECUTABLE( SparseMatrixExample_setElements_map SparseMatrixExample_setElements_map.cpp )
   ADD_CUSTOM_COMMAND( COMMAND SparseMatrixExample_setElements_map >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SparseMatrixExample_setElements_map.out
                       OUTPUT SparseMatrixExample_setElements_map.out )

   ADD_EXECUTABLE( SparseMatrixExample_getCompressedRowLengths SparseMatrixExample_getCompressedRowLengths.cpp )
   ADD_CUSTOM_COMMAND( COMMAND SparseMatrixExample_getCompressedRowLengths >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SparseMatrixExample_getCompressedRowLengths.out
                       OUTPUT SparseMatrixExample_getCompressedRowLengths.out )

   ADD_EXECUTABLE( SparseMatrixExample_getConstRow SparseMatrixExample_getConstRow.cpp )
   ADD_CUSTOM_COMMAND( COMMAND SparseMatrixExample_getConstRow >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SparseMatrixExample_getConstRow.out
                       OUTPUT SparseMatrixExample_getConstRow.out )

   ADD_EXECUTABLE( SparseMatrixExample_getRow SparseMatrixExample_getRow.cpp )
   ADD_CUSTOM_COMMAND( COMMAND SparseMatrixExample_getRow >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SparseMatrixExample_getRow.out
                       OUTPUT SparseMatrixExample_getRow.out )

   ADD_EXECUTABLE( SparseMatrixExample_setElement SparseMatrixExample_setElement.cpp )
   ADD_CUSTOM_COMMAND( COMMAND SparseMatrixExample_setElement >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SparseMatrixExample_setElement.out
                       OUTPUT SparseMatrixExample_setElement.out )

   ADD_EXECUTABLE( SparseMatrixExample_addElement SparseMatrixExample_addElement.cpp )
   ADD_CUSTOM_COMMAND( COMMAND SparseMatrixExample_addElement >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SparseMatrixExample_addElement.out
                       OUTPUT SparseMatrixExample_addElement.out )

   ADD_EXECUTABLE( SparseMatrixExample_getElement SparseMatrixExample_getElement.cpp )
   ADD_CUSTOM_COMMAND( COMMAND SparseMatrixExample_getElement >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SparseMatrixExample_getElement.out
                       OUTPUT SparseMatrixExample_getElement.out )

   ADD_EXECUTABLE( SparseMatrixExample_rowsReduction SparseMatrixExample_rowsReduction.cpp )
   ADD_CUSTOM_COMMAND( COMMAND SparseMatrixExample_rowsReduction >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SparseMatrixExample_rowsReduction.out
                       OUTPUT SparseMatrixExample_rowsReduction.out )

   ADD_EXECUTABLE( SparseMatrixExample_allRowsReduction SparseMatrixExample_allRowsReduction.cpp )
   ADD_CUSTOM_COMMAND( COMMAND SparseMatrixExample_allRowsReduction >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SparseMatrixExample_allRowsReduction.out
                       OUTPUT SparseMatrixExample_allRowsReduction.out )

   ADD_EXECUTABLE( SparseMatrixExample_forElements SparseMatrixExample_forElements.cpp )
   ADD_CUSTOM_COMMAND( COMMAND SparseMatrixExample_forElements >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SparseMatrixExample_forElements.out
                       OUTPUT SparseMatrixExample_forElements.out )

   ADD_EXECUTABLE( SparseMatrixExample_forEachElement SparseMatrixExample_forEachElement.cpp )
   ADD_CUSTOM_COMMAND( COMMAND SparseMatrixExample_forEachElement >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SparseMatrixExample_forEachElement.out
                       OUTPUT SparseMatrixExample_forEachElement.out )

   ADD_EXECUTABLE( SparseMatrixViewExample_getSerializationType SparseMatrixViewExample_getSerializationType.cpp )
   ADD_CUSTOM_COMMAND( COMMAND SparseMatrixViewExample_getSerializationType >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SparseMatrixViewExample_getSerializationType.out
                       OUTPUT SparseMatrixViewExample_getSerializationType.out )

   ADD_EXECUTABLE( SparseMatrixViewExample_getCompressedRowLengths SparseMatrixViewExample_getCompressedRowLengths.cpp )
   ADD_CUSTOM_COMMAND( COMMAND SparseMatrixViewExample_getCompressedRowLengths >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SparseMatrixViewExample_getCompressedRowLengths.out
                       OUTPUT SparseMatrixViewExample_getCompressedRowLengths.out )

   ADD_EXECUTABLE( SparseMatrixViewExample_getConstRow SparseMatrixViewExample_getConstRow.cpp )
   ADD_CUSTOM_COMMAND( COMMAND SparseMatrixViewExample_getConstRow >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SparseMatrixViewExample_getConstRow.out
                       OUTPUT SparseMatrixViewExample_getConstRow.out )

   ADD_EXECUTABLE( SparseMatrixViewExample_getRow SparseMatrixViewExample_getRow.cpp )
   ADD_CUSTOM_COMMAND( COMMAND SparseMatrixViewExample_getRow >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SparseMatrixViewExample_getRow.out
                       OUTPUT SparseMatrixViewExample_getRow.out )

   ADD_EXECUTABLE( SparseMatrixViewExample_setElement SparseMatrixViewExample_setElement.cpp )
   ADD_CUSTOM_COMMAND( COMMAND SparseMatrixViewExample_setElement >
                       ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SparseMatrixViewExample_setElement.out
                       OUTPUT SparseMatrixViewExample_setElement.out )

   ADD_EXECUTABLE( SparseMatrixViewExample_addElement SparseMatrixViewExample_addElement.cpp )
   ADD_CUSTOM_COMMAND( COMMAND SparseMatrixViewExample_addElement >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SparseMatrixViewExample_addElement.out
                       OUTPUT SparseMatrixViewExample_addElement.out )

   ADD_EXECUTABLE( SparseMatrixViewExample_getElement SparseMatrixViewExample_getElement.cpp )
   ADD_CUSTOM_COMMAND( COMMAND SparseMatrixViewExample_getElement >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SparseMatrixViewExample_getElement.out
                       OUTPUT SparseMatrixViewExample_getElement.out )

   ADD_EXECUTABLE( SparseMatrixViewExample_rowsReduction SparseMatrixViewExample_rowsReduction.cpp )
   ADD_CUSTOM_COMMAND( COMMAND SparseMatrixViewExample_rowsReduction >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SparseMatrixViewExample_rowsReduction.out
                       OUTPUT SparseMatrixViewExample_rowsReduction.out )

   ADD_EXECUTABLE( SparseMatrixViewExample_allRowsReduction SparseMatrixViewExample_allRowsReduction.cpp )
   ADD_CUSTOM_COMMAND( COMMAND SparseMatrixViewExample_allRowsReduction >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SparseMatrixViewExample_allRowsReduction.out
                       OUTPUT SparseMatrixViewExample_allRowsReduction.out )

   ADD_EXECUTABLE( SparseMatrixViewExample_forElements SparseMatrixViewExample_forElements.cpp )
   ADD_CUSTOM_COMMAND( COMMAND SparseMatrixViewExample_forElements >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SparseMatrixViewExample_forElements.out
                       OUTPUT SparseMatrixViewExample_forElements.out )

   ADD_EXECUTABLE( SparseMatrixViewExample_forEachElement SparseMatrixViewExample_forEachElement.cpp )
   ADD_CUSTOM_COMMAND( COMMAND SparseMatrixViewExample_forEachElement >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SparseMatrixViewExample_forEachElement.out
                       OUTPUT SparseMatrixViewExample_forEachElement.out )

ENDIF()

ADD_CUSTOM_TARGET( RunSparseMatricesExamples ALL DEPENDS
   SparseMatrixExample_Constructor_init_list_1.out
   SparseMatrixExample_Constructor_init_list_2.out
   SparseMatrixExample_Constructor_rowCapacities_vector.out
   SparseMatrixExample_Constructor_std_map.out
   SparseMatrixExample_getSerializationType.out
   SparseMatrixExample_setRowCapacities.out
   SparseMatrixExample_setElements.out
   SparseMatrixExample_setElements_map.out
   SparseMatrixExample_getCompressedRowLengths.out
   SparseMatrixExample_getConstRow.out
   SparseMatrixExample_getRow.out
   SparseMatrixExample_setElement.out
   SparseMatrixExample_addElement.out
   SparseMatrixExample_getElement.out
   SparseMatrixExample_rowsReduction.out
   SparseMatrixExample_allRowsReduction.out
   SparseMatrixExample_forElements.out
   SparseMatrixExample_forEachElement.out
   SparseMatrixViewExample_getSerializationType.out
   SparseMatrixViewExample_getCompressedRowLengths.out
   SparseMatrixViewExample_getConstRow.out
   SparseMatrixViewExample_getRow.out
   SparseMatrixViewExample_setElement.out
   SparseMatrixViewExample_addElement.out
   SparseMatrixViewExample_getElement.out
   SparseMatrixViewExample_rowsReduction.out
   SparseMatrixViewExample_allRowsReduction.out
   SparseMatrixViewExample_forElements.out
   SparseMatrixViewExample_forEachElement.out
)

