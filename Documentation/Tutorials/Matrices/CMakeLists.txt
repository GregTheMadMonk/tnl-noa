IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( DenseMatrixExample_Constructor_init_list DenseMatrixExample_Constructor_init_list.cu )
   ADD_CUSTOM_COMMAND( COMMAND DenseMatrixExample_Constructor_init_list >
                       ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/DenseMatrixExample_Constructor_init_list.out 
                       OUTPUT DenseMatrixExample_Constructor_init_list.out )

   CUDA_ADD_EXECUTABLE( DenseMatrixExample_addElement DenseMatrixExample_addElement.cu )
   ADD_CUSTOM_COMMAND( COMMAND DenseMatrixExample_addElement >
                       ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/DenseMatrixExample_addElement.out
                       OUTPUT DenseMatrixExample_addElement.out )

   CUDA_ADD_EXECUTABLE( DenseMatrixExample_setElement DenseMatrixExample_setElement.cu )
   ADD_CUSTOM_COMMAND( COMMAND DenseMatrixExample_setElement >
                       ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/DenseMatrixExample_setElement.out
                       OUTPUT DenseMatrixExample_setElement.out )

   CUDA_ADD_EXECUTABLE( DenseMatrixExample_forRows DenseMatrixExample_forRows.cu )
   ADD_CUSTOM_COMMAND( COMMAND DenseMatrixExample_forRows >
                       ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/DenseMatrixExample_forRows.out
                       OUTPUT DenseMatrixExample_forRows.out )

   CUDA_ADD_EXECUTABLE( DenseMatrixExample_rowsReduction_vectorProduct DenseMatrixExample_rowsReduction_vectorProduct.cu )
   ADD_CUSTOM_COMMAND( COMMAND DenseMatrixExample_rowsReduction_vectorProduct >
                       ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/DenseMatrixExample_rowsReduction_vectorProduct.out
                       OUTPUT DenseMatrixExample_rowsReduction_vectorProduct.out )

   CUDA_ADD_EXECUTABLE( DenseMatrixExample_rowsReduction_maxNorm DenseMatrixExample_rowsReduction_maxNorm.cu )
   ADD_CUSTOM_COMMAND( COMMAND DenseMatrixExample_rowsReduction_maxNorm >
                       ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/DenseMatrixExample_rowsReduction_maxNorm.out
                       OUTPUT DenseMatrixExample_rowsReduction_maxNorm.out )

   CUDA_ADD_EXECUTABLE( DenseMatrixViewExample_setElement DenseMatrixViewExample_setElement.cu )
   ADD_CUSTOM_COMMAND( COMMAND DenseMatrixViewExample_setElement >
                       ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/DenseMatrixViewExample_setElement.out OUTPUT
                       DenseMatrixViewExample_setElement.out )

   CUDA_ADD_EXECUTABLE( DenseMatrixViewExample_data_encapsulation DenseMatrixViewExample_data_encapsulation.cu )
   ADD_CUSTOM_COMMAND( COMMAND DenseMatrixViewExample_data_encapsulation >
                       ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/DenseMatrixViewExample_data_encapsulation.out OUTPUT
                       DenseMatrixViewExample_data_encapsulation.out )

   CUDA_ADD_EXECUTABLE( SparseMatrixExample_Constructor_init_list_2 SparseMatrixExample_Constructor_init_list_2.cu )
   ADD_CUSTOM_COMMAND( COMMAND SparseMatrixExample_Constructor_init_list_2 >
                       ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SparseMatrixExample_Constructor_init_list_2.out
                       OUTPUT SparseMatrixExample_Constructor_init_list_2.out )

   CUDA_ADD_EXECUTABLE( SparseMatrixExample_setRowCapacities SparseMatrixExample_setRowCapacities.cu )
   ADD_CUSTOM_COMMAND( COMMAND SparseMatrixExample_setRowCapacities >
                       ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SparseMatrixExample_setRowCapacities.out
                       OUTPUT SparseMatrixExample_setRowCapacities.out )

   CUDA_ADD_EXECUTABLE( SparseMatrixExample_Constructor_rowCapacities_vector SparseMatrixExample_Constructor_rowCapacities_vector.cu )
   ADD_CUSTOM_COMMAND( COMMAND SparseMatrixExample_Constructor_rowCapacities_vector >
                       ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SparseMatrixExample_Constructor_rowCapacities_vector.out
                       OUTPUT SparseMatrixExample_Constructor_rowCapacities_vector.out )

   CUDA_ADD_EXECUTABLE( SparseMatrixExample_Constructor_std_map SparseMatrixExample_Constructor_std_map.cu )
   ADD_CUSTOM_COMMAND( COMMAND SparseMatrixExample_Constructor_std_map >
                       ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SparseMatrixExample_Constructor_std_map.out
                       OUTPUT SparseMatrixExample_Constructor_std_map.out )

   CUDA_ADD_EXECUTABLE( SparseMatrixExample_setElements SparseMatrixExample_setElements.cu )
   ADD_CUSTOM_COMMAND( COMMAND SparseMatrixExample_setElements >
                       ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SparseMatrixExample_setElements.out
                       OUTPUT SparseMatrixExample_setElements.out )

   CUDA_ADD_EXECUTABLE( SparseMatrixExample_setElements_map SparseMatrixExample_setElements_map.cu )
   ADD_CUSTOM_COMMAND( COMMAND SparseMatrixExample_setElements_map >
                       ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SparseMatrixExample_setElements_map.out
                       OUTPUT SparseMatrixExample_setElements_map.out )

   CUDA_ADD_EXECUTABLE( SparseMatrixExample_setElement SparseMatrixExample_setElement.cu )
   ADD_CUSTOM_COMMAND( COMMAND SparseMatrixExample_setElement >
                       ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SparseMatrixExample_setElement.out
                       OUTPUT SparseMatrixExample_setElement.out )

   CUDA_ADD_EXECUTABLE( SparseMatrixExample_addElement SparseMatrixExample_addElement.cu )
   ADD_CUSTOM_COMMAND( COMMAND SparseMatrixExample_addElement >
                       ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SparseMatrixExample_addElement.out
                       OUTPUT SparseMatrixExample_addElement.out )

   CUDA_ADD_EXECUTABLE( SparseMatrixExample_forRows SparseMatrixExample_forRows.cu )
   ADD_CUSTOM_COMMAND( COMMAND SparseMatrixExample_forRows >
                       ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SparseMatrixExample_forRows.out
                       OUTPUT SparseMatrixExample_forRows.out )

   CUDA_ADD_EXECUTABLE( SparseMatrixExample_rowsReduction_vectorProduct SparseMatrixExample_rowsReduction_vectorProduct.cu )
   ADD_CUSTOM_COMMAND( COMMAND SparseMatrixExample_rowsReduction_vectorProduct >
                       ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SparseMatrixExample_rowsReduction_vectorProduct.out
                       OUTPUT SparseMatrixExample_rowsReduction_vectorProduct.out )

   CUDA_ADD_EXECUTABLE( SparseMatrixViewExample_setElement SparseMatrixViewExample_setElement.cu )
   ADD_CUSTOM_COMMAND( COMMAND SparseMatrixViewExample_setElement >
                       ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SparseMatrixViewExample_setElement.out
                       OUTPUT SparseMatrixViewExample_setElement.out )

   CUDA_ADD_EXECUTABLE( SymmetricSparseMatrixExample SymmetricSparseMatrixExample.cu )
   ADD_CUSTOM_COMMAND( COMMAND SymmetricSparseMatrixExample >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SymmetricSparseMatrixExample.out
                        OUTPUT SymmetricSparseMatrixExample.out )
   ####
   # THe following examples/benchmarks run for very long time
   CUDA_ADD_EXECUTABLE( DenseMatrixSetup_Benchmark_cuda DenseMatrixSetup_Benchmark.cu )
   CUDA_ADD_EXECUTABLE( SparseMatrixSetup_Benchmark_cuda SparseMatrixSetup_Benchmark.cu )
   CUDA_ADD_EXECUTABLE( MultidiagonalMatrixSetup_Benchmark_cuda MultidiagonalMatrixSetup_Benchmark.cu )

ELSE()

   ####
   # THe following examples/benchmarks run for very long time
   ADD_EXECUTABLE( DenseMatrixSetup_Benchmark DenseMatrixSetup_Benchmark.cpp )
   ADD_EXECUTABLE( SparseMatrixSetup_Benchmark SparseMatrixSetup_Benchmark.cpp )
   ADD_EXECUTABLE( MultidiagonalMatrixSetup_Benchmark MultidiagonalMatrixSetup_Benchmark.cpp )
ENDIF()

IF( BUILD_CUDA )
ADD_CUSTOM_TARGET( TutorialsMatricesCuda ALL DEPENDS
   DenseMatrixExample_Constructor_init_list.out
   DenseMatrixExample_addElement.out
   DenseMatrixExample_setElement.out
   DenseMatrixExample_forRows.out
   DenseMatrixExample_rowsReduction_vectorProduct.out
   DenseMatrixExample_rowsReduction_maxNorm.out
   DenseMatrixViewExample_setElement.out
   DenseMatrixViewExample_data_encapsulation.out
   SparseMatrixExample_Constructor_init_list_2.out
   SparseMatrixExample_setRowCapacities.out
   SparseMatrixExample_Constructor_std_map.out
   SparseMatrixExample_setElements.out
   SparseMatrixExample_setElements_map.out
   SparseMatrixExample_setElement.out
   SparseMatrixExample_forRows.out
   SparseMatrixExample_rowsReduction_vectorProduct.out
   SparseMatrixViewExample_setElement.out
   SymmetricSparseMatrixExample.out
 )
ELSE()
ADD_CUSTOM_TARGET( TutorialsMatrices ALL DEPENDS
)
ENDIF()
#
#ADD_CUSTOM_TARGET( TutorialsPointers ALL DEPENDS
#   UniquePointerHostExample.out
#)