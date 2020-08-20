IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( DenseMatrixExample_Constructor_init_list DenseMatrixExample_Constructor_init_list.cu )
   ADD_CUSTOM_COMMAND( COMMAND DenseMatrixExample_Constructor_init_list > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/DenseMatrixExample_Constructor_init_list.out OUTPUT DenseMatrixExample_Constructor_init_list.out )

   CUDA_ADD_EXECUTABLE( DenseMatrixExample_addElement DenseMatrixExample_addElement.cu )
   ADD_CUSTOM_COMMAND( COMMAND DenseMatrixExample_addElement > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/DenseMatrixExample_addElement.out OUTPUT DenseMatrixExample_addElement.out )

   CUDA_ADD_EXECUTABLE( DenseMatrixExample_setElement DenseMatrixExample_setElement.cu )
   ADD_CUSTOM_COMMAND( COMMAND DenseMatrixExample_setElement > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/DenseMatrixExample_setElement.out OUTPUT DenseMatrixExample_setElement.out )

   CUDA_ADD_EXECUTABLE( DenseMatrixExample_forRows DenseMatrixExample_forRows.cu )
   ADD_CUSTOM_COMMAND( COMMAND DenseMatrixExample_forRows > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/DenseMatrixExample_forRows.out OUTPUT DenseMatrixExample_forRows.out )
   
   CUDA_ADD_EXECUTABLE( DenseMatrixExample_rowsReduction_vectorProduct DenseMatrixExample_rowsReduction_vectorProduct.cu )
   ADD_CUSTOM_COMMAND( COMMAND DenseMatrixExample_rowsReduction_vectorProduct > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/DenseMatrixExample_rowsReduction_vectorProduct.out OUTPUT DenseMatrixExample_rowsReduction_vectorProduct.out )
   
   CUDA_ADD_EXECUTABLE( DenseMatrixExample_rowsReduction_maxNorm DenseMatrixExample_rowsReduction_maxNorm.cu )
   ADD_CUSTOM_COMMAND( COMMAND DenseMatrixExample_rowsReduction_maxNorm > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/DenseMatrixExample_rowsReduction_maxNorm.out OUTPUT DenseMatrixExample_rowsReduction_maxNorm.out )

   CUDA_ADD_EXECUTABLE( DenseMatrixViewExample_setElement DenseMatrixViewExample_setElement.cu )
   ADD_CUSTOM_COMMAND( COMMAND DenseMatrixViewExample_setElement > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/DenseMatrixViewExample_setElement.out OUTPUT DenseMatrixViewExample_setElement.out )

ELSE()
#   ADD_EXECUTABLE( UniquePointerExample UniquePointerExample.cpp )
#   ADD_CUSTOM_COMMAND( COMMAND UniquePointerExample > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/UniquePointerExample.out OUTPUT UniquePointerExample.out )
ENDIF()
#
#ADD_EXECUTABLE( UniquePointerHostExample UniquePointerHostExample.cpp )
#ADD_CUSTOM_COMMAND( COMMAND UniquePointerHostExample > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/UniquePointerHostExample.out OUTPUT UniquePointerHostExample.out )
#
#
IF( BUILD_CUDA )
ADD_CUSTOM_TARGET( TutorialsMatricesCuda ALL DEPENDS
   DenseMatrixExample_Constructor_init_list.out
   DenseMatrixExample_addElement.out
   DenseMatrixExample_setElement.out
   DenseMatrixExample_forRows.out
   DenseMatrixExample_rowsReduction_vectorProduct.out
   DenseMatrixExample_rowsReduction_maxNorm.out
   DenseMatrixViewExample_setElement.out
 )
ENDIF()
#
#ADD_CUSTOM_TARGET( TutorialsPointers ALL DEPENDS
#   UniquePointerHostExample.out
#)