ADD_EXECUTABLE( LambdaMatrixExample_Constructor LambdaMatrixExample_Constructor.cpp )
ADD_CUSTOM_COMMAND( COMMAND LambdaMatrixExample_Constructor >
                     ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/LambdaMatrixExample_Constructor.out
                    OUTPUT LambdaMatrixExample_Constructor.out )

ADD_EXECUTABLE( LambdaMatrixExample_getCompressedRowLengths LambdaMatrixExample_getCompressedRowLengths.cpp )
ADD_CUSTOM_COMMAND( COMMAND LambdaMatrixExample_getCompressedRowLengths >
                     ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/LambdaMatrixExample_getCompressedRowLengths.out
                    OUTPUT LambdaMatrixExample_getCompressedRowLengths.out )

ADD_EXECUTABLE( LambdaMatrixExample_getNonzeroElementsCount LambdaMatrixExample_getNonzeroElementsCount.cpp )
ADD_CUSTOM_COMMAND( COMMAND LambdaMatrixExample_getNonzeroElementsCount >
                     ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/LambdaMatrixExample_getNonzeroElementsCount.out
                    OUTPUT LambdaMatrixExample_getNonzeroElementsCount.out )

IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( LambdaMatrixExample_Laplace_cuda LambdaMatrixExample_Laplace.cu )
   ADD_CUSTOM_COMMAND( COMMAND LambdaMatrixExample_Laplace_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/LambdaMatrixExample_Laplace.out
                     OUTPUT LambdaMatrixExample_Laplace.out )

   CUDA_ADD_EXECUTABLE( LambdaMatrixExample_Laplace_2_cuda LambdaMatrixExample_Laplace_2.cu )
   ADD_CUSTOM_COMMAND( COMMAND LambdaMatrixExample_Laplace_2_cuda >
                       ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/LambdaMatrixExample_Laplace_2.out
                       OUTPUT LambdaMatrixExample_Laplace_2.out )

                     CUDA_ADD_EXECUTABLE( LambdaMatrixExample_rowsReduction_cuda LambdaMatrixExample_rowsReduction.cu )
   ADD_CUSTOM_COMMAND( COMMAND LambdaMatrixExample_rowsReduction_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/LambdaMatrixExample_rowsReduction.out
                       OUTPUT LambdaMatrixExample_rowsReduction.out )

   CUDA_ADD_EXECUTABLE( LambdaMatrixExample_allRowsReduction_cuda LambdaMatrixExample_allRowsReduction.cu )
   ADD_CUSTOM_COMMAND( COMMAND LambdaMatrixExample_allRowsReduction_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/LambdaMatrixExample_allRowsReduction.out
                       OUTPUT LambdaMatrixExample_allRowsReduction.out )

   CUDA_ADD_EXECUTABLE( LambdaMatrixExample_forElements_cuda LambdaMatrixExample_forElements.cu )
   ADD_CUSTOM_COMMAND( COMMAND LambdaMatrixExample_forElements_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/LambdaMatrixExample_forElements.out
                       OUTPUT LambdaMatrixExample_forElements.out )

   CUDA_ADD_EXECUTABLE( LambdaMatrixExample_forEachElement_cuda LambdaMatrixExample_forEachElement.cu )
   ADD_CUSTOM_COMMAND( COMMAND LambdaMatrixExample_forEachElement_cuda >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/LambdaMatrixExample_forEachElement.out
                       OUTPUT LambdaMatrixExample_forEachElement.out )

ELSE()
   ADD_EXECUTABLE( LambdaMatrixExample_Laplace LambdaMatrixExample_Laplace.cpp )
   ADD_CUSTOM_COMMAND( COMMAND LambdaMatrixExample_Laplace >
                       ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/LambdaMatrixExample_Laplace.out
                       OUTPUT LambdaMatrixExample_Laplace.out )

   ADD_EXECUTABLE( LambdaMatrixExample_Laplace_2 LambdaMatrixExample_Laplace_2.cpp )
   ADD_CUSTOM_COMMAND( COMMAND LambdaMatrixExample_Laplace_2 >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/LambdaMatrixExample_Laplace_2.out
                        OUTPUT LambdaMatrixExample_Laplace_2.out )

   ADD_EXECUTABLE( LambdaMatrixExample_rowsReduction LambdaMatrixExample_rowsReduction.cpp )
   ADD_CUSTOM_COMMAND( COMMAND LambdaMatrixExample_rowsReduction >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/LambdaMatrixExample_rowsReduction.out
                       OUTPUT LambdaMatrixExample_rowsReduction.out )

   ADD_EXECUTABLE( LambdaMatrixExample_allRowsReduction LambdaMatrixExample_allRowsReduction.cpp )
   ADD_CUSTOM_COMMAND( COMMAND LambdaMatrixExample_allRowsReduction >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/LambdaMatrixExample_allRowsReduction.out
                       OUTPUT LambdaMatrixExample_allRowsReduction.out )

   ADD_EXECUTABLE( LambdaMatrixExample_forElements LambdaMatrixExample_forElements.cpp )
   ADD_CUSTOM_COMMAND( COMMAND LambdaMatrixExample_forElements >
                       ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/LambdaMatrixExample_forElements.out
                       OUTPUT LambdaMatrixExample_forElements.out )

   ADD_EXECUTABLE( LambdaMatrixExample_forEachElement LambdaMatrixExample_forEachElement.cpp )
   ADD_CUSTOM_COMMAND( COMMAND LambdaMatrixExample_forEachElement >
                        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/LambdaMatrixExample_forEachElement.out
                       OUTPUT LambdaMatrixExample_forEachElement.out )
ENDIF()

ADD_CUSTOM_TARGET( RunLambdaMatricesExamples ALL DEPENDS
   LambdaMatrixExample_Constructor.out
   LambdaMatrixExample_Laplace.out
   LambdaMatrixExample_Laplace_2.out
   LambdaMatrixExample_getCompressedRowLengths.out
   LambdaMatrixExample_getNonzeroElementsCount.out
   LambdaMatrixExample_rowsReduction.out
   LambdaMatrixExample_allRowsReduction.out
   LambdaMatrixExample_forElements.out
   LambdaMatrixExample_forEachElement.out
)

