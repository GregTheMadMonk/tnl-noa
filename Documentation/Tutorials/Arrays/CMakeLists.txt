IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( ArrayAllocation ArrayAllocation.cu )
   ADD_CUSTOM_COMMAND( COMMAND ArrayAllocation > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/ArrayAllocation.out OUTPUT ArrayAllocation.out )
   CUDA_ADD_EXECUTABLE( ArrayBinding-1 ArrayBinding-1.cu )
   ADD_CUSTOM_COMMAND( COMMAND ArrayBinding-1 > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/ArrayBinding-1.out OUTPUT ArrayBinding-1.out )
   CUDA_ADD_EXECUTABLE( ArrayBinding-2 ArrayBinding-2.cu )
   ADD_CUSTOM_COMMAND( COMMAND ArrayBinding-2 > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/ArrayBinding-2.out OUTPUT ArrayBinding-2.out )
   CUDA_ADD_EXECUTABLE( ArrayBinding-3 ArrayBinding-3.cu )
   ADD_CUSTOM_COMMAND( COMMAND ArrayBinding-3 > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/ArrayBinding-3.out OUTPUT ArrayBinding-3.out )
   CUDA_ADD_EXECUTABLE( ArrayIO ArrayIO.cu )
   ADD_CUSTOM_COMMAND( COMMAND ArrayIO > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/ArrayIO.out OUTPUT ArrayIO.out )
   CUDA_ADD_EXECUTABLE( ArrayView-1 ArrayView-1.cu )
   ADD_CUSTOM_COMMAND( COMMAND ArrayView-1 > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/ArrayView-1.out OUTPUT ArrayView-1.out )
   CUDA_ADD_EXECUTABLE( ArrayView-2 ArrayView-2.cu )
   ADD_CUSTOM_COMMAND( COMMAND ArrayView-2 > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/ArrayView-2.out OUTPUT ArrayView-2.out )
   CUDA_ADD_EXECUTABLE( ArrayViewEvaluate ArrayViewEvaluate.cu )
   ADD_CUSTOM_COMMAND( COMMAND ArrayViewEvaluate > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/ArrayViewEvaluate.out OUTPUT ArrayViewEvaluate.out )
   CUDA_ADD_EXECUTABLE( ContainsValue ContainsValue.cu )
   ADD_CUSTOM_COMMAND( COMMAND ContainsValue > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/ContainsValue.out OUTPUT ContainsValue.out )
   CUDA_ADD_EXECUTABLE( ElementsAccessing-1 ElementsAccessing-1.cu )
   ADD_CUSTOM_COMMAND( COMMAND ElementsAccessing-1 > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/ElementsAccessing-1.out OUTPUT ElementsAccessing-1.out )
   CUDA_ADD_EXECUTABLE( ElementsAccessing-2 ElementsAccessing-2.cu )
   ADD_CUSTOM_COMMAND( COMMAND ElementsAccessing-2 > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/ElementsAccessing-2.out OUTPUT ElementsAccessing-2.out )
ENDIF()

IF( BUILD_CUDA )
ADD_CUSTOM_TARGET( TutorialsArrays-cuda ALL DEPENDS
   ArrayAllocation.out
   ArrayBinding-1.out
   ArrayBinding-2.out
   ArrayBinding-3.out
   ArrayIO.out
   ArrayView-1.out
   ArrayView-2.out
   ContainsValue.out
   ElementsAccessing-1.out
   ElementsAccessing-2.out
   ArrayViewEvaluate.out )
ENDIF()
