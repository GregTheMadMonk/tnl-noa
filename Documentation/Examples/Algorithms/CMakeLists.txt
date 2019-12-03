IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE(ParallelForExampleCuda ParallelForExample.cu)
   ADD_CUSTOM_COMMAND( COMMAND ParallelForExampleCuda > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/ParallelForExample.out OUTPUT ParallelForExample.out )
ELSE()
   ADD_EXECUTABLE(ParallelForExample ParallelForExample.cpp)
   ADD_CUSTOM_COMMAND( COMMAND ParallelForExample > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/ParallelForExample.out OUTPUT ParallelForExample.out )
ENDIF()

IF( BUILD_CUDA )
ADD_CUSTOM_TARGET( RunAlgorithmsExamples-cuda ALL DEPENDS
   ParallelForExample.out
 )
ELSE()
ADD_CUSTOM_TARGET( RunAlgorithmsExamples ALL DEPENDS
   ParallelForExample.out
 )
ENDIF()