IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( ParallelForExample ParallelForExample.cu )
   ADD_CUSTOM_COMMAND( COMMAND ParallelForExample > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/ParallelForExample.out OUTPUT ParallelForExample.out )
ENDIF()

IF( BUILD_CUDA )
ADD_CUSTOM_TARGET( ForLoops-cuda ALL DEPENDS
   ParallelForExample.out )
ENDIF()
