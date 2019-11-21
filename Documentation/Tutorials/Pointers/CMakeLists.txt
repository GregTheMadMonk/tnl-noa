IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( UniquePointerExample UniquePointerExample.cu )
   ADD_CUSTOM_COMMAND( COMMAND UniquePointerExample > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/UniquePointerExample.out OUTPUT UniquePointerExample.out )
ELSE()
   ADD_EXECUTABLE( UniquePointerExample UniquePointerExample.cu )
   ADD_CUSTOM_COMMAND( COMMAND UniquePointerExample > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/UniquePointerExample.out OUTPUT UniquePointerExample.out )
ENDIF()

IF( BUILD_CUDA )
ADD_CUSTOM_TARGET( TutorialsPointers ALL DEPENDS
   UniquePointerExample.out )
ENDIF()
