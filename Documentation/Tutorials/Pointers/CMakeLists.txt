IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( UniquePointerExample UniquePointerExample.cu )
   ADD_CUSTOM_COMMAND( COMMAND UniquePointerExample > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/UniquePointerExample.out OUTPUT UniquePointerExample.out )
   CUDA_ADD_EXECUTABLE( SharedPointerExample SharedPointerExample.cu )
   ADD_CUSTOM_COMMAND( COMMAND SharedPointerExample > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SharedPointerExample.out OUTPUT SharedPointerExample.out )
ELSE()
   ADD_EXECUTABLE( UniquePointerExample UniquePointerExample.cpp )
   ADD_CUSTOM_COMMAND( COMMAND UniquePointerExample > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/UniquePointerExample.out OUTPUT UniquePointerExample.out )
ENDIF()

ADD_EXECUTABLE( UniquePointerHostExample UniquePointerHostExample.cpp )
ADD_CUSTOM_COMMAND( COMMAND UniquePointerHostExample > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/UniquePointerHostExample.out OUTPUT UniquePointerHostExample.out )


IF( BUILD_CUDA )
ADD_CUSTOM_TARGET( TutorialsPointersCuda ALL DEPENDS
   UniquePointerExample.out
   SharedPointerExample.out )
ENDIF()

ADD_CUSTOM_TARGET( TutorialsPointers ALL DEPENDS
   UniquePointerHostExample.out
)