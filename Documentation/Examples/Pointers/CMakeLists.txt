IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE(UniquePointerExampleCuda UniquePointerExample.cu)
   ADD_CUSTOM_COMMAND( COMMAND UniquePointerExampleCuda > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/UniquePointerExample.out OUTPUT UniquePointerExample.out )
   CUDA_ADD_EXECUTABLE(SharedPointerExampleCuda SharedPointerExample.cu)
   ADD_CUSTOM_COMMAND( COMMAND SharedPointerExampleCuda > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SharedPointerExample.out OUTPUT SharedPointerExample.out )
   CUDA_ADD_EXECUTABLE(DevicePointerExampleCuda DevicePointerExample.cu)
   ADD_CUSTOM_COMMAND( COMMAND DevicePointerExampleCuda > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/DevicePointerExample.out OUTPUT DevicePointerExample.out )

ADD_CUSTOM_TARGET( RunPointersExamples ALL DEPENDS
   UniquePointerExample.out
   SharedPointerExample.out
   DevicePointerExample.out
 )

ENDIF()
