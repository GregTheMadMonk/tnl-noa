IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( ArrayExampleCuda ArrayExample.cu )
   ADD_CUSTOM_COMMAND( COMMAND ArrayExampleCuda > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/ArrayExample.out OUTPUT ArrayExample.out )
ELSE()
   ADD_EXECUTABLE( ArrayExample ArrayExample.cpp )
   ADD_CUSTOM_COMMAND( COMMAND ArrayExample > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/ArrayExample.out OUTPUT ArrayExample.out )
ENDIF()

IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( ArrayViewExampleCuda ArrayViewExample.cu )
   ADD_CUSTOM_COMMAND( COMMAND ArrayViewExampleCuda > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/ArrayViewExample.out OUTPUT ArrayViewExample.out )
ELSE()
   ADD_EXECUTABLE( ArrayViewExample ArrayViewExample.cpp )
   ADD_CUSTOM_COMMAND( COMMAND ArrayViewExample > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/ArrayViewExample.out OUTPUT ArrayViewExample.out )
ENDIF()

ADD_EXECUTABLE( VectorExample VectorExample.cpp )


IF( BUILD_CUDA )
ADD_CUSTOM_TARGET( RunContainersExamples-cuda ALL DEPENDS
   ArrayExample.out
   ArrayViewExample.out )
ELSE()
ADD_CUSTOM_TARGET( RunContainersExamples ALL DEPENDS
   ArrayExample.out
   ArrayViewExample.out )
ENDIF()
