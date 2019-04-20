IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( ArrayExampleCuda ArrayExample.cu )
   ADD_CUSTOM_COMMAND( COMMAND ArrayExampleCuda > ArrayExampleCuda.out OUTPUT ArrayExampleCuda.out )
ELSE()
   ADD_EXECUTABLE( ArrayExample ArrayExample.cpp )
   ADD_CUSTOM_COMMAND( COMMAND ArrayExample > ArrayExample.out OUTPUT ArrayExample.out )
ENDIF()

IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( ArrayViewExampleCuda ArrayViewExample.cu )
   ADD_CUSTOM_COMMAND( COMMAND ArrayViewExampleCuda > ArrayViewExampleCuda.out OUTPUT ArrayViewExampleCuda.out )
ELSE()
   ADD_EXECUTABLE( ArrayViewExample ArrayViewExample.cpp )
   ADD_CUSTOM_COMMAND( COMMAND ArrayViewExample > ArrayViewExample.out OUTPUT ArrayViewExample.out )
ENDIF()


IF( BUILD_CUDA )
ADD_CUSTOM_TARGET( RunContainersExamples-cuda ALL DEPENDS
   ArrayExampleCuda.out
   ArrayViewExampleCuda.out )
ELSE()
ADD_CUSTOM_TARGET( RunContainersExamples ALL DEPENDS
   ArrayExample.out
   ArrayViewExample.out )
ENDIF()
