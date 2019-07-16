IF( BUILD_CUDA )
#   CUDA_ADD_EXECUTABLE( ArrayAllocation ArrayAllocation.cu )
#   ADD_CUSTOM_COMMAND( COMMAND ArrayAllocation > ArrayAllocation.out OUTPUT ArrayAllocation.out )
ENDIF()

IF( BUILD_CUDA )
ADD_CUSTOM_TARGET( TutorialsReduction-cuda ALL DEPENDS
 )
ENDIF()
