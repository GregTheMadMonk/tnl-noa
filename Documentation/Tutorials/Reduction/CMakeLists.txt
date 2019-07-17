IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( SumExample SumExample.cu )
   ADD_CUSTOM_COMMAND( COMMAND SumExample > SumExample.out OUTPUT SumExample.out )
   CUDA_ADD_EXECUTABLE( ProductExample ProductExample.cu )
   ADD_CUSTOM_COMMAND( COMMAND ProductExample > ProductExample.out OUTPUT ProductExample.out )
   CUDA_ADD_EXECUTABLE( ScalarProductExample ScalarProductExample.cu )
   ADD_CUSTOM_COMMAND( COMMAND ScalarProductExample > ScalarProductExample.out OUTPUT ScalarProductExample.out )
   CUDA_ADD_EXECUTABLE( MaximumNormExample MaximumNormExample.cu )
   ADD_CUSTOM_COMMAND( COMMAND MaximumNormExample > MaximumNormExample.out OUTPUT MaximumNormExample.out )
ENDIF()

IF( BUILD_CUDA )
ADD_CUSTOM_TARGET( TutorialsReduction-cuda ALL DEPENDS
   SumExample.out
   ProductExample.out
   ScalarProductExample.out
   MaximumNormExample.out )
ENDIF()
