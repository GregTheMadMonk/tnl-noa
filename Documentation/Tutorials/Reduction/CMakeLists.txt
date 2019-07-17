IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( SumExample SumExample.cu )
   ADD_CUSTOM_COMMAND( COMMAND SumExample > SumExample.out OUTPUT SumExample.out )
   CUDA_ADD_EXECUTABLE( ProductExample ProductExample.cu )
   ADD_CUSTOM_COMMAND( COMMAND ProductExample > ProductExample.out OUTPUT ProductExample.out )
   CUDA_ADD_EXECUTABLE( ScalarProductExample ScalarProductExample.cu )
   ADD_CUSTOM_COMMAND( COMMAND ScalarProductExample > ScalarProductExample.out OUTPUT ScalarProductExample.out )
   CUDA_ADD_EXECUTABLE( MaximumNormExample MaximumNormExample.cu )
   ADD_CUSTOM_COMMAND( COMMAND MaximumNormExample > MaximumNormExample.out OUTPUT MaximumNormExample.out )
   CUDA_ADD_EXECUTABLE( ComparisonExample ComparisonExample.cu )
   ADD_CUSTOM_COMMAND( COMMAND ComparisonExample > ComparisonExample.out OUTPUT ComparisonExample.out )
#   CUDA_ADD_EXECUTABLE( UpdateAndResidueExample UpdateAndResidueExample.cu )
#   ADD_CUSTOM_COMMAND( COMMAND UpdateAndResidueExample > UpdateAndResidueExample.out OUTPUT UpdateAndResidueExample.out )
ENDIF()

ADD_EXECUTABLE( UpdateAndResidueExample UpdateAndResidueExample.cpp )
ADD_CUSTOM_COMMAND( COMMAND UpdateAndResidueExample > UpdateAndResidueExample.out OUTPUT UpdateAndResidueExample.out )


IF( BUILD_CUDA )
ADD_CUSTOM_TARGET( TutorialsReduction-cuda ALL DEPENDS
   SumExample.out
   ProductExample.out
   ScalarProductExample.out
   MaximumNormExample.out 
   ComparisonExample.out
   UpdateAndResidueExample.out )
ENDIF()
