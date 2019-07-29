IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( Expressions Expressions.cu )
   ADD_CUSTOM_COMMAND( COMMAND Expressions > Expressions.out OUTPUT Expressions.out )
   CUDA_ADD_EXECUTABLE( Reduction Reduction.cu )
   ADD_CUSTOM_COMMAND( COMMAND Reduction > Reduction.out OUTPUT Reduction.out )
ENDIF()

IF( BUILD_CUDA )
ADD_CUSTOM_TARGET( TutorialsVectors-cuda ALL DEPENDS
   Expressions.out
   Reduction.out )
ENDIF()
