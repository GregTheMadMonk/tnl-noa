IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE(ParallelForExampleCuda ParallelForExample.cu)
   ADD_CUSTOM_COMMAND( COMMAND ParallelForExampleCuda > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/ParallelForExample.out OUTPUT ParallelForExample.out )
ELSE()
   ADD_EXECUTABLE(ParallelForExample ParallelForExample.cpp)
   ADD_CUSTOM_COMMAND( COMMAND ParallelForExample > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/ParallelForExample.out OUTPUT ParallelForExample.out )
ENDIF()

ADD_EXECUTABLE(staticForExample staticForExample.cpp)
ADD_CUSTOM_COMMAND( COMMAND staticForExample > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/staticForExample.out OUTPUT staticForExample.out )

ADD_EXECUTABLE(unrolledForExample unrolledForExample.cpp)
ADD_CUSTOM_COMMAND( COMMAND unrolledForExample > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/unrolledForExample.out OUTPUT unrolledForExample.out )

ADD_CUSTOM_TARGET( RunAlgorithmsExamples ALL DEPENDS
   ParallelForExample.out
   unrolledForExample.out
   staticForExample.out
)
