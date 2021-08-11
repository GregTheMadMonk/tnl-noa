IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( SortingExampleCuda SortingExample.cu)
   ADD_CUSTOM_COMMAND( COMMAND SortingExampleCuda > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SortingExample.out OUTPUT SortingExample.out )

   CUDA_ADD_EXECUTABLE( SortingExample2Cuda SortingExample2.cu)
   ADD_CUSTOM_COMMAND( COMMAND SortingExample2Cuda > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SortingExample2.out OUTPUT SortingExample2.out )

   CUDA_ADD_EXECUTABLE( SortingExample3Cuda SortingExample3.cu)
   ADD_CUSTOM_COMMAND( COMMAND SortingExample3Cuda > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SortingExample3.out OUTPUT SortingExample3.out )

   CUDA_ADD_EXECUTABLE(ParallelForExampleCuda ParallelForExample.cu)
   ADD_CUSTOM_COMMAND( COMMAND ParallelForExampleCuda > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/ParallelForExample.out OUTPUT ParallelForExample.out )

   CUDA_ADD_EXECUTABLE(reduceArrayExampleCuda reduceArrayExample.cu)
   ADD_CUSTOM_COMMAND( COMMAND reduceArrayExampleCuda > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/reduceArrayExample.out OUTPUT reduceArrayExample.out )

   CUDA_ADD_EXECUTABLE(reduceWithArgumentArrayExampleCuda reduceWithArgumentArrayExample.cu)
   ADD_CUSTOM_COMMAND( COMMAND reduceWithArgumentArrayExampleCuda > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/reduceWithArgumentArrayExample.out OUTPUT reduceWithArgumentArrayExample.out )
ELSE()
   ADD_EXECUTABLE( SortingExample SortingExample.cpp)
   ADD_CUSTOM_COMMAND( COMMAND SortingExample > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SortingExample.out OUTPUT SortingExample.out )

   ADD_EXECUTABLE( SortingExample2 SortingExample2.cpp)
   ADD_CUSTOM_COMMAND( COMMAND SortingExample2 > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SortingExample2.out OUTPUT SortingExample2.out )

   ADD_EXECUTABLE( SortingExample3 SortingExample3.cpp)
   ADD_CUSTOM_COMMAND( COMMAND SortingExample3 > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/SortingExample3.out OUTPUT SortingExample3.out )

   ADD_EXECUTABLE(ParallelForExample ParallelForExample.cpp)
   ADD_CUSTOM_COMMAND( COMMAND ParallelForExample > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/ParallelForExample.out OUTPUT ParallelForExample.out )

   ADD_EXECUTABLE(reduceArrayExample reduceArrayExample.cpp)
   ADD_CUSTOM_COMMAND( COMMAND reduceArrayExample > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/reduceArrayExample.out OUTPUT reduceArrayExample.out )

   ADD_EXECUTABLE(reduceWithArgumentArrayExample reduceWithArgumentArrayExample.cpp)
   ADD_CUSTOM_COMMAND( COMMAND reduceWithArgumentArrayExample > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/reduceWithArgumentArrayExample.out OUTPUT reduceWithArgumentArrayExample.out )
ENDIF()

ADD_EXECUTABLE(staticForExample staticForExample.cpp)
ADD_CUSTOM_COMMAND( COMMAND staticForExample > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/staticForExample.out OUTPUT staticForExample.out )

ADD_EXECUTABLE(unrolledForExample unrolledForExample.cpp)
ADD_CUSTOM_COMMAND( COMMAND unrolledForExample > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/unrolledForExample.out OUTPUT unrolledForExample.out )

ADD_CUSTOM_TARGET( RunAlgorithmsExamples ALL DEPENDS
   SortingExample.out
   SortingExample2.out
   SortingExample3.out
   ParallelForExample.out
   reduceArrayExample.out
   reduceWithArgumentArrayExample.out
   unrolledForExample.out
   staticForExample.out
)
