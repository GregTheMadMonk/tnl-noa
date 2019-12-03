IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( ParallelForExample ParallelForExample.cu )
   ADD_CUSTOM_COMMAND( COMMAND ParallelForExample > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/ParallelForExample.out OUTPUT ParallelForExample.out )
   CUDA_ADD_EXECUTABLE( ParallelForExample-2D ParallelForExample-2D.cu )
   CUDA_ADD_EXECUTABLE( ParallelForExample-3D ParallelForExample-3D.cu )
ELSE()
   ADD_EXECUTABLE( ParallelForExample-2D ParallelForExample-2D.cpp )
   ADD_EXECUTABLE( ParallelForExample-3D ParallelForExample-3D.cpp )
ENDIF()

ADD_EXECUTABLE( StaticForExample StaticForExample.cpp )
ADD_CUSTOM_COMMAND( COMMAND StaticForExample > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticForExample.out OUTPUT StaticForExample.out )

ADD_EXECUTABLE( TemplateStaticForExample TemplateStaticForExample.cpp )
ADD_CUSTOM_COMMAND( COMMAND TemplateStaticForExample > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/TemplateStaticForExample.out OUTPUT TemplateStaticForExample.out )

IF( BUILD_CUDA )
ADD_CUSTOM_TARGET( ForLoops-cuda ALL DEPENDS
   ParallelForExample.out
   StaticForExample.out
   TemplateStaticForExample.out )
ENDIF()
