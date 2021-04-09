ADD_SUBDIRECTORY( Algorithms )
ADD_SUBDIRECTORY( Containers )
ADD_SUBDIRECTORY( Pointers )
ADD_SUBDIRECTORY( Matrices )

ADD_EXECUTABLE( FileExample FileExample.cpp )
ADD_CUSTOM_COMMAND( COMMAND FileExample > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/FileExample.out OUTPUT FileExample.out )

IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE(FileExampleCuda FileExampleCuda.cu)
   ADD_CUSTOM_COMMAND( COMMAND FileExampleCuda > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/FileExampleCuda.out OUTPUT FileExampleCuda.out )
ENDIF()

ADD_EXECUTABLE( FileExampleSaveAndLoad FileExampleSaveAndLoad.cpp )
ADD_CUSTOM_COMMAND( COMMAND FileExampleSaveAndLoad > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/FileExampleSaveAndLoad.out OUTPUT FileExampleSaveAndLoad.out )

ADD_EXECUTABLE( FileNameExample FileNameExample.cpp )
ADD_CUSTOM_COMMAND( COMMAND FileNameExample > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/FileNameExample.out OUTPUT FileNameExample.out )

ADD_EXECUTABLE( FileNameExampleDistributedSystemNodeCoordinates FileNameExampleDistributedSystemNodeCoordinates.cpp )
ADD_CUSTOM_COMMAND( COMMAND FileNameExampleDistributedSystemNodeCoordinates > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/FileNameExampleDistributedSystemNodeCoordinates.out OUTPUT FileNameExampleDistributedSystemNodeCoordinates.out )


ADD_EXECUTABLE( FileNameExampleDistributedSystemNodeId FileNameExampleDistributedSystemNodeId.cpp )
ADD_CUSTOM_COMMAND( COMMAND FileNameExampleDistributedSystemNodeId > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/FileNameExampleDistributedSystemNodeId.out OUTPUT FileNameExampleDistributedSystemNodeId.out )

ADD_EXECUTABLE( ObjectExample_getType ObjectExample_getType.cpp )
ADD_CUSTOM_COMMAND( COMMAND ObjectExample_getType > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/ObjectExample_getType.out OUTPUT ObjectExample_getType.out )

ADD_EXECUTABLE( ParameterContainerExample ParameterContainerExample.cpp )
ADD_EXECUTABLE( ConfigDescriptionExample ConfigDescriptionExample.cpp )
ADD_EXECUTABLE( LoggerExample LoggerExample.cpp )
ADD_EXECUTABLE( MathExample MathExample.cpp )

ADD_EXECUTABLE( ParseObjectTypeExample ParseObjectTypeExample.cpp )
ADD_CUSTOM_COMMAND( COMMAND ParseObjectTypeExample > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/ParseObjectTypeExample.out OUTPUT ParseObjectTypeExample.out )

ADD_EXECUTABLE( StringExample StringExample.cpp )
ADD_CUSTOM_COMMAND( COMMAND StringExample > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StringExample.out OUTPUT StringExample.out )

ADD_EXECUTABLE( StringExampleGetAllocatedSize StringExampleGetAllocatedSize.cpp )
ADD_CUSTOM_COMMAND( COMMAND StringExampleGetAllocatedSize > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StringExampleGetAllocatedSize.out OUTPUT StringExampleGetAllocatedSize.out )

ADD_EXECUTABLE( StringExampleReplace StringExampleReplace.cpp )
ADD_CUSTOM_COMMAND( COMMAND StringExampleReplace > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StringExampleReplace.out OUTPUT StringExampleReplace.out )

ADD_EXECUTABLE( StringExampleSetSize StringExampleSetSize.cpp )
ADD_CUSTOM_COMMAND( COMMAND StringExampleSetSize > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StringExampleSetSize.out OUTPUT StringExampleSetSize.out )

ADD_EXECUTABLE( StringExampleSplit StringExampleSplit.cpp )
ADD_CUSTOM_COMMAND( COMMAND StringExampleSplit > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StringExampleSplit.out OUTPUT StringExampleSplit.out )

ADD_EXECUTABLE( StringExampleStrip StringExampleStrip.cpp )
ADD_CUSTOM_COMMAND( COMMAND StringExampleStrip > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StringExampleStrip.out OUTPUT StringExampleStrip.out )

ADD_EXECUTABLE( TimerExample TimerExample.cpp )
ADD_CUSTOM_COMMAND( COMMAND TimerExample > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/TimerExample.out OUTPUT TimerExample.out )

ADD_EXECUTABLE( TimerExampleLogger TimerExampleLogger.cpp )
ADD_CUSTOM_COMMAND( COMMAND TimerExampleLogger > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/TimerExampleLogger.out OUTPUT TimerExampleLogger.out )


ADD_CUSTOM_TARGET( RunExamples ALL DEPENDS
   FileExample.out
   FileExampleSaveAndLoad.out
   FileNameExample.out
   FileNameExampleDistributedSystemNodeCoordinates.out
   FileNameExampleDistributedSystemNodeId.out
   ObjectExample_getType.out
   ParseObjectTypeExample.out
   StringExample.out
   StringExampleGetAllocatedSize.out
   StringExampleReplace.out
   StringExampleSplit.out
   StringExampleStrip.out
   TimerExample.out
   TimerExampleLogger.out )

if( BUILD_CUDA )
   ADD_CUSTOM_TARGET( RunExamples-cuda ALL DEPENDS
      FileExampleCuda.out )
ENDIF()
