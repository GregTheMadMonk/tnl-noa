add_subdirectory( simple-examples )
add_subdirectory( heat-equation )
add_subdirectory( transport-equation )
add_subdirectory( navier-stokes )
add_subdirectory( inviscid-flow )
add_subdirectory( inviscid-flow-sw )
add_subdirectory( inviscid-flow-vl )
#add_subdirectory( mean-curvature-flow )
add_subdirectory( flow )
add_subdirectory( flow-sw )
add_subdirectory( flow-vl )

#add_subdirectory( mean-curvature-flow )
#add_subdirectory( hamilton-jacobi )
#add_subdirectory( hamilton-jacobi-parallel )
#add_subdirectory( fast-sweeping )
#add_subdirectory( hamilton-jacobi-parallel-map )
#add_subdirectory( fast-sweeping-map )
#add_subdirectory( narrow-band )

ADD_EXECUTABLE( ArrayExample ArrayExample.cpp )

ADD_EXECUTABLE( ConfigDescriptionExample ConfigDescriptionExample.cpp )
ADD_EXECUTABLE( FileExample FileExample.cpp )
ADD_EXECUTABLE( ListExample ListExample.cpp )
ADD_EXECUTABLE( LoggerExample LoggerExample.cpp )
ADD_EXECUTABLE( MathExample MathExample.cpp )

ADD_EXECUTABLE( ObjectExample_getType ObjectExample_getType.cpp )
ADD_CUSTOM_COMMAND( COMMAND ObjectExample_getType > ObjectExample_getType.out OUTPUT ObjectExample_getType.out )

ADD_EXECUTABLE( ParameterContainerExample ParameterContainerExample.cpp )

ADD_EXECUTABLE( StringExample StringExample.cpp )
ADD_CUSTOM_COMMAND( COMMAND StringExample > StringExample.out OUTPUT StringExample.out )

ADD_EXECUTABLE( StringExampleGetAllocatedSize StringExampleGetAllocatedSize.cpp )
ADD_CUSTOM_COMMAND( COMMAND StringExampleGetAllocatedSize > StringExampleGetAllocatedSize.out OUTPUT StringExampleGetAllocatedSize.out )

ADD_EXECUTABLE( StringExampleReplace StringExampleReplace.cpp )
ADD_CUSTOM_COMMAND( COMMAND StringExampleReplace > StringExampleReplace.out OUTPUT StringExampleReplace.out )

ADD_EXECUTABLE( StringExampleSetSize StringExampleSetSize.cpp )
ADD_CUSTOM_COMMAND( COMMAND StringExampleSetSize > StringExampleSetSize.out OUTPUT StringExampleSetSize.out )

ADD_EXECUTABLE( StringExampleSplit StringExampleSplit.cpp )
ADD_CUSTOM_COMMAND( COMMAND StringExampleSplit > StringExampleSplit.out OUTPUT StringExampleSplit.out )

ADD_EXECUTABLE( StringExampleStrip StringExampleStrip.cpp )
ADD_CUSTOM_COMMAND( COMMAND StringExampleStrip > StringExampleStrip.out OUTPUT StringExampleStrip.out )

ADD_EXECUTABLE( TimerExample TimerExample.cpp )
ADD_CUSTOM_COMMAND( COMMAND TimerExample > TimerExample.out OUTPUT TimerExample.out )

ADD_EXECUTABLE( TimerExampleLogger TimerExampleLogger.cpp )
ADD_CUSTOM_COMMAND( COMMAND TimerExampleLogger > TimerExampleLogger.out OUTPUT TimerExampleLogger.out )

ADD_EXECUTABLE( VectorExample VectorExample.cpp )

ADD_CUSTOM_TARGET( run ALL DEPENDS
   ObjectExample_getType.out
   StringExample.out
   StringExampleGetAllocatedSize.out
   StringExampleReplace.out
   StringExampleSplit.out
   StringExampleStrip.out
   TimerExample.out
   TimerExampleLogger.out )
