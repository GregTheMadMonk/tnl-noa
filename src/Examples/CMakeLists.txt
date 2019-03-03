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

ADD_EXECUTABLE( ParameterContainerExample ParameterContainerExample.cpp )

ADD_EXECUTABLE( StringExample StringExample.cpp )
ADD_CUSTOM_COMMAND( OUTPUT StringExample.out
                    COMMAND StringExample
                    DEPENDS StringExample
                    COMMENT Executing StringExample )
ADD_EXECUTABLE( StringExampleGetAllocatedSize StringExampleGetAllocatedSize.cpp )
ADD_EXECUTABLE( StringExampleReplace StringExampleReplace.cpp )
ADD_EXECUTABLE( StringExampleSetSize StringExampleSetSize.cpp )
ADD_EXECUTABLE( StringExampleSplit StringExampleSplit.cpp )
ADD_EXECUTABLE( StringExampleStrip StringExampleStrip.cpp )
ADD_EXECUTABLE( TimerExample TimerExample.cpp )

ADD_EXECUTABLE( VectorExample VectorExample.cpp )
