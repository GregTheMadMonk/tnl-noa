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

ADD_EXECUTABLE( FileExample FileExample.cpp )
target_link_libraries( FileExample tnl )

ADD_EXECUTABLE( StringExample StringExample.cpp )
target_link_libraries( StringExample tnl )

ADD_EXECUTABLE( StringExampleGetSize StringExampleGetSize.cpp )
target_link_libraries( StringExampleGetSize tnl )

ADD_EXECUTABLE( StringExampleGetAllocatedSize StringExampleGetAllocatedSize.cpp )
target_link_libraries( StringExampleGetAllocatedSize tnl )

ADD_EXECUTABLE( TimerExample TimerExample.cpp )
target_link_libraries( TimerExample tnl )