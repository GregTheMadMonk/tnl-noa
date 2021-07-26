if( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( tnl-benchmark-mesh-cuda tnl-benchmark-mesh.cu )
   install( TARGETS tnl-benchmark-mesh-cuda RUNTIME DESTINATION bin )
endif()

ADD_EXECUTABLE( tnl-benchmark-mesh tnl-benchmark-mesh.cpp )

install( TARGETS tnl-benchmark-mesh RUNTIME DESTINATION bin )
