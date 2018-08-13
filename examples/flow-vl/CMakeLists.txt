set( tnl_flow_vl_HEADERS
     CompressibleConservativeVariables.h )

set( tnl_flow_vl_SOURCES     
     navierStokes.cpp
     navierStokes.cu )
               
IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE(tnl-navier-stokes-vl${debugExt} navierStokes.cu)
   target_link_libraries (tnl-navier-stokes-vl${debugExt} tnl${debugExt}-${tnlVersion}  ${CUSPARSE_LIBRARY} )
ELSE(  BUILD_CUDA )               
   ADD_EXECUTABLE(tnl-navier-stokes-vl${debugExt} navierStokes.cpp)     
   target_link_libraries (tnl-navier-stokes-vl${debugExt} tnl${debugExt}-${tnlVersion} )
ENDIF( BUILD_CUDA )


INSTALL( TARGETS tnl-navier-stokes-vl${debugExt}
         RUNTIME DESTINATION bin
         PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE )
        
INSTALL( FILES run-navier-stokes-vl
               ${tnl_inviscid_flow_SOURCES}
         DESTINATION share/tnl-${tnlVersion}/examples/navier-stokes-vl )
