set( tnl_flow_sw_HEADERS
     CompressibleConservativeVariables.h )

set( tnl_flow_sw_SOURCES     
     navierStokes.cpp
     navierStokes.cu )
               
IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE(tnl-navier-stokes-sw${debugExt} navierStokes.cu)
   target_link_libraries (tnl-navier-stokes-sw${debugExt} tnl${debugExt}-${tnlVersion}  ${CUSPARSE_LIBRARY} )
ELSE(  BUILD_CUDA )               
   ADD_EXECUTABLE(tnl-navier-stokes-sw${debugExt} navierStokes.cpp)     
   target_link_libraries (tnl-navier-stokes-sw${debugExt} tnl${debugExt}-${tnlVersion} )
ENDIF( BUILD_CUDA )


INSTALL( TARGETS tnl-navier-stokes-sw${debugExt}
         RUNTIME DESTINATION bin
         PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE )
        
INSTALL( FILES run-navier-stokes-sw
               ${tnl_inviscid_flow_SOURCES}
         DESTINATION share/tnl-${tnlVersion}/examples/navier-stokes-sw )
