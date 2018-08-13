set( tnl_inviscid_flow_vl_HEADERS
     CompressibleConservativeVariables.h )

set( tnl_inviscid_flow_vl_SOURCES     
     euler.cpp
     euler.cu )
               
IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE(tnl-euler-vl${debugExt} euler.cu)
   target_link_libraries (tnl-euler-vl${debugExt} tnl${debugExt}-${tnlVersion}  ${CUSPARSE_LIBRARY} )
ELSE(  BUILD_CUDA )               
   ADD_EXECUTABLE(tnl-euler-vl${debugExt} euler.cpp)     
   target_link_libraries (tnl-euler-vl${debugExt} tnl${debugExt}-${tnlVersion} )
ENDIF( BUILD_CUDA )


INSTALL( TARGETS tnl-euler-vl${debugExt}
         RUNTIME DESTINATION bin
         PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE )
        
INSTALL( FILES run-euler-vl
               ${tnl_inviscid_flow_SOURCES}
         DESTINATION share/tnl-${tnlVersion}/examples/inviscid-flow-vl )
