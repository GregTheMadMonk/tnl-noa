set( tnl_heat_equation_SOURCES     
     advection.cpp
     advection.cu )
               
IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( tnl-advection{debugExt} advection.cu)   
   target_link_libraries( tnl-advection${debugExt} tnl${debugExt}-${tnlVersion}  ${CUSPARSE_LIBRARY} )
ELSE(  BUILD_CUDA )               
   ADD_EXECUTABLE( tnl-advection${debugExt} advection.cpp)     
   target_link_libraries( tnl-advection${debugExt} tnl${debugExt}-${tnlVersion} )
ENDIF( BUILD_CUDA )


INSTALL( TARGETS tnl-advection${debugExt}
         RUNTIME DESTINATION bin
         PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE )
        
INSTALL( FILES tnl-run-advection
               ${tnl_heat_equation_SOURCES}
         DESTINATION share/tnl-${tnlVersion}/examples/advection )
