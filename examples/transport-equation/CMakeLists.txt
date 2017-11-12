set( tnl_transport_equation_SOURCES     
     tnl-transport-equation.cpp
     tnl-transport-equation.cu )

set( tnl_transport_equation_HEADERS
     tnl-transport-equation.h
     transportEquationBuildConfigTag.h
     transportEquationProblem.h
     transportEquationProblem_impl.h )
     
               
IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( tnl-transport-equation${debugExt} tnl-transport-equation.cu)   
   target_link_libraries( tnl-transport-equation${debugExt} tnl${debugExt}-${tnlVersion}  ${CUSPARSE_LIBRARY} )
   CUDA_ADD_EXECUTABLE( tnl-transport-equation-eoc${debugExt} tnl-transport-equation-eoc.cu)   
   target_link_libraries( tnl-transport-equation-eoc${debugExt} tnl${debugExt}-${tnlVersion}  ${CUSPARSE_LIBRARY} )

ELSE(  BUILD_CUDA )               
   ADD_EXECUTABLE( tnl-transport-equation${debugExt} tnl-transport-equation.cpp)     
   target_link_libraries( tnl-transport-equation${debugExt} tnl${debugExt}-${tnlVersion} )
   ADD_EXECUTABLE( tnl-transport-equation-eoc${debugExt} tnl-transport-equation-eoc.cpp)     
   target_link_libraries( tnl-transport-equation-eoc${debugExt} tnl${debugExt}-${tnlVersion} )
ENDIF( BUILD_CUDA )


INSTALL( TARGETS tnl-transport-equation${debugExt}
                 tnl-transport-equation-eoc${debugExt}
         RUNTIME DESTINATION bin
         PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE )
        
INSTALL( FILES tnl-run-transport-equation
               ${tnl_transport_equation_SOURCES}
               ${tnl_transport_equation_HEADERS}
         DESTINATION share/tnl-${tnlVersion}/examples/transport-equation )
