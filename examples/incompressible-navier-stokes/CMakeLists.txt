set( tnl_incompressible_navier_stokes_SOURCES     
     tnl-incompressible-navier-stokes.cpp
tnlExplicitINSTimeStepper_impl.h
tnlExplicitINSTimeStepper.h
tnlIncompressibleNavierStokesProblem_impl.h
tnlIncompressibleNavierStokesProblem.h
tnlNSFastBuildConfig.h
visit_writer.h
visit_writer.cpp
solver.h
base.h
      )
               
IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE(tnl-incompressible-navier-stokes${debugExt} tnl-incompressible-navier-stokes.cu)
   
ELSE(  BUILD_CUDA )               
   ADD_EXECUTABLE(tnl-incompressible-navier-stokes${debugExt} ${tnl_incompressible_navier_stokes_SOURCES})
   
ENDIF( BUILD_CUDA )

target_link_libraries (tnl-incompressible-navier-stokes${debugExt} tnl${debugExt}-${tnlVersion} )

INSTALL( TARGETS tnl-incompressible-navier-stokes${debugExt}
         RUNTIME DESTINATION bin
         PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE )
        
#INSTALL( FILES tnl-run-incompressible-navier-stokes
#         DESTINATION share/tnl-${tnlVersion}/examples/incompressible-navier-stokes )
