add_subdirectory( share )

INSTALL( FILES Makefile
               main.cpp
               navierStokesSolver.h
               navierStokesSolver_impl.h
               navierStokesSetter.h
               navierStokesSetter_impl.h
               navier-stokes.cfg.desc
         DESTINATION share/tnl-${tnlVersion}/examples/navier-stokes )