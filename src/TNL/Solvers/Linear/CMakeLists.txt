ADD_SUBDIRECTORY( Preconditioners )

SET( headers BICGStab.h
             BICGStab_impl.h
             BICGStabL.h
             BICGStabL_impl.h
             CG.h
             CG_impl.h
             GMRES.h
             GMRES_impl.h
             Jacobi.h
             LinearResidueGetter.h
             LinearResidueGetter_impl.h
             LinearSolver.h
             SOR.h
             SOR_impl.h
             TFQMR.h
             TFQMR_impl.h
             Traits.h
             UmfpackWrapper.h
             UmfpackWrapper_impl.h
   )

INSTALL( FILES ${headers} DESTINATION ${TNL_TARGET_INCLUDE_DIRECTORY}/Solvers/Linear )
