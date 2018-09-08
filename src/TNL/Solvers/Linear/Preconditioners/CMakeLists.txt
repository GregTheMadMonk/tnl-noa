SET( headers Preconditioner.h
             Diagonal.h
             Diagonal_impl.h
             ILU0.h
             ILU0_impl.h
             ILUT.h
             ILUT_impl.h
   )

INSTALL( FILES ${headers} DESTINATION ${TNL_TARGET_INCLUDE_DIRECTORY}/Solvers/Linear/Preconditioners )
