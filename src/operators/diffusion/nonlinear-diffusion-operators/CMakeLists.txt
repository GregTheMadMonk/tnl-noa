SET( headers tnlOneSideDiffNonlinearOperator.h
             tnlOneSideDiffNonlinearOperator_impl.h
	     tnlFiniteVolumeNonlinearOperator.h
             tnlFiniteVolumeNonlinearOperator_impl.h )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/operators/diffusion/nonlinear-diffusion-operators )
