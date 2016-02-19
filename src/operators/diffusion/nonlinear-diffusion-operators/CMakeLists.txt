SET( headers tnlFiniteVolumeNonlinearOperator.h
             tnlFiniteVolumeNonlinearOperator_impl.h )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/operators/diffusion/nonlinear-diffusion-operators )
