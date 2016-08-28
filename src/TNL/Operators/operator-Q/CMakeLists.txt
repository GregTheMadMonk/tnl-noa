SET( headers tnlOneSideDiffOperatorQ.h             
             tnlOneSideDiffOperatorQ_impl.h
	     tnlFiniteVolumeOperatorQ.h             
             tnlFiniteVolumeOperatorQ_impl.h )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/TNL/Operators/operator-Q )
