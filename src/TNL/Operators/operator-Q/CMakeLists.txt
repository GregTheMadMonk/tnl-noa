SET( headers tnlOneSideDiffOperatorQ.h             
             tnlOneSideDiffOperatorQ_impl.h
	     tnlFiniteVolumeOperatorQ.h             
             tnlFiniteVolumeOperatorQ_impl.h )

INSTALL( FILES ${headers} DESTINATION ${TNL_TARGET_INCLUDE_DIRECTORY}/Operators/operator-Q )
