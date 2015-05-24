SET( headers tnlExactOperatorQ.h
             tnlExactOperatorQ_impl.h
             tnlOneSideDiffOperatorQ.h             
             tnlOneSideDiffOperatorQ_impl.h )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/operators/operator-Q )
