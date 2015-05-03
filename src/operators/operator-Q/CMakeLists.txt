SET( headers tnlExactOperatorQ.h
             tnlExactOperatorQ_impl.h
             tnlOneSideDiffOperatorQForGraph.h             
             tnlOneSideDiffOperatorQForGraph_impl.h )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/operators/operator-Q )
