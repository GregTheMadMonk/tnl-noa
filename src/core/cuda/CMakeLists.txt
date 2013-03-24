set( headers device-check.h
             cuda-reduction.h
             reduction-operations.h )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/core/cuda )