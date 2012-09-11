SET( headers config_base.h
             hexahedral.h
             linear.h
             quadrilateral.h
             tetrahedral.h
             triangular.h
             triangular_noedges.h )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/mesh/config )