SET( headers godunovEikonal.h
	         parallelGodunovEikonal.h
   )
   
INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/operators/godunov-eikonal )
