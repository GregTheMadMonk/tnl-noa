SET( headers tnlMPIMesh.h
             tnlMPIMesh2D.h 
             tnlMPIMesh3D.h  
             curve-ident.h )
                    
SET( libtnldiffincludedir ${TNL_INCLUDE_DIR}/diff )
SET( libtnldiffinclude_HEADERS ${headers} )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/diff )