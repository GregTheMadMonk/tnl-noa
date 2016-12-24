SET( headers EllpackIndexMultimap.h
             EllpackIndexMultimap_impl.h
             EllpackIndexMultimapValues.h
             EllpackIndexMultimapValues_impl.h
             StaticEllpackIndexMultimap.h
             StaticEllpackIndexMultimap_impl.h
             StaticEllpackIndexMultimapValues.h
             StaticEllpackIndexMultimapValues_impl.h )

INSTALL( FILES ${headers} DESTINATION ${TNL_TARGET_INCLUDE_DIRECTORY}/Experimental/Multimaps )
