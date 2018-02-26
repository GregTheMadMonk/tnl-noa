set( headers godunovEikonal.h
	     parallelGodunovEikonal.h
	     parallelGodunovMap.h
	     godunovEikonal1D_impl.h
             godunovEikonal2D_impl.h
             godunovEikonal3D_impl.h
             parallelGodunovEikonal1D_impl.h
             parallelGodunovEikonal2D_impl.h
             parallelGodunovEikonal3D_impl.h
             parallelGodunovMap2D_impl.h)

SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/implementation/operators/godunov-eikonal )

if( BUILD_CUDA)
      set( tnl_implementation_operators_godunov-eikonal_CUDA__SOURCES
        ${common_SOURCES}
        PARENT_SCOPE )
endif()

set( tnl_implementation_operators_godunov-eikonal_SOURCES
     ${common_SOURCES}
     PARENT_SCOPE )

INSTALL( FILES ${headers} DESTINATION ${TNL_TARGET_INCLUDE_DIRECTORY}/Operators/Godunov-Eikonal/ )
