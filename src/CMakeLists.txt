INCLUDE_DIRECTORIES( config core debug diff matrix mesh solver )

add_subdirectory( config )
add_subdirectory( core )
add_subdirectory( debug )
add_subdirectory( diff )
add_subdirectory( matrix )
add_subdirectory( mesh )
add_subdirectory( solver )

# Build the main library
set( libtnl_0_1_la_LIBADD
     libtnlconfig${mpiExt}${debugExt}-0.1
     libtnlcore${mpiExt}${debugExt}-0.1
     libtnldebug${mpiExt}${debugExt}-0.1
     libtnldiff${mpiExt}${debugExt}-0.1 
     libtnlmatrix${mpiExt}${debugExt}-0.1
     libtnlmesh${mpiExt}${debugExt}-0.1
     libtnlsolver${mpiExt}${debugExt}-0.1
    )

add_library( libtnl${mpiExt}${debugExt}-0.1 SHARED dummy.cpp )

target_link_libraries( libtnl${mpiExt}${debugExt}-0.1 ${libtnl_0_1_la_LIBADD} )
if (BUILD_MPI)
  target_link_libraries(libtnl${mpiExt}${debugExt}-0.1 mpi)
endif()


