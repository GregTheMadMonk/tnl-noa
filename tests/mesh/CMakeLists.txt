ADD_SUBDIRECTORY( share )

set( headers mesh_info.h
             test_configs.h )
             
ADD_EXECUTABLE( tnl-mesh-tests${mpiExt}${debugExt} ${headers} main.cpp )
TARGET_LINK_LIBRARIES( tnl-mesh-tests${mpiExt}${debugExt} ${CPPUNIT_LIBRARIES}
                       tnl${mpiExt}${debugExt}-0.1 )

ADD_TEST( tnl-mesh-tests${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/tnl-mesh-tests${mpiExt}${debugExt}  )
