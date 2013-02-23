SET( headers tnlBasicTypesSetter_impl.h )

set( tnl_implementation_SOURCES
     ${tnl_implementation_config_SOURCES}
     PARENT_SCOPE )

INSTALL( FILES ${headers} DESTINATION include/tnl-${tnlVersion}/implementation/config )

