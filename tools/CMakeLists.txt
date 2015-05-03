add_subdirectory (src)
add_subdirectory (share)

CONFIGURE_FILE( "tnl-compile.in" "${PROJECT_TOOLS_PATH}/tnl-compile" )
CONFIGURE_FILE( "tnl-link.in" "${PROJECT_TOOLS_PATH}/tnl-link" )

INSTALL( FILES ${PROJECT_TOOLS_PATH}/tnl-compile 
               ${PROJECT_TOOLS_PATH}/tnl-link
         DESTINATION bin
         PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE )