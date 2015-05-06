add_subdirectory (src)
add_subdirectory (share)
add_subdirectory (tnl-quickstart)

CONFIGURE_FILE( "tnl-compile.in" "${PROJECT_TOOLS_PATH}/tnl-compile" @ONLY )
CONFIGURE_FILE( "tnl-link.in" "${PROJECT_TOOLS_PATH}/tnl-link" @ONLY )

INSTALL( FILES ${PROJECT_TOOLS_PATH}/tnl-compile 
               ${PROJECT_TOOLS_PATH}/tnl-link
         DESTINATION bin
         PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE )