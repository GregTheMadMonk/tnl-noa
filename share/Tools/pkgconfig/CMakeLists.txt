CONFIGURE_FILE( "tnl.pc.in" "${PROJECT_TOOLS_PATH}/share/pkgconfig/tnl.pc" @ONLY )
INSTALL( FILES ${PROJECT_TOOLS_PATH}/share/pkgconfig/tnl.pc
         DESTINATION share/pkgconfig )

if( CUDA_FOUND )
    CONFIGURE_FILE( "cuda.pc.in" "${PROJECT_TOOLS_PATH}/share/pkgconfig/cuda.pc" @ONLY )
    CONFIGURE_FILE( "tnl-cuda.pc.in" "${PROJECT_TOOLS_PATH}/share/pkgconfig/tnl-cuda.pc" @ONLY )
    INSTALL( FILES ${PROJECT_TOOLS_PATH}/share/pkgconfig/cuda.pc
                   ${PROJECT_TOOLS_PATH}/share/pkgconfig/tnl-cuda.pc
             DESTINATION share/pkgconfig )
endif()

if( OPENMP_FOUND )
    CONFIGURE_FILE( "tnl-openmp.pc.in" "${PROJECT_TOOLS_PATH}/share/pkgconfig/tnl-openmp.pc" @ONLY )
    INSTALL( FILES ${PROJECT_TOOLS_PATH}/share/pkgconfig/tnl-openmp.pc
             DESTINATION share/pkgconfig )
endif()
