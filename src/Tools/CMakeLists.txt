add_subdirectory (tnl-quickstart)

CONFIGURE_FILE( "tnl-compile.in" "${PROJECT_TOOLS_PATH}/tnl-compile" @ONLY )
CONFIGURE_FILE( "tnl-link.in" "${PROJECT_TOOLS_PATH}/tnl-link" @ONLY )
CONFIGURE_FILE( "tnl-bindir.in" "${PROJECT_TOOLS_PATH}/tnl-bindir" @ONLY )

SET( tnlcurve2gnuplotsources tnlcurve2gnuplot.cpp )

SET( tnlfunctionsbenchmarksources functions-benchmark.cpp
                                  functions-benchmark.h )

set( tnl_grid_setup_SOURCES tnl-grid-setup.cpp
                            tnl-grid-setup.h )
                            
set( tnl_mesh_convert_SOURCES tnl-mesh-convert.cpp
                            tnl-mesh-convert.h )                            
                            
set( tnl_init_SOURCES tnl-init.cpp
                      tnl-init.h )

set( tnl_view_SOURCES tnl-view.cpp
                      tnl-view.h )
                      
set( tnl_diff_SOURCES tnl-diff.cpp
                      tnl-diff.h ) 

set( tnl_dicom_reader_SOURCES tnl-dicom-reader.cpp )

set( tnl_image_converter_SOURCES tnl-image-converter.cpp )

ADD_EXECUTABLE(tnl-grid-setup${debugExt} ${tnl_grid_setup_SOURCES})
target_link_libraries (tnl-grid-setup${debugExt} tnl${debugExt}-${tnlVersion} )

ADD_EXECUTABLE(tnl-mesh-convert${debugExt} ${tnl_mesh_convert_SOURCES})
target_link_libraries (tnl-mesh-convert${debugExt} tnl${debugExt}-${tnlVersion} )

ADD_EXECUTABLE(tnl-init${debugExt} ${tnl_init_SOURCES})
target_link_libraries (tnl-init${debugExt} tnl${debugExt}-${tnlVersion} )

ADD_EXECUTABLE(tnl-view${debugExt} ${tnl_view_SOURCES})
target_link_libraries (tnl-view${debugExt} tnl${debugExt}-${tnlVersion} )

ADD_EXECUTABLE(tnl-diff${debugExt} ${tnl_diff_SOURCES})
target_link_libraries (tnl-diff${debugExt} tnl${debugExt}-${tnlVersion} )

ADD_EXECUTABLE(tnl-image-converter${debugExt} ${tnl_image_converter_SOURCES})
target_link_libraries (tnl-image-converter${debugExt} tnl${debugExt}-${tnlVersion} ${PNG_LIBRARIES} ${JPEG_LIBRARIES} )

ADD_EXECUTABLE(tnl-dicom-reader${debugExt} ${tnl_dicom_reader_SOURCES})
target_link_libraries (tnl-dicom-reader${debugExt} tnl${debugExt}-${tnlVersion} ${DCMTK_LIBRARIES} )

ADD_EXECUTABLE( tnl-functions-benchmark${debugExt} ${tnlfunctionsbenchmarksources} )
target_link_libraries( tnl-functions-benchmark${debugExt} tnl${debugExt}-${tnlVersion} )

ADD_EXECUTABLE(tnl-curve2gnuplot${debugExt} ${tnlcurve2gnuplotsources})
target_link_libraries (tnl-curve2gnuplot${debugExt} tnl${debugExt}-${tnlVersion} )

IF( BUILD_CUDA )
    CUDA_ADD_EXECUTABLE( tnl-cuda-arch${debugExt} tnl-cuda-arch.cu
                         OPTIONS ${CUDA_ADD_EXECUTABLE_OPTIONS} )
    SET_TARGET_PROPERTIES( tnl-cuda-arch${debugExt} PROPERTIES CUDA_COMPILE_FLAGS "${CXX_OPTIMIZE_FLAGS}" )
ENDIF()
         
IF( BUILD_CUDA )
   INSTALL( TARGETS tnl-cuda-arch${debugExt}
            RUNTIME DESTINATION bin
            PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE )
ENDIF()

INSTALL( FILES ${PROJECT_TOOLS_PATH}/tnl-init${debugExt}
               ${PROJECT_TOOLS_PATH}/tnl-view${debugExt}
               ${PROJECT_TOOLS_PATH}/tnl-diff${debugExt}
               ${PROJECT_TOOLS_PATH}/tnl-grid-setup${debugExt}
               ${PROJECT_TOOLS_PATH}/tnl-mesh-convert${debugExt}
               ${PROJECT_TOOLS_PATH}/tnl-dicom-reader${debugExt}
               ${PROJECT_TOOLS_PATH}/tnl-image-converter${debugExt}
               ${PROJECT_TOOLS_PATH}/tnl-curve2gnuplot${debugExt}
               tnl-time-series2png
               tnl-err2eoc
               tnl-eoc-test-log
               ${PROJECT_TOOLS_PATH}/tnl-compile 
               ${PROJECT_TOOLS_PATH}/tnl-link
               ${PROJECT_TOOLS_PATH}/tnl-bindir
         DESTINATION bin
         PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE )
