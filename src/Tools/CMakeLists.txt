add_subdirectory (tnl-quickstart)

CONFIGURE_FILE( "tnl-compile.in" "${PROJECT_TOOLS_PATH}/tnl-compile" @ONLY )
CONFIGURE_FILE( "tnl-link.in" "${PROJECT_TOOLS_PATH}/tnl-link" @ONLY )
CONFIGURE_FILE( "tnl-bindir.in" "${PROJECT_TOOLS_PATH}/tnl-bindir" @ONLY )

ADD_EXECUTABLE(tnl-grid-setup${debugExt} tnl-grid-setup.cpp )
target_link_libraries (tnl-grid-setup${debugExt} tnl${debugExt}-${tnlVersion} )

#ADD_EXECUTABLE(tnl-mesh-convert${debugExt} tnl-mesh-convert.cpp )
#target_link_libraries (tnl-mesh-convert${debugExt} tnl${debugExt}-${tnlVersion} )

ADD_EXECUTABLE(tnl-init${debugExt} tnl-init.cpp )
target_link_libraries (tnl-init${debugExt} tnl${debugExt}-${tnlVersion} )

ADD_EXECUTABLE(tnl-view${debugExt} tnl-view.cpp )
target_link_libraries (tnl-view${debugExt} tnl${debugExt}-${tnlVersion} )

ADD_EXECUTABLE(tnl-diff${debugExt} tnl-diff.cpp )
target_link_libraries (tnl-diff${debugExt} tnl${debugExt}-${tnlVersion} )

ADD_EXECUTABLE(tnl-image-converter${debugExt} tnl-image-converter.cpp )
target_link_libraries (tnl-image-converter${debugExt} tnl${debugExt}-${tnlVersion} ${PNG_LIBRARIES} ${JPEG_LIBRARIES} )

ADD_EXECUTABLE(tnl-dicom-reader${debugExt} tnl-dicom-reader.cpp )
target_link_libraries (tnl-dicom-reader${debugExt} tnl${debugExt}-${tnlVersion} ${DCMTK_LIBRARIES} )

ADD_EXECUTABLE( tnl-functions-benchmark${debugExt} functions-benchmark.cpp )
target_link_libraries( tnl-functions-benchmark${debugExt} tnl${debugExt}-${tnlVersion} )

ADD_EXECUTABLE(tnl-curve2gnuplot${debugExt} tnlcurve2gnuplot.cpp )
target_link_libraries (tnl-curve2gnuplot${debugExt} tnl${debugExt}-${tnlVersion} )

IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( tnl-cuda-arch${debugExt} tnl-cuda-arch.cu )
   INSTALL( TARGETS tnl-cuda-arch${debugExt}
            RUNTIME DESTINATION bin
            PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE )
ENDIF()

INSTALL( FILES ${PROJECT_TOOLS_PATH}/tnl-init${debugExt}
               ${PROJECT_TOOLS_PATH}/tnl-view${debugExt}
               ${PROJECT_TOOLS_PATH}/tnl-diff${debugExt}
               ${PROJECT_TOOLS_PATH}/tnl-grid-setup${debugExt}
               #${PROJECT_TOOLS_PATH}/tnl-mesh-convert${debugExt}
               ${PROJECT_TOOLS_PATH}/tnl-dicom-reader${debugExt}
               ${PROJECT_TOOLS_PATH}/tnl-image-converter${debugExt}
               ${PROJECT_TOOLS_PATH}/tnl-curve2gnuplot${debugExt}
               tnl-time-series2png
               tnl-err2eoc
               tnl-eoc-test-log
               tnl-log-to-html.py
               ${PROJECT_TOOLS_PATH}/tnl-compile 
               ${PROJECT_TOOLS_PATH}/tnl-link
               ${PROJECT_TOOLS_PATH}/tnl-bindir
         DESTINATION bin
         PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE )
