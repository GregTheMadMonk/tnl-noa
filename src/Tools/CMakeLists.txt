add_subdirectory (tnl-quickstart)

CONFIGURE_FILE( "tnl-compile.in" "${PROJECT_TOOLS_PATH}/tnl-compile" @ONLY )
CONFIGURE_FILE( "tnl-link.in" "${PROJECT_TOOLS_PATH}/tnl-link" @ONLY )
CONFIGURE_FILE( "tnl-bindir.in" "${PROJECT_TOOLS_PATH}/tnl-bindir" @ONLY )

ADD_EXECUTABLE(tnl-grid-setup tnl-grid-setup.cpp )
target_link_libraries (tnl-grid-setup tnl )

#ADD_EXECUTABLE(tnl-mesh-convert tnl-mesh-convert.cpp )
#target_link_libraries (tnl-mesh-convert tnl )

ADD_EXECUTABLE(tnl-init tnl-init.cpp )
target_link_libraries (tnl-init tnl )

# TODO: refactoring, this is useful in multiple places
find_package( VTK )
if( VTK_FOUND )
   include(${VTK_USE_FILE})

   AddCompilerFlag( "-DHAVE_VTK " )

   add_executable(tnl-view tnl-view.cpp )

   target_link_libraries( tnl-view
                          vtkCommonCore
                          vtkIOLegacy
                          tnl )
else( VTK_FOUND )
   add_executable(tnl-view tnl-view.cpp )
   target_link_libraries (tnl-view tnl )
endif( VTK_FOUND )

ADD_EXECUTABLE(tnl-diff tnl-diff.cpp )
target_link_libraries (tnl-diff tnl )

ADD_EXECUTABLE(tnl-image-converter tnl-image-converter.cpp )
target_link_libraries (tnl-image-converter tnl ${PNG_LIBRARIES} ${JPEG_LIBRARIES} )

ADD_EXECUTABLE(tnl-dicom-reader tnl-dicom-reader.cpp )
target_link_libraries (tnl-dicom-reader tnl ${DCMTK_LIBRARIES} )

ADD_EXECUTABLE( tnl-functions-benchmark functions-benchmark.cpp )
target_link_libraries( tnl-functions-benchmark tnl )

ADD_EXECUTABLE(tnl-curve2gnuplot tnlcurve2gnuplot.cpp )
target_link_libraries (tnl-curve2gnuplot tnl )

IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( tnl-cuda-arch tnl-cuda-arch.cu )
   INSTALL( TARGETS tnl-cuda-arch
            RUNTIME DESTINATION bin
            PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE )
ENDIF()

INSTALL( TARGETS tnl-init
                 tnl-view
                 tnl-diff
                 tnl-grid-setup
                 #tnl-mesh-convert
                 tnl-dicom-reader
                 tnl-image-converter
                 tnl-curve2gnuplot
         DESTINATION bin )

INSTALL( FILES ${PROJECT_TOOLS_PATH}/tnl-bindir
               ${PROJECT_TOOLS_PATH}/tnl-compile
               ${PROJECT_TOOLS_PATH}/tnl-link
               tnl-time-series2png
               tnl-err2eoc
               tnl-eoc-test-log
               tnl-log-to-html.py
         DESTINATION bin
         PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE )
