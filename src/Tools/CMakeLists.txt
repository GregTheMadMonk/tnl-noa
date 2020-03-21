add_subdirectory (tnl-quickstart)

CONFIGURE_FILE( "tnl-compile.in" "${PROJECT_TOOLS_PATH}/tnl-compile" @ONLY )
CONFIGURE_FILE( "tnl-bindir.in" "${PROJECT_TOOLS_PATH}/tnl-bindir" @ONLY )

ADD_EXECUTABLE(tnl-grid-setup tnl-grid-setup.cpp )
ADD_EXECUTABLE(tnl-grid-to-mesh tnl-grid-to-mesh.cpp )
ADD_EXECUTABLE(tnl-mesh-converter tnl-mesh-converter.cpp )
ADD_EXECUTABLE(tnl-init tnl-init.cpp )
ADD_EXECUTABLE(tnl-view tnl-view.cpp )
ADD_EXECUTABLE(tnl-diff tnl-diff.cpp )
ADD_EXECUTABLE(tnl-lattice-init tnl-lattice-init.cpp )

ADD_EXECUTABLE(tnl-image-converter tnl-image-converter.cpp )
if( PNG_FOUND )
   target_link_libraries(tnl-image-converter ${PNG_LIBRARIES} )
endif()
if( JPEG_FOUND )
   target_link_libraries(tnl-image-converter ${JPEG_LIBRARIES} )
endif()

ADD_EXECUTABLE(tnl-dicom-reader tnl-dicom-reader.cpp )
if( DCMTK_FOUND )
   target_link_libraries(tnl-dicom-reader ${DCMTK_LIBRARIES} )
endif()

find_package( ZLIB )
if( ZLIB_FOUND )
   target_compile_definitions(tnl-view PUBLIC "-DHAVE_ZLIB")
   target_include_directories(tnl-view PUBLIC ${ZLIB_INCLUDE_DIRS})
   target_link_libraries(tnl-view ${ZLIB_LIBRARIES})
endif()

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
                 tnl-grid-to-mesh
                 tnl-mesh-converter
                 tnl-dicom-reader
                 tnl-image-converter
                 tnl-lattice-init
         DESTINATION bin )

INSTALL( FILES ${PROJECT_TOOLS_PATH}/tnl-bindir
               ${PROJECT_TOOLS_PATH}/tnl-compile
               tnl-err2eoc
               tnl-log-to-html.py
         DESTINATION bin
         PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE )

