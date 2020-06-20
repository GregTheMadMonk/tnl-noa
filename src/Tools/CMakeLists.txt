add_subdirectory (tnl-quickstart)

CONFIGURE_FILE( "tnl-compile.in" "${PROJECT_TOOLS_PATH}/tnl-compile" @ONLY )
CONFIGURE_FILE( "tnl-bindir.in" "${PROJECT_TOOLS_PATH}/tnl-bindir" @ONLY )

ADD_EXECUTABLE(tnl-grid-setup tnl-grid-setup.cpp )
ADD_EXECUTABLE(tnl-grid-to-mesh tnl-grid-to-mesh.cpp )
ADD_EXECUTABLE(tnl-mesh-converter tnl-mesh-converter.cpp )
ADD_EXECUTABLE(tnl-game-of-life tnl-game-of-life.cpp )
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

   target_compile_definitions(tnl-mesh-converter PUBLIC "-DHAVE_ZLIB")
   target_include_directories(tnl-mesh-converter PUBLIC ${ZLIB_INCLUDE_DIRS})
   target_link_libraries(tnl-mesh-converter ${ZLIB_LIBRARIES})

   target_compile_definitions(tnl-game-of-life PUBLIC "-DHAVE_ZLIB")
   target_include_directories(tnl-game-of-life PUBLIC ${ZLIB_INCLUDE_DIRS})
   target_link_libraries(tnl-game-of-life ${ZLIB_LIBRARIES})
endif()

find_package( tinyxml2 QUIET )
if( tinyxml2_FOUND )
   target_compile_definitions(tnl-mesh-converter PUBLIC "-DHAVE_TINYXML2")
   target_link_libraries(tnl-mesh-converter tinyxml2::tinyxml2)

   target_compile_definitions(tnl-game-of-life PUBLIC "-DHAVE_TINYXML2")
   target_link_libraries(tnl-game-of-life tinyxml2::tinyxml2)
endif()

find_package( METIS QUIET )
if( METIS_FOUND )
   add_executable(tnl-decompose-mesh tnl-decompose-mesh.cpp)
   target_include_directories(tnl-decompose-mesh PUBLIC ${METIS_INCLUDE_DIRS})
   target_link_libraries(tnl-decompose-mesh ${METIS_LIBRARIES})
   if( ZLIB_FOUND )
      target_compile_definitions(tnl-decompose-mesh PUBLIC "-DHAVE_ZLIB")
      target_include_directories(tnl-decompose-mesh PUBLIC ${ZLIB_INCLUDE_DIRS})
      target_link_libraries(tnl-decompose-mesh ${ZLIB_LIBRARIES})
   endif()
   if( tinyxml2_FOUND )
      target_compile_definitions(tnl-decompose-mesh PUBLIC "-DHAVE_TINYXML2")
      target_link_libraries(tnl-decompose-mesh tinyxml2::tinyxml2)
   endif()
   install( TARGETS tnl-decompose-mesh DESTINATION bin )
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
                 tnl-game-of-life
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
