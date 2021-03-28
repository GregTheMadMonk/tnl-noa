set( COMMON_EXAMPLES
)
set( CPP_EXAMPLES
   ReadMeshExample
   ${COMMON_EXAMPLES}
)
set( CUDA_EXAMPLES
   ${COMMON_EXAMPLES}
)

foreach( target IN ITEMS ${CPP_EXAMPLES} )
   add_executable( ${target} ${target}.cpp )
   add_custom_command( COMMAND ${target} > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out OUTPUT ${target}.out )
   set( CPP_OUTPUTS ${CPP_OUTPUTS} ${target}.out )
endforeach()
add_custom_target( RunTutorialsMeshesExamples ALL DEPENDS ${CPP_OUTPUTS} )

if( BUILD_CUDA )
   foreach( target IN ITEMS ${CUDA_EXAMPLES} )
      cuda_add_executable( ${target}-cuda ${target}.cu OPTIONS )
      add_custom_command( COMMAND ${target}-cuda > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out OUTPUT ${target}.out )
      set( CUDA_OUTPUTS ${CUDA_OUTPUTS} ${target}.out )
   endforeach()
   add_custom_target( RunTutorialsMeshesCudaExamples ALL DEPENDS ${CUDA_OUTPUTS} )
endif()

find_package( ZLIB )
if( ZLIB_FOUND )
   foreach( target IN ITEMS ${CPP_EXAMPLES} ${CUDA_EXAMPLES} )
      target_compile_definitions(${target} PUBLIC "-DHAVE_ZLIB")
      target_include_directories(${target} PUBLIC ${ZLIB_INCLUDE_DIRS})
      target_link_libraries(${target} ${ZLIB_LIBRARIES})
   endforeach()
endif()

find_package( tinyxml2 QUIET )
if( tinyxml2_FOUND )
   foreach( target IN ITEMS ${CPP_EXAMPLES} ${CUDA_EXAMPLES} )
      target_compile_definitions(${target} PUBLIC "-DHAVE_TINYXML2")
      target_link_libraries(${target} tinyxml2::tinyxml2)
   endforeach()
endif()


set( DATA_FILES
   example-triangles.vtu
)

foreach( file IN ITEMS ${DATA_FILES} )
   configure_file(${file} ${CMAKE_CURRENT_BINARY_DIR}/${file} COPYONLY)
endforeach()
