ADD_EXECUTABLE( BoundaryTagsTest${mpiExt}${debugExt} ${headers} BoundaryTagsTest.cpp )
SET_TARGET_PROPERTIES( BoundaryTagsTest${mpiExt}${debugExt} PROPERTIES COMPILE_FLAGS "${CXX_TESTS_FLAGS}" )
TARGET_LINK_LIBRARIES( BoundaryTagsTest${mpiExt}${debugExt} ${GTEST_BOTH_LIBRARIES}
                                                           tnl${mpiExt}${debugExt}-${tnlVersion} )

if( BUILD_CUDA AND ${CMAKE_CXX_COMPILER} MATCHES ".*clang\\+\\+" )
   # FIXME: nvcc fails with "Invalid memory reference" error, so we assume that ${CXX} is clang++
   #CUDA_ADD_EXECUTABLE( MeshTest${mpiExt}${debugExt} ${headers} MeshTest.cu OPTIONS "${CXX_TESTS_FLAGS}" )
   #TARGET_LINK_LIBRARIES( MeshTest${mpiExt}${debugExt} ${GTEST_BOTH_LIBRARIES}
   #                                                        tnl${mpiExt}${debugExt}-${tnlVersion} )
   ADD_EXECUTABLE( MeshTest${mpiExt}${debugExt} ${headers} MeshTest.cpp )
   SET_TARGET_PROPERTIES( MeshTest${mpiExt}${debugExt} PROPERTIES COMPILE_FLAGS
                          "${CXX_TESTS_FLAGS} -DHAVE_CUDA -x cuda --cuda-path=${CUDA_TOOLKIT_ROOT_DIR} --cuda-gpu-arch=sm_21" )
   TARGET_LINK_LIBRARIES( MeshTest${mpiExt}${debugExt}
                          ${GTEST_BOTH_LIBRARIES}
                          tnl${mpiExt}${debugExt}-${tnlVersion}
                          cudart dl rt "-L ${CUDA_TOOLKIT_ROOT_DIR}/lib64/" )

   ADD_EXECUTABLE( MeshOrderingTest${mpiExt}${debugExt} ${headers} MeshOrderingTest.cpp )
   SET_TARGET_PROPERTIES( MeshOrderingTest${mpiExt}${debugExt} PROPERTIES COMPILE_FLAGS
                          "${CXX_TESTS_FLAGS} -DHAVE_CUDA -x cuda --cuda-path=${CUDA_TOOLKIT_ROOT_DIR} --cuda-gpu-arch=sm_21" )
   TARGET_LINK_LIBRARIES( MeshOrderingTest${mpiExt}${debugExt}
                          ${GTEST_BOTH_LIBRARIES}
                          tnl${mpiExt}${debugExt}-${tnlVersion}
                          cudart dl rt "-L ${CUDA_TOOLKIT_ROOT_DIR}/lib64/" )
else()
   ADD_EXECUTABLE( MeshTest${mpiExt}${debugExt} ${headers} MeshTest.cpp )
   SET_TARGET_PROPERTIES( MeshTest${mpiExt}${debugExt} PROPERTIES COMPILE_FLAGS "${CXX_TESTS_FLAGS}" )
   TARGET_LINK_LIBRARIES( MeshTest${mpiExt}${debugExt} ${GTEST_BOTH_LIBRARIES}
                                                              tnl${mpiExt}${debugExt}-${tnlVersion} )

   ADD_EXECUTABLE( MeshOrderingTest${mpiExt}${debugExt} ${headers} MeshOrderingTest.cpp )
   SET_TARGET_PROPERTIES( MeshOrderingTest${mpiExt}${debugExt} PROPERTIES COMPILE_FLAGS "${CXX_TESTS_FLAGS}" )
   TARGET_LINK_LIBRARIES( MeshOrderingTest${mpiExt}${debugExt} ${GTEST_BOTH_LIBRARIES}
                                                              tnl${mpiExt}${debugExt}-${tnlVersion} )
endif()

ADD_EXECUTABLE( MeshEntityTest${mpiExt}${debugExt} ${headers} MeshEntityTest.cpp )
SET_TARGET_PROPERTIES( MeshEntityTest${mpiExt}${debugExt} PROPERTIES COMPILE_FLAGS "${CXX_TESTS_FLAGS}" )
TARGET_LINK_LIBRARIES( MeshEntityTest${mpiExt}${debugExt} ${GTEST_BOTH_LIBRARIES}
                                                           tnl${mpiExt}${debugExt}-${tnlVersion} )

ADD_EXECUTABLE( MultimapTest${mpiExt}${debugExt} ${headers} MultimapTest.cpp )
SET_TARGET_PROPERTIES( MultimapTest${mpiExt}${debugExt} PROPERTIES COMPILE_FLAGS "${CXX_TESTS_FLAGS}" )
TARGET_LINK_LIBRARIES( MultimapTest${mpiExt}${debugExt} ${GTEST_BOTH_LIBRARIES}
                                                           tnl${mpiExt}${debugExt}-${tnlVersion} )

ADD_EXECUTABLE( StaticMultimapTest${mpiExt}${debugExt} ${headers} StaticMultimapTest.cpp )
SET_TARGET_PROPERTIES( StaticMultimapTest${mpiExt}${debugExt} PROPERTIES COMPILE_FLAGS "${CXX_TESTS_FLAGS}" )
TARGET_LINK_LIBRARIES( StaticMultimapTest${mpiExt}${debugExt} ${GTEST_BOTH_LIBRARIES}
                                                           tnl${mpiExt}${debugExt}-${tnlVersion} )


ADD_TEST( BoundaryTagsTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/BoundaryTagsTest${mpiExt}${debugExt} )
ADD_TEST( MeshTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/MeshTest${mpiExt}${debugExt} )
ADD_TEST( MeshOrderingTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/MeshOrderingTest${mpiExt}${debugExt} )
ADD_TEST( MeshEntityTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/MeshEntityTest${mpiExt}${debugExt} )
ADD_TEST( MultimapTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/MultimapTest${mpiExt}${debugExt} )
ADD_TEST( StaticMultimapTest${mpiExt}${debugExt} ${EXECUTABLE_OUTPUT_PATH}/MultimapTest${mpiExt}${debugExt} )



##
## Tests with VTK
##

find_package( VTK )
if( VTK_FOUND )
   include(${VTK_USE_FILE})

   AddCompilerFlag( "-DHAVE_VTK " )
   SET( VTK_COMMON_LIBRARIES vtkCommonCore ; vtkIOLegacy )
endif( VTK_FOUND )

if( BUILD_CUDA AND ${CMAKE_CXX_COMPILER} MATCHES ".*clang\\+\\+" )
   # FIXME: nvcc fails with "Invalid memory reference" error, so we assume that ${CXX} is clang++
   #CUDA_ADD_EXECUTABLE( MeshReaderTest${mpiExt}${debugExt} ${headers} MeshTest.cu OPTIONS "${CXX_TESTS_FLAGS}" )
   #TARGET_LINK_LIBRARIES( MeshReaderTest${mpiExt}${debugExt} ${GTEST_BOTH_LIBRARIES}
   #                                                        vtkCommonCore
   #                                                        vtkIOLegacy
   #                                                        tnl${mpiExt}${debugExt}-${tnlVersion} )
   ADD_EXECUTABLE( MeshReaderTest${mpiExt}${debugExt} ${headers} MeshReaderTest.cpp )
   SET_TARGET_PROPERTIES( MeshReaderTest${mpiExt}${debugExt} PROPERTIES COMPILE_FLAGS
                          "${CXX_TESTS_FLAGS} -DHAVE_CUDA -x cuda --cuda-path=${CUDA_TOOLKIT_ROOT_DIR} --cuda-gpu-arch=sm_21" )
   TARGET_LINK_LIBRARIES( MeshReaderTest${mpiExt}${debugExt}
                          ${GTEST_BOTH_LIBRARIES}
                          ${VTK_COMMON_LIBRARIES}
                          tnl${mpiExt}${debugExt}-${tnlVersion}
                          cudart dl rt "-L ${CUDA_TOOLKIT_ROOT_DIR}/lib64/" )
else()
   add_executable( MeshReaderTest${mpiExt}${debugExt} MeshReaderTest.cpp )
   set_target_properties( MeshReaderTest${mpiExt}${debugExt} PROPERTIES COMPILE_FLAGS "${CXX_TESTS_FLAGS}" )

   #target_link_libraries( MeshReaderTest${mpiExt}${debugExt}
   #                       ${VTK_LIBRARIES}
   #                       tnl${debugExt}-${tnlVersion} )
   target_link_libraries( MeshReaderTest${mpiExt}${debugExt}
                          ${GTEST_BOTH_LIBRARIES}
                          ${VTK_COMMON_LIBRARIES}
                          tnl${debugExt}-${tnlVersion} )
endif()
