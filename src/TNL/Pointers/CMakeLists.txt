SET( headers DevicePointer.h
             SharedPointer.h
             SharedPointerCuda.h
             SharedPointerHost.h
             SharedPointerMic.h
             SmartPointer.h
             SmartPointersRegister.h
             UniquePointer.h
   )
   
SET( CURRENT_DIR ${CMAKE_SOURCE_DIR}/src/TNL/Pointers )
set( common_SOURCES
     ${CURRENT_DIR}/SmartPointersRegister.cpp
   )

SET( tnl_pointers_SOURCES 
     ${common_SOURCES}
     PARENT_SCOPE )

if( BUILD_CUDA )
SET( tnl_pointers_CUDA__SOURCES
     ${common_SOURCES} 
     PARENT_SCOPE )
endif() 

INSTALL( FILES ${headers} DESTINATION ${TNL_TARGET_INCLUDE_DIRECTORY}/Pointers )
