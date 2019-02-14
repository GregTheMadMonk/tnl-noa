SET( headers DevicePointer.h
             SharedPointer.h
             SharedPointerCuda.h
             SharedPointerHost.h
             SharedPointerMic.h
             SmartPointer.h
             SmartPointersRegister.h
             UniquePointer.h
   )

INSTALL( FILES ${headers} DESTINATION ${TNL_TARGET_INCLUDE_DIRECTORY}/Pointers )
