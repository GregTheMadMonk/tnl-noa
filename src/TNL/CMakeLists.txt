ADD_SUBDIRECTORY( Config )
ADD_SUBDIRECTORY( Containers )
ADD_SUBDIRECTORY( Communicators )
ADD_SUBDIRECTORY( Debugging )
ADD_SUBDIRECTORY( Devices )
ADD_SUBDIRECTORY( Exceptions )
ADD_SUBDIRECTORY( Experimental )
ADD_SUBDIRECTORY( Functions )
ADD_SUBDIRECTORY( Images )
ADD_SUBDIRECTORY( Matrices )
ADD_SUBDIRECTORY( Meshes )
ADD_SUBDIRECTORY( Operators )
ADD_SUBDIRECTORY( Pointers )
ADD_SUBDIRECTORY( Problems )
ADD_SUBDIRECTORY( Solvers )

set( headers
     Atomic.h
     Assert.h
     CudaSharedMemory.h
     CudaStreamPool.h
     File.h
     File_impl.h
     FileName.h
     FileName.hpp
     Object.h
     Object_impl.h
     Logger.h
     Logger_impl.h
     Math.h
     ParallelFor.h
     param-types.h
     StaticFor.h
     String.h
     String_impl.h
     Timer.h
     Timer_impl.h
     StaticVectorFor.h )

INSTALL( FILES ${headers} DESTINATION ${TNL_TARGET_INCLUDE_DIRECTORY} )
