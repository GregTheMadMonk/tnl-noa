/***************************************************************************
                          MemoryHelpers.h  -  description
                             -------------------
    begin                : Aug 19, 2019
    copyright            : (C) 2019 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <iostream>

#include <TNL/Cuda/CheckDevice.h>
#include <TNL/Exceptions/CudaSupportMissing.h>
#include <TNL/Exceptions/CudaBadAlloc.h>

namespace TNL {
namespace Cuda {

template< typename ObjectType >
[[deprecated("Allocators and MemoryOperations hould be used instead.")]]
ObjectType* passToDevice( const ObjectType& object )
{
#ifdef HAVE_CUDA
   ObjectType* deviceObject;
   if( cudaMalloc( ( void** ) &deviceObject,
                   ( size_t ) sizeof( ObjectType ) ) != cudaSuccess )
      throw Exceptions::CudaBadAlloc();
   if( cudaMemcpy( ( void* ) deviceObject,
                   ( void* ) &object,
                   sizeof( ObjectType ),
                   cudaMemcpyHostToDevice ) != cudaSuccess )
   {
      TNL_CHECK_CUDA_DEVICE;
      cudaFree( ( void* ) deviceObject );
      TNL_CHECK_CUDA_DEVICE;
      return 0;
   }
   return deviceObject;
#else
   throw Exceptions::CudaSupportMissing();
#endif
}

template< typename ObjectType >
[[deprecated("Allocators and MemoryOperations hould be used instead.")]]
ObjectType passFromDevice( const ObjectType* object )
{
#ifdef HAVE_CUDA
   ObjectType aux;
   cudaMemcpy( ( void* ) aux,
               ( void* ) &object,
               sizeof( ObjectType ),
               cudaMemcpyDeviceToHost );
   TNL_CHECK_CUDA_DEVICE;
   return aux;
#else
   throw Exceptions::CudaSupportMissing();
#endif
}

template< typename ObjectType >
[[deprecated("Allocators and MemoryOperations hould be used instead.")]]
void passFromDevice( const ObjectType* deviceObject,
                     ObjectType& hostObject )
{
#ifdef HAVE_CUDA
   cudaMemcpy( ( void* ) &hostObject,
               ( void* ) deviceObject,
               sizeof( ObjectType ),
               cudaMemcpyDeviceToHost );
   TNL_CHECK_CUDA_DEVICE;
#else
   throw Exceptions::CudaSupportMissing();
#endif
}

template< typename ObjectType >
[[deprecated("Allocators and MemoryOperations hould be used instead.")]]
void freeFromDevice( ObjectType* deviceObject )
{
#ifdef HAVE_CUDA
   cudaFree( ( void* ) deviceObject );
   TNL_CHECK_CUDA_DEVICE;
#else
   throw Exceptions::CudaSupportMissing();
#endif
}

template< typename ObjectType >
void print( const ObjectType* deviceObject, std::ostream& str = std::cout )
{
#ifdef HAVE_CUDA
   ObjectType hostObject;
   passFromDevice( deviceObject, hostObject );
   str << hostObject;
#endif
}

} // namespace Cuda
} // namespace TNL
