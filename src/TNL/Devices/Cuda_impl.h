/***************************************************************************
                          Cuda_impl.h  -  description
                             -------------------
    begin                : Jan 21, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Devices/Cuda.h>

namespace TNL {
namespace Devices {   

__cuda_callable__ 
inline int Cuda::getMaxGridSize()
{
   // TODO: make it preprocessor macro constant defined in tnlConfig
   return 65535;
};

__cuda_callable__
inline int Cuda::getMaxBlockSize()
{
   // TODO: make it preprocessor macro constant defined in tnlConfig
   return 1024;
};

__cuda_callable__ 
inline int Cuda::getWarpSize()
{
   // TODO: make it preprocessor macro constant defined in tnlConfig
   return 32;
}

#ifdef HAVE_CUDA
template< typename Index >
__device__ Index Cuda::getGlobalThreadIdx( const Index gridIdx )
{
   return ( gridIdx * Cuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
}
#endif


__cuda_callable__ 
inline int Cuda::getNumberOfSharedMemoryBanks()
{
   // TODO: make it preprocessor macro constant defined in tnlConfig
   return 32;
}

template< typename ObjectType >
ObjectType* Cuda::passToDevice( const ObjectType& object )
{
#ifdef HAVE_CUDA
   ObjectType* deviceObject;
   if( cudaMalloc( ( void** ) &deviceObject,
                   ( size_t ) sizeof( ObjectType ) ) != cudaSuccess )
   {
      checkCudaDevice;
      return 0;
   }
   if( cudaMemcpy( ( void* ) deviceObject,
                   ( void* ) &object,
                   sizeof( ObjectType ),
                   cudaMemcpyHostToDevice ) != cudaSuccess )
   {
      checkCudaDevice;
      cudaFree( deviceObject );
      return 0;
   }
   return deviceObject;
#else
   Assert( false, std::cerr << "CUDA support is missing." );
   return 0;
#endif
}

template< typename ObjectType >
ObjectType Cuda::passFromDevice( const ObjectType* object )
{
#ifdef HAVE_CUDA
   ObjectType aux;
   cudaMemcpy( ( void* ) aux,
               ( void* ) &object,
               sizeof( ObjectType ),
               cudaMemcpyDeviceToHost );
   checkCudaDevice;
   return aux;
#else
   Assert( false, std::cerr << "CUDA support is missing." );
   return 0;
#endif
}

template< typename ObjectType >
void Cuda::passFromDevice( const ObjectType* deviceObject,
                              ObjectType& hostObject )
{
#ifdef HAVE_CUDA
   cudaMemcpy( ( void* ) &hostObject,
               ( void* ) deviceObject,
               sizeof( ObjectType ),
               cudaMemcpyDeviceToHost );
   checkCudaDevice;
#else
   Assert( false, std::cerr << "CUDA support is missing." );
#endif
}

template< typename ObjectType >
void Cuda::print( const ObjectType* deviceObject, std::ostream& str )
{
#ifdef HAVE_CUDA
   ObjectType hostObject;
   passFromDevice( deviceObject, hostObject );
   str << hostObject;
#endif
}


template< typename ObjectType >
void Cuda::freeFromDevice( ObjectType* deviceObject )
{
#ifdef HAVE_CUDA
   cudaFree( deviceObject );
   checkCudaDevice;
#else
   Assert( false, std::cerr << "CUDA support is missing." );
#endif
}

#ifdef HAVE_CUDA
template< typename Index >
__device__ Index Cuda::getInterleaving( const Index index )
{
   return index + index / Cuda::getNumberOfSharedMemoryBanks();
}

template< typename Element >
__device__ getSharedMemory< Element >::operator Element*()
{
   extern __shared__ int __sharedMemory[];
   return ( Element* ) __sharedMemory;
};

__device__ inline getSharedMemory< double >::operator double*()
{
   extern __shared__ double __sharedMemoryDouble[];
   return ( double* ) __sharedMemoryDouble;
};

__device__ inline getSharedMemory< long int >::operator long int*()
{
   extern __shared__ long int __sharedMemoryLongInt[];
   return ( long int* ) __sharedMemoryLongInt;
};

#endif /* HAVE_CUDA */

} // namespace Devices
} // namespace TNL
