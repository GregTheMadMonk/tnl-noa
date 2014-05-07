/***************************************************************************
                          tnlCuda_impl.h  -  description
                             -------------------
    begin                : Jan 21, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef TNLCUDA_IMPL_H_
#define TNLCUDA_IMPL_H_

#ifdef HAVE_CUDA
__host__ __device__
#endif
inline tnlDeviceEnum tnlCuda::getDevice()
{
   return tnlCudaDevice;
};

#ifdef HAVE_CUDA
__host__ __device__
#endif
inline int tnlCuda::getMaxGridSize()
{
   // TODO: make it preprocessor macro constant defined in tnlConfig
   return 65536;
};

#ifdef HAVE_CUDA
__host__ __device__
#endif
inline int tnlCuda::getMaxBlockSize()
{
   // TODO: make it preprocessor macro constant defined in tnlConfig
   return 1024;
};

#ifdef HAVE_CUDA
__host__ __device__
#endif
inline int tnlCuda::getWarpSize()
{
   // TODO: make it preprocessor macro constant defined in tnlConfig
   return 32;
}

#ifdef HAVE_CUDA
__host__ __device__
#endif
inline int tnlCuda::getNumberOfSharedMemoryBanks()
{
   // TODO: make it preprocessor macro constant defined in tnlConfig
   return 32;
}


template< typename ObjectType >
ObjectType* tnlCuda::passToDevice( const ObjectType& object )
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
#endif
}

template< typename ObjectType >
void tnlCuda::freeFromDevice( ObjectType* deviceObject )
{
#ifdef HAVE_CUDA
   cudaFree( deviceObject );
   checkCudaDevice;
#endif
}

#ifdef HAVE_CUDA

template< typename Index >
__device__ Index tnlCuda::getInterleaving( const Index index )
{
   return index + index / tnlCuda::getNumberOfSharedMemoryBanks();
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
#endif


#endif /* TNLCUDA_IMPL_H_ */
