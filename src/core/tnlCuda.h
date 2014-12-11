/***************************************************************************
                          tnlCuda.h  -  description
                             -------------------
    begin                : Nov 7, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
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

#ifndef TNLCUDA_H_
#define TNLCUDA_H_

#include <iostream>
#include <unistd.h>
#include <core/tnlDevice.h>
#include <core/tnlString.h>
#include <core/tnlAssert.h>

class tnlCuda
{
   public:

   enum { DeviceType = tnlCudaDevice };

   static tnlString getDeviceType();

#ifdef HAVE_CUDA
   __host__ __device__
#endif
   static inline tnlDeviceEnum getDevice();


#ifdef HAVE_CUDA
   __host__ __device__
#endif
   static inline int getMaxGridSize();

#ifdef HAVE_CUDA
   __host__ __device__
#endif
   static inline int getMaxBlockSize();

#ifdef HAVE_CUDA
   __host__ __device__
#endif
   static inline int getWarpSize();

#ifdef HAVE_CUDA
   template< typename Index >
   __device__ static Index getGlobalThreadIdx( const Index gridIdx = 0 );
#endif

#ifdef HAVE_CUDA
   __host__ __device__
#endif
   static inline int getNumberOfSharedMemoryBanks();

   static int getGPUTransferBufferSize();

   static int getNumberOfBlocks( const int threads,
                                 const int blockSize );

   static int getNumberOfGrids( const int blocks,
                                const int gridSize = getMaxGridSize() );

   static size_t getFreeMemory();

   template< typename ObjectType >
   static ObjectType* passToDevice( const ObjectType& object );

   template< typename ObjectType >
   static ObjectType passFromDevice( const ObjectType* object );

   template< typename ObjectType >
   static void passFromDevice( const ObjectType* deviceObject,
                               ObjectType& hostObject );

   template< typename ObjectType >
   static void freeFromDevice( ObjectType* object );

   template< typename ObjectType >
   static void print( const ObjectType* object, ostream& str = std::cout );

#ifdef HAVE_CUDA
   template< typename Index >
   static __device__ Index getInterleaving( const Index index );
#endif

   static bool checkDevice( const char* file_name, int line );

};

#define checkCudaDevice tnlCuda::checkDevice( __FILE__, __LINE__ )

#define tnlCudaSupportMissingMessage \
   std::cerr << "The CUDA support is missing in the source file " << __FILE__ << " at line " << __LINE__ << ". Please set WITH_CUDA=yes in the install script. " << std::endl;


// TODO: This would be nice in tnlCuda but C++ standard does not allow it.
#ifdef HAVE_CUDA
   template< typename Element >
   struct getSharedMemory
   {
       __device__ operator Element*();
   };

   template<>
   struct getSharedMemory< double >
   {
       inline __device__ operator double*();
   };

   template<>
   struct getSharedMemory< long int >
   {
       inline __device__ operator long int*();
   };

#endif

#include <implementation/core/tnlCuda_impl.h>

#endif /* TNLCUDA_H_ */
