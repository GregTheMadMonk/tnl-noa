/***************************************************************************
                          Devices::Cuda.h  -  description
                             -------------------
    begin                : Nov 7, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <iostream>
#include <unistd.h>
#include <TNL/String.h>
#include <TNL/Assert.h>
#include <TNL/SmartPointersRegister.h>

namespace TNL {

namespace Config { 
   class ConfigDescription;
   class ParameterContainer;
}

namespace Devices {

#ifdef HAVE_CUDA
#define __cuda_callable__ __device__ __host__
#else
#define __cuda_callable__
#endif


class Cuda
{
   public:

   static String getDeviceType();

   __cuda_callable__ static inline int getMaxGridSize();

   __cuda_callable__ static inline int getMaxBlockSize();

   __cuda_callable__ static inline int getWarpSize();

#ifdef HAVE_CUDA
   static int getDeviceId();
   
   template< typename Index >
   __device__ static Index getGlobalThreadIdx( const Index gridIdx = 0 );
#endif

   __cuda_callable__ static inline int getNumberOfSharedMemoryBanks();

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
   static void print( const ObjectType* object, std::ostream& str = std::cout );

#ifdef HAVE_CUDA
   template< typename Index >
   static __device__ Index getInterleaving( const Index index );
#endif

#ifdef HAVE_CUDA
   /****
    * I do not know why, but it is more reliable to pass the error code instead
    * of calling cudaGetLastError() inside the method.
    * We recommend to use macro 'checkCudaDevice' defined bellow.
    */
   static bool checkDevice( const char* file_name, int line, cudaError error );
#else
   static bool checkDevice() { return false;};
#endif
   
   static void configSetup( Config::ConfigDescription& config, const String& prefix = "" );
      
   static bool setup( const Config::ParameterContainer& parameters,
                      const String& prefix = "" );
   
   static void insertSmartPointer( SmartPointer* pointer );
   
   static void removeSmartPointer( SmartPointer* pointer );
   
   static bool synchronizeDevice( int deviceId = 0  );
   
   protected:
   
      static SmartPointersRegister smartPointersRegister;


};

#ifdef HAVE_CUDA
#define checkCudaDevice TNL::Devices::Cuda::checkDevice( __FILE__, __LINE__, cudaGetLastError() )
#else
#define checkCudaDevice TNL::Devices::Cuda::checkDevice()
#endif

#define CudaSupportMissingMessage \
   std::cerr << "The CUDA support is missing in the source file " << __FILE__ << " at line " << __LINE__ << ". Please set WITH_CUDA=yes in the install script. " << std::endl;


// TODO: This would be nice in Cuda but C++ standard does not allow it.
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

} // namespace Devices
} // namespace TNL   
   
#include <TNL/Devices/Cuda_impl.h>
