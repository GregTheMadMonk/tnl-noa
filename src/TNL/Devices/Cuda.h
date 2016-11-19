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
#include <TNL/Timer.h>

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

   __cuda_callable__ static inline constexpr int getMaxGridSize();

   __cuda_callable__ static inline constexpr int getMaxBlockSize();

   __cuda_callable__ static inline constexpr int getWarpSize();

   __cuda_callable__ static inline constexpr int getNumberOfSharedMemoryBanks();

   static inline constexpr int getGPUTransferBufferSize();

#ifdef HAVE_CUDA
   __device__ static inline int
   getGlobalThreadIdx( const int gridIdx = 0,
                       const int gridSize = getMaxGridSize() );
#endif

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

   /****
    * Declaration of variables for dynamic shared memory is difficult in
    * templated functions. For example, the following does not work for
    * different types T:
    *
    *    template< typename T >
    *    void foo()
    *    {
    *        extern __shared__ T shx[];
    *    }
    *
    * This is because extern variables must be declared exactly once. In
    * templated functions we need to have same variable name with different
    * type, which causes the conflict. In CUDA samples they solve the problem
    * using template specialization via classes, but using one base type and
    * reinterpret_cast works too.
    * See http://stackoverflow.com/a/19339004/4180822 for reference.
    */
   template< typename Element, size_t Alignment = sizeof( Element ) >
   static __device__ Element* getSharedMemory();
#endif

#ifdef HAVE_CUDA
   /****
    * I do not know why, but it is more reliable to pass the error code instead
    * of calling cudaGetLastError() inside the method.
    * We recommend to use macro 'checkCudaDevice' defined bellow.
    */
   static bool checkDevice( const char* file_name, int line, cudaError error );
#else
   static bool checkDevice() { return false; };
#endif
   
   static void configSetup( Config::ConfigDescription& config, const String& prefix = "" );
      
   static bool setup( const Config::ParameterContainer& parameters,
                      const String& prefix = "" );
   
   static void insertSmartPointer( SmartPointer* pointer );
   
   static void removeSmartPointer( SmartPointer* pointer );
   
   // Negative deviceId means that CudaDeviceInfo::getActiveDevice will be
   // called to get the device ID.
   static bool synchronizeDevice( int deviceId = -1 );
   
   static Timer smartPointersSynchronizationTimer;
   
   protected:
   
   static SmartPointersRegister smartPointersRegister;
};

#ifdef HAVE_CUDA
#define checkCudaDevice ::TNL::Devices::Cuda::checkDevice( __FILE__, __LINE__, cudaGetLastError() )
#else
#define checkCudaDevice ::TNL::Devices::Cuda::checkDevice()
#endif

#define CudaSupportMissingMessage \
   std::cerr << "The CUDA support is missing in the source file " << __FILE__ << " at line " << __LINE__ << ". Please set WITH_CUDA=yes in the install script. " << std::endl;

} // namespace Devices
} // namespace TNL   
   
#include <TNL/Devices/Cuda_impl.h>
