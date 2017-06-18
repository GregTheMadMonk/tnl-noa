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
   /***
    * This function is obsolete and should be replaced by the following functions.
    */
   __device__ static inline int
   getGlobalThreadIdx( const int gridIdx = 0,
                       const int gridSize = getMaxGridSize() );   

   __device__ static inline int
   getGlobalThreadIdx_x( const dim3& gridIdx );

   __device__ static inline int
   getGlobalThreadIdx_y( const dim3& gridIdx );

   __device__ static inline int
   getGlobalThreadIdx_z( const dim3& gridIdx );   
#endif

   /****
    * This functions helps to count number of CUDA blocks depending on the 
    * number of the CUDA threads and the block size.
    * It is obsolete and it will be replaced by setupThreads.
    */
   static int getNumberOfBlocks( const int threads,
                                 const int blockSize );

   /****
    * This functions helps to count number of CUDA grids depending on the 
    * number of the CUDA blocks and maximum grid size.
    * It is obsolete and it will be replaced by setupThreads.
    */
   static int getNumberOfGrids( const int blocks,
                                const int gridSize = getMaxGridSize() );
   
#ifdef HAVE_CUDA   
   /*! This method sets up gridSize and computes number of grids depending
    *  on total number of CUDA threads.
    */
   static void setupThreads( const dim3& blockSize,
                             dim3& blocksCount,
                             dim3& gridsCount,
                             long long int xThreads,
                             long long int yThreads = 0,
                             long long int zThreads = 0 );
   
   /*! This method sets up grid size when one iterates over more grids.
    * If gridIdx.? < gridsCount.? then the gridSize.? is set to maximum
    * allowed by CUDA. Otherwise gridSize.? is set to the size of the grid
    * in the last loop i.e. blocksCount.? % maxGridSize.?.
    */
   static void setupGrid( const dim3& blocksCount,
                          const dim3& gridsCount,
                          const dim3& gridIdx,
                          dim3& gridSize );
   
   static void printThreadsSetup( const dim3& blockSize,
                                  const dim3& blocksCount,
                                  const dim3& gridSize,
                                  const dim3& gridsCount,
                                  std::ostream& str = std::cout );
#endif   

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

#ifdef HAVE_CUDA
std::ostream& operator << ( std::ostream& str, const dim3& d );
#endif

} // namespace Devices
} // namespace TNL   
   
#include <TNL/Devices/Cuda_impl.h>
