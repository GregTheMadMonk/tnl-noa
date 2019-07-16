/***************************************************************************
                          Cuda.h  -  description
                             -------------------
    begin                : Nov 7, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <iostream>

#include <TNL/String.h>
#include <TNL/Assert.h>
#include <TNL/Pointers/SmartPointersRegister.h>
#include <TNL/Timer.h>
#include <TNL/Devices/CudaCallable.h>
#include <TNL/Config/ConfigDescription.h>
#include <TNL/Config/ParameterContainer.h>

namespace TNL {
namespace Devices {

class Cuda
{
   public:

   static inline String getDeviceType();

   // TODO: Remove getDeviceType();
   static inline String getType() { return getDeviceType();};

   static inline void configSetup( Config::ConfigDescription& config, const String& prefix = "" );

   static inline bool setup( const Config::ParameterContainer& parameters,
                             const String& prefix = "" );

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
   static inline int getNumberOfBlocks( const int threads,
                                        const int blockSize );

   /****
    * This functions helps to count number of CUDA grids depending on the 
    * number of the CUDA threads and maximum grid size.
    * It is obsolete and it will be replaced by setupThreads.
    */
   static inline int getNumberOfGrids( const int threads,
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
   template< typename Element >
   static __device__ Element* getSharedMemory();
#endif

#ifdef HAVE_CUDA
   /****
    * I do not know why, but it is more reliable to pass the error code instead
    * of calling cudaGetLastError() inside the method.
    * We recommend to use macro 'TNL_CHECK_CUDA_DEVICE' defined bellow.
    */
   static inline void checkDevice( const char* file_name, int line, cudaError error );
#else
   static inline void checkDevice() {}
#endif

   static inline void insertSmartPointer( Pointers::SmartPointer* pointer );

   static inline void removeSmartPointer( Pointers::SmartPointer* pointer );

   // Negative deviceId means that CudaDeviceInfo::getActiveDevice will be
   // called to get the device ID.
   static inline bool synchronizeDevice( int deviceId = -1 );

   static inline Timer& getSmartPointersSynchronizationTimer();

   ////
   // When we transfer data between the GPU and the CPU we use 5 MB buffer. This
   // size should ensure good performance -- see.
   // http://wiki.accelereyes.com/wiki/index.php/GPU_Memory_Transfer .
   // We use the same buffer size even for retyping data during IO operations.
   //
   static constexpr std::size_t TransferBufferSize = 5 * 2<<20;


   protected:

   static inline Pointers::SmartPointersRegister& getSmartPointersRegister();
};

#ifdef HAVE_CUDA
#define TNL_CHECK_CUDA_DEVICE ::TNL::Devices::Cuda::checkDevice( __FILE__, __LINE__, cudaGetLastError() )
#else
#define TNL_CHECK_CUDA_DEVICE ::TNL::Devices::Cuda::checkDevice()
#endif

#ifdef HAVE_CUDA
namespace {
   std::ostream& operator << ( std::ostream& str, const dim3& d );
}
#endif

#ifdef HAVE_CUDA
#if __CUDA_ARCH__ < 600
namespace {
   __device__ double atomicAdd(double* address, double val);
}
#endif
#endif

} // namespace Devices
} // namespace TNL

#include <TNL/Devices/Cuda_impl.h>
