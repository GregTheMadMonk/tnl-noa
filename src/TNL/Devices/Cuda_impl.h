/***************************************************************************
                          Cuda_impl.h  -  description
                             -------------------
    begin                : Jan 21, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Math.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Devices/CudaDeviceInfo.h>
#include <TNL/Exceptions/CudaBadAlloc.h>
#include <TNL/Exceptions/CudaSupportMissing.h>
#include <TNL/Exceptions/CudaRuntimeError.h>
#include <TNL/CudaSharedMemory.h>
#include <TNL/Config/ConfigDescription.h>
#include <TNL/Config/ParameterContainer.h>

namespace TNL {
namespace Devices {

inline String Cuda::getDeviceType()
{
   return String( "Cuda" );
}

inline void
Cuda::configSetup( Config::ConfigDescription& config,
                   const String& prefix )
{
#ifdef HAVE_CUDA
   config.addEntry< int >( prefix + "cuda-device", "Choose CUDA device to run the computation.", 0 );
#else
   config.addEntry< int >( prefix + "cuda-device", "Choose CUDA device to run the computation (not supported on this system).", 0 );
#endif
}

inline bool
Cuda::setup( const Config::ParameterContainer& parameters,
             const String& prefix )
{
#ifdef HAVE_CUDA
   int cudaDevice = parameters.getParameter< int >( prefix + "cuda-device" );
   if( cudaSetDevice( cudaDevice ) != cudaSuccess )
   {
      std::cerr << "I cannot activate CUDA device number " << cudaDevice << "." << std::endl;
      return false;
   }
   getSmartPointersSynchronizationTimer().reset();
   getSmartPointersSynchronizationTimer().stop();
#endif
   return true;
}

__cuda_callable__
inline constexpr int Cuda::getMaxGridSize()
{
   return 65535;
}

__cuda_callable__
inline constexpr int Cuda::getMaxBlockSize()
{
   return 1024;
}

__cuda_callable__
inline constexpr int Cuda::getWarpSize()
{
   return 32;
}

__cuda_callable__
inline constexpr int Cuda::getNumberOfSharedMemoryBanks()
{
   return 32;
}

inline constexpr int Cuda::getGPUTransferBufferSize()
{
   return 1 << 20;
}

#ifdef HAVE_CUDA
__device__ inline int Cuda::getGlobalThreadIdx( const int gridIdx, const int gridSize )
{
   return ( gridIdx * gridSize + blockIdx.x ) * blockDim.x + threadIdx.x;
}

__device__ inline int Cuda::getGlobalThreadIdx_x( const dim3& gridIdx )
{
   return ( gridIdx.x * getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
}

__device__ inline int Cuda::getGlobalThreadIdx_y( const dim3& gridIdx )
{
   return ( gridIdx.y * getMaxGridSize() + blockIdx.y ) * blockDim.y + threadIdx.y;
}

__device__ inline int Cuda::getGlobalThreadIdx_z( const dim3& gridIdx )
{
   return ( gridIdx.z * getMaxGridSize() + blockIdx.z ) * blockDim.z + threadIdx.z;
}
#endif

inline int Cuda::getNumberOfBlocks( const int threads,
                                    const int blockSize )
{
   return roundUpDivision( threads, blockSize );
}

inline int Cuda::getNumberOfGrids( const int blocks,
                                   const int gridSize )
{
   return roundUpDivision( blocks, gridSize );
}

#ifdef HAVE_CUDA
inline void Cuda::setupThreads( const dim3& blockSize,
                                dim3& blocksCount,
                                dim3& gridsCount,
                                long long int xThreads,
                                long long int yThreads,
                                long long int zThreads )
{
   blocksCount.x = max( 1, xThreads / blockSize.x + ( xThreads % blockSize.x != 0 ) );
   blocksCount.y = max( 1, yThreads / blockSize.y + ( yThreads % blockSize.y != 0 ) );
   blocksCount.z = max( 1, zThreads / blockSize.z + ( zThreads % blockSize.z != 0 ) );
   
   /****
    * TODO: Fix the following:
    * I do not known how to get max grid size in kernels :(
    * 
    * Also, this is very slow. */
   /*int currentDevice( 0 );
   cudaGetDevice( currentDevice );
   cudaDeviceProp properties;
   cudaGetDeviceProperties( &properties, currentDevice );
   gridsCount.x = blocksCount.x / properties.maxGridSize[ 0 ] + ( blocksCount.x % properties.maxGridSize[ 0 ] != 0 );
   gridsCount.y = blocksCount.y / properties.maxGridSize[ 1 ] + ( blocksCount.y % properties.maxGridSize[ 1 ] != 0 );
   gridsCount.z = blocksCount.z / properties.maxGridSize[ 2 ] + ( blocksCount.z % properties.maxGridSize[ 2 ] != 0 );
   */
   gridsCount.x = blocksCount.x / getMaxGridSize() + ( blocksCount.x % getMaxGridSize() != 0 );
   gridsCount.y = blocksCount.y / getMaxGridSize() + ( blocksCount.y % getMaxGridSize() != 0 );
   gridsCount.z = blocksCount.z / getMaxGridSize() + ( blocksCount.z % getMaxGridSize() != 0 );
}

inline void Cuda::setupGrid( const dim3& blocksCount,
                             const dim3& gridsCount,
                             const dim3& gridIdx,
                             dim3& gridSize )
{
   /* TODO: this is extremely slow!!!!
   int currentDevice( 0 );
   cudaGetDevice( &currentDevice );
   cudaDeviceProp properties;
   cudaGetDeviceProperties( &properties, currentDevice );*/
 
   /****
    * TODO: fix the following
   if( gridIdx.x < gridsCount.x )
      gridSize.x = properties.maxGridSize[ 0 ];
   else
      gridSize.x = blocksCount.x % properties.maxGridSize[ 0 ];
   
   if( gridIdx.y < gridsCount.y )
      gridSize.y = properties.maxGridSize[ 1 ];
   else
      gridSize.y = blocksCount.y % properties.maxGridSize[ 1 ];

   if( gridIdx.z < gridsCount.z )
      gridSize.z = properties.maxGridSize[ 2 ];
   else
      gridSize.z = blocksCount.z % properties.maxGridSize[ 2 ];*/
   
   if( gridIdx.x < gridsCount.x - 1 )
      gridSize.x = getMaxGridSize();
   else
      gridSize.x = blocksCount.x % getMaxGridSize();
   
   if( gridIdx.y < gridsCount.y - 1 )
      gridSize.y = getMaxGridSize();
   else
      gridSize.y = blocksCount.y % getMaxGridSize();

   if( gridIdx.z < gridsCount.z - 1 )
      gridSize.z = getMaxGridSize();
   else
      gridSize.z = blocksCount.z % getMaxGridSize();
}

inline void Cuda::printThreadsSetup( const dim3& blockSize,
                                     const dim3& blocksCount,
                                     const dim3& gridSize,
                                     const dim3& gridsCount,
                                     std::ostream& str )
{
   str << "Block size: " << blockSize << std::endl
       << " Blocks count: " << blocksCount << std::endl
       << " Grid size: " << gridSize << std::endl
       << " Grids count: " << gridsCount << std::endl;
}
#endif


template< typename ObjectType >
ObjectType* Cuda::passToDevice( const ObjectType& object )
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
ObjectType Cuda::passFromDevice( const ObjectType* object )
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
void Cuda::passFromDevice( const ObjectType* deviceObject,
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
   cudaFree( ( void* ) deviceObject );
   TNL_CHECK_CUDA_DEVICE;
#else
   throw Exceptions::CudaSupportMissing();
#endif
}

#ifdef HAVE_CUDA
template< typename Index >
__device__ Index Cuda::getInterleaving( const Index index )
{
   return index + index / Cuda::getNumberOfSharedMemoryBanks();
}

template< typename Element >
__device__ Element* Cuda::getSharedMemory()
{
   return CudaSharedMemory< Element >();
}

#ifdef HAVE_CUDA
inline void Cuda::checkDevice( const char* file_name, int line, cudaError error )
{
   if( error != cudaSuccess )
      throw Exceptions::CudaRuntimeError( error, file_name, line );
}
#endif

inline void Cuda::insertSmartPointer( SmartPointer* pointer )
{
   getSmartPointersRegister().insert( pointer, Devices::CudaDeviceInfo::getActiveDevice() );
}

inline void Cuda::removeSmartPointer( SmartPointer* pointer )
{
   getSmartPointersRegister().remove( pointer, Devices::CudaDeviceInfo::getActiveDevice() );
}

inline bool Cuda::synchronizeDevice( int deviceId )
{
#ifdef HAVE_CUDA_UNIFIED_MEMORY
   return true;
#else
   if( deviceId < 0 )
      deviceId = Devices::CudaDeviceInfo::getActiveDevice();
   getSmartPointersSynchronizationTimer().start();
   bool b = getSmartPointersRegister().synchronizeDevice( deviceId );
   getSmartPointersSynchronizationTimer().stop();
   return b;
#endif
}

inline Timer& Cuda::getSmartPointersSynchronizationTimer()
{
   static Timer timer;
   return timer;
}

inline SmartPointersRegister& Cuda::getSmartPointersRegister()
{
   static SmartPointersRegister reg;
   return reg;
}

#ifdef HAVE_CUDA
namespace {
   std::ostream& operator << ( std::ostream& str, const dim3& d )
   {
      str << "( " << d.x << ", " << d.y << ", " << d.z << " )";
      return str;
   }
}
#endif

// double-precision atomicAdd function for Maxwell and older GPUs
// copied from: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
#if __CUDA_ARCH__ < 600
namespace {
   __device__ double atomicAdd(double* address, double val)
   {
       unsigned long long int* address_as_ull =
                                 (unsigned long long int*)address;
       unsigned long long int old = *address_as_ull, assumed;

       do {
           assumed = old;
           old = atomicCAS(address_as_ull, assumed,
                           __double_as_longlong(val +
                                  __longlong_as_double(assumed)));

       // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
       } while (assumed != old);

       return __longlong_as_double(old);
   }
} // namespace
#endif

#endif /* HAVE_CUDA */

} // namespace Devices
} // namespace TNL
