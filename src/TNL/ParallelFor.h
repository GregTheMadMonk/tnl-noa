/***************************************************************************
                          ParallelFor.h  -  description
                             -------------------
    begin                : Mar 10, 2017
    copyright            : (C) 2017 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Devices/CudaDeviceInfo.h>
#include <TNL/Math.h>

/*
 * The implementation of ParallelFor is not meant to provide maximum performance
 * at every cost, but maximum flexibility for operating with data stored on the
 * device.
 *
 * The grid-stride loop for CUDA has been inspired by Nvidia's blog post:
 * https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
 *
 * Implemented by: Jakub Klinkovsky
 */

namespace TNL {

template< typename Device = Devices::Host >
struct ParallelFor
{
   template< typename Index,
             typename Function,
             typename... FunctionArgs >
   static void exec( Index start, Index end, Function f, FunctionArgs... args )
   {
#ifdef HAVE_OPENMP
      #pragma omp parallel for if( TNL::Devices::Host::isOMPEnabled() && end - start > 512 )
#endif
      for( Index i = start; i < end; i++ )
         f( i, args... );
   }
};

template< typename Device = Devices::Host >
struct ParallelFor2D
{
   template< typename Index,
             typename Function,
             typename... FunctionArgs >
   static void exec( Index startX, Index startY, Index endX, Index endY, Function f, FunctionArgs... args )
   {
#ifdef HAVE_OPENMP
      #pragma omp parallel for if( TNL::Devices::Host::isOMPEnabled() )
#endif
      for( Index i = startX; i < endX; i++ )
      for( Index j = startY; j < endY; j++ )
         f( i, j, args... );
   }
};

template< typename Device = Devices::Host >
struct ParallelFor3D
{
   template< typename Index,
             typename Function,
             typename... FunctionArgs >
   static void exec( Index startX, Index startY, Index startZ, Index endX, Index endY, Index endZ, Function f, FunctionArgs... args )
   {
#ifdef HAVE_OPENMP
      #pragma omp parallel for collapse(2) if( TNL::Devices::Host::isOMPEnabled() )
#endif
      for( Index i = startX; i < endX; i++ )
      for( Index j = startY; j < endY; j++ )
      for( Index k = startZ; k < endZ; k++ )
         f( i, j, k, args... );
   }
};

#ifdef HAVE_CUDA
template< bool gridStrideX = true,
          typename Index,
          typename Function,
          typename... FunctionArgs >
__global__ void
ParallelForKernel( Index start, Index end, Function f, FunctionArgs... args )
{
   Index i = start + blockIdx.x * blockDim.x + threadIdx.x;
   while( i < end ) {
      f( i, args... );
      if( gridStrideX ) i += blockDim.x * gridDim.x;
      else break;
   }
}

template< bool gridStrideX = true,
          bool gridStrideY = true,
          typename Index,
          typename Function,
          typename... FunctionArgs >
__global__ void
ParallelFor2DKernel( Index startX, Index startY, Index endX, Index endY, Function f, FunctionArgs... args )
{
   Index j = startY + blockIdx.y * blockDim.y + threadIdx.y;
   Index i = startX + blockIdx.x * blockDim.x + threadIdx.x;
   while( j < endY ) {
      while( i < endX ) {
         f( i, j, args... );
         if( gridStrideX ) i += blockDim.x * gridDim.x;
         else break;
      }
      if( gridStrideY ) j += blockDim.y * gridDim.y;
      else break;
   }
}

template< bool gridStrideX = true,
          bool gridStrideY = true,
          bool gridStrideZ = true,
          typename Index,
          typename Function,
          typename... FunctionArgs >
__global__ void
ParallelFor3DKernel( Index startX, Index startY, Index startZ, Index endX, Index endY, Index endZ, Function f, FunctionArgs... args )
{
   Index k = startZ + blockIdx.z * blockDim.z + threadIdx.z;
   Index j = startY + blockIdx.y * blockDim.y + threadIdx.y;
   Index i = startX + blockIdx.x * blockDim.x + threadIdx.x;
   while( k < endZ ) {
      while( j < endY ) {
         while( i < endX ) {
            f( i, j, k, args... );
            if( gridStrideX ) i += blockDim.x * gridDim.x;
            else break;
         }
         if( gridStrideY ) j += blockDim.y * gridDim.y;
         else break;
      }
      if( gridStrideZ ) k += blockDim.z * gridDim.z;
      else break;
   }
}
#endif

template<>
struct ParallelFor< Devices::Cuda >
{
   template< typename Index,
             typename Function,
             typename... FunctionArgs >
   static void exec( Index start, Index end, Function f, FunctionArgs... args )
   {
#ifdef HAVE_CUDA
      if( end > start ) {
         dim3 blockSize( 256 );
         dim3 gridSize;
         gridSize.x = TNL::min( Devices::Cuda::getMaxGridSize(), Devices::Cuda::getNumberOfBlocks( end - start, blockSize.x ) );

         if( Devices::Cuda::getNumberOfGrids( end - start ) == 1 )
            ParallelForKernel< false ><<< gridSize, blockSize >>>( start, end, f, args... );
         else {
            // decrease the grid size and align to the number of multiprocessors
            const int desGridSize = 32 * Devices::CudaDeviceInfo::getCudaMultiprocessors( Devices::CudaDeviceInfo::getActiveDevice() );
            gridSize.x = TNL::min( desGridSize, Devices::Cuda::getNumberOfBlocks( end - start, blockSize.x ) );
            ParallelForKernel< true ><<< gridSize, blockSize >>>( start, end, f, args... );
         }

         cudaDeviceSynchronize();
         TNL_CHECK_CUDA_DEVICE;
      }
#else
      throw Exceptions::CudaSupportMissing();
#endif
   }
};

template<>
struct ParallelFor2D< Devices::Cuda >
{
   template< typename Index,
             typename Function,
             typename... FunctionArgs >
   static void exec( Index startX, Index startY, Index endX, Index endY, Function f, FunctionArgs... args )
   {
#ifdef HAVE_CUDA
      if( endX > startX && endY > startY ) {
         const Index sizeX = endX - startX;
         const Index sizeY = endY - startY;

         dim3 blockSize;
         if( sizeX >= sizeY * sizeY ) {
            blockSize.x = TNL::min( 256, sizeX );
            blockSize.y = 1;
         }
         else if( sizeY >= sizeX * sizeX ) {
            blockSize.x = 1;
            blockSize.y = TNL::min( 256, sizeY );
         }
         else {
            blockSize.x = TNL::min( 32, sizeX );
            blockSize.y = TNL::min( 8, sizeY );
         }
         dim3 gridSize;
         gridSize.x = TNL::min( Devices::Cuda::getMaxGridSize(), Devices::Cuda::getNumberOfBlocks( sizeX, blockSize.x ) );
         gridSize.y = TNL::min( Devices::Cuda::getMaxGridSize(), Devices::Cuda::getNumberOfBlocks( sizeY, blockSize.y ) );

         dim3 gridCount;
         gridCount.x = Devices::Cuda::getNumberOfGrids( sizeX );
         gridCount.y = Devices::Cuda::getNumberOfGrids( sizeY );

         if( gridCount.x == 1 && gridCount.y == 1 )
            ParallelFor2DKernel< false, false ><<< gridSize, blockSize >>>
               ( startX, startY, endX, endY, f, args... );
         else if( gridCount.x == 1 && gridCount.y > 1 )
            ParallelFor2DKernel< false, true ><<< gridSize, blockSize >>>
               ( startX, startY, endX, endY, f, args... );
         else if( gridCount.x > 1 && gridCount.y == 1 )
            ParallelFor2DKernel< true, false ><<< gridSize, blockSize >>>
               ( startX, startY, endX, endY, f, args... );
         else
            ParallelFor2DKernel< true, true ><<< gridSize, blockSize >>>
               ( startX, startY, endX, endY, f, args... );

         cudaDeviceSynchronize();
         TNL_CHECK_CUDA_DEVICE;
      }
#else
      throw Exceptions::CudaSupportMissing();
#endif
   }
};

template<>
struct ParallelFor3D< Devices::Cuda >
{
   template< typename Index,
             typename Function,
             typename... FunctionArgs >
   static void exec( Index startX, Index startY, Index startZ, Index endX, Index endY, Index endZ, Function f, FunctionArgs... args )
   {
#ifdef HAVE_CUDA
      if( endX > startX && endY > startY && endZ > startZ ) {
         const Index sizeX = endX - startX;
         const Index sizeY = endY - startY;
         const Index sizeZ = endZ - startZ;

         dim3 blockSize;
         if( sizeX >= sizeY * sizeZ ) {
            blockSize.x = TNL::min( 256, sizeX );
            blockSize.y = 1;
            blockSize.z = 1;
         }
         else if( sizeY >= sizeX * sizeZ ) {
            blockSize.x = 1;
            blockSize.y = TNL::min( 256, sizeY );
            blockSize.z = 1;
         }
         else if( sizeZ >= sizeX * sizeY ) {
            blockSize.x = 1;
            blockSize.y = 1;
            blockSize.z = TNL::min( 256, sizeZ );
         }
         else {
            blockSize.x = TNL::min( 16, sizeX );
            blockSize.y = TNL::min( 4, sizeY );
            blockSize.z = TNL::min( 4, sizeZ );
         }
         dim3 gridSize;
         gridSize.x = TNL::min( Devices::Cuda::getMaxGridSize(), Devices::Cuda::getNumberOfBlocks( sizeX, blockSize.x ) );
         gridSize.y = TNL::min( Devices::Cuda::getMaxGridSize(), Devices::Cuda::getNumberOfBlocks( sizeY, blockSize.y ) );
         gridSize.z = TNL::min( Devices::Cuda::getMaxGridSize(), Devices::Cuda::getNumberOfBlocks( sizeZ, blockSize.z ) );

         dim3 gridCount;
         gridCount.x = Devices::Cuda::getNumberOfGrids( sizeX );
         gridCount.y = Devices::Cuda::getNumberOfGrids( sizeY );
         gridCount.z = Devices::Cuda::getNumberOfGrids( sizeZ );

         if( gridCount.x == 1 && gridCount.y == 1 && gridCount.z == 1 )
            ParallelFor3DKernel< false, false, false ><<< gridSize, blockSize >>>
               ( startX, startY, startZ, endX, endY, endZ, f, args... );
         else if( gridCount.x == 1 && gridCount.y == 1 && gridCount.z > 1 )
            ParallelFor3DKernel< false, false, true ><<< gridSize, blockSize >>>
               ( startX, startY, startZ, endX, endY, endZ, f, args... );
         else if( gridCount.x == 1 && gridCount.y > 1 && gridCount.z == 1 )
            ParallelFor3DKernel< false, true, false ><<< gridSize, blockSize >>>
               ( startX, startY, startZ, endX, endY, endZ, f, args... );
         else if( gridCount.x > 1 && gridCount.y == 1 && gridCount.z == 1 )
            ParallelFor3DKernel< true, false, false ><<< gridSize, blockSize >>>
               ( startX, startY, startZ, endX, endY, endZ, f, args... );
         else if( gridCount.x == 1 && gridCount.y > 1 && gridCount.z > 1 )
            ParallelFor3DKernel< false, true, true ><<< gridSize, blockSize >>>
               ( startX, startY, startZ, endX, endY, endZ, f, args... );
         else if( gridCount.x > 1 && gridCount.y > 1 && gridCount.z == 1 )
            ParallelFor3DKernel< true, true, false ><<< gridSize, blockSize >>>
               ( startX, startY, startZ, endX, endY, endZ, f, args... );
         else if( gridCount.x > 1 && gridCount.y == 1 && gridCount.z > 1 )
            ParallelFor3DKernel< true, false, true ><<< gridSize, blockSize >>>
               ( startX, startY, startZ, endX, endY, endZ, f, args... );
         else
            ParallelFor3DKernel< true, true, true ><<< gridSize, blockSize >>>
               ( startX, startY, startZ, endX, endY, endZ, f, args... );

         cudaDeviceSynchronize();
         TNL_CHECK_CUDA_DEVICE;
      }
#else
      throw Exceptions::CudaSupportMissing();
#endif
   }
};

} // namespace TNL
