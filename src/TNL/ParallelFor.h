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

#ifdef HAVE_CUDA
template< typename Index,
          typename Function,
          typename... FunctionArgs >
__global__ void
ParallelForKernel( Index start, Index end, Function f, FunctionArgs... args )
{
   for( Index i = start + blockIdx.x * blockDim.x + threadIdx.x;
        i < end;
        i += blockDim.x * gridDim.x )
   {
      f( i, args... );
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
         const int desGridSize = 32 * Devices::CudaDeviceInfo::getCudaMultiprocessors( Devices::CudaDeviceInfo::getActiveDevice() );
         gridSize.x = TNL::min( desGridSize, Devices::Cuda::getNumberOfBlocks( end - start, blockSize.x ) );

         Devices::Cuda::synchronizeDevice();
         ParallelForKernel<<< gridSize, blockSize >>>( start, end, f, args... );
         cudaDeviceSynchronize();
         TNL_CHECK_CUDA_DEVICE;
      }
#endif
   }
};

} // namespace TNL
