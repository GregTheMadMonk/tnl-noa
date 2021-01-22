/***************************************************************************
                          CSRKernels.h -  description
                             -------------------
    begin                : Jan 20, 2021 -> Joe Biden inauguration
    copyright            : (C) 2021 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Cuda/LaunchHelpers.h>
#include <TNL/Containers/VectorView.h>
#include <TNL/Algorithms/Segments/details/LambdaAdapter.h>

namespace TNL {
   namespace Algorithms {
      namespace Segments {
         namespace details {


#ifdef HAVE_CUDA
template< typename Device,
          typename Index,
          typename Fetch,
          typename Reduction,
          typename ResultKeeper,
          typename Real,
          typename... Args >
__global__
void RowsReductionVectorKernel(
    int gridIdx,
    const TNL::Containers::VectorView< Index, TNL::Devices::Cuda, Index > offsets,
    Index first,
    Index last,
    Fetch& fetch,
    const Reduction& reduction,
    ResultKeeper& keeper,
    const Real& zero,
    Args... args )
{
    /***
     * We map one warp to each segment
     */
    const Index segmentIdx =  TNL::Cuda::getGlobalThreadIdx( gridIdx ) / TNL::Cuda::getWarpSize() + first;
    if( segmentIdx >= last )
        return;

    const int laneIdx = threadIdx.x & 31; // & is cheaper than %
    Index endIdx = offsets[ segmentIdx + 1] ;

    Index localIdx( laneIdx );
    Real aux = zero;
    for( Index globalIdx = offsets[ segmentIdx ] + localIdx; i < endIdx; i += TNL::Cuda::getWarpSize() )
    {
      aux = reduce( aux, details::FetchLambdaAdapter< IndexType, Fetch >::call( fetch, segmentIdx, localIdx, globalIdx, compute ) );
      localIdx += TNL::Cuda::getWarpSize();
    }

   /****
    * Reduction in each warp which means in each segment.
    */
   aux += __shfl_down_sync(0xFFFFFFFF, aux, 16);
   aux += __shfl_down_sync(0xFFFFFFFF, aux, 8);
   aux += __shfl_down_sync(0xFFFFFFFF, aux, 4);
   aux += __shfl_down_sync(0xFFFFFFFF, aux, 2);
   aux += __shfl_down_sync(0xFFFFFFFF, aux, 1);

   if( laneIdx == 0 )
    keeper( segmentIdx, aux )



    /*const Index warpID = ((gridID * MAX_X_DIM) + (blockIdx.x * blockDim.x) + threadIdx.x) / warpSize;
    if (warpID >= rows)
      return;

   Real result = 0.0;
   const Index laneID = threadIdx.x & 31; // & is cheaper than %
   Index endID = rowPointers[warpID + 1];

   // Calculate result 
   for (Index i = rowPointers[warpID] + laneID; i < endID; i += warpSize)
      result += values[i] * inVector[columnIndexes[i]];

   // Reduction 
   result += __shfl_down_sync(0xFFFFFFFF, result, 16);
   result += __shfl_down_sync(0xFFFFFFFF, result, 8);
   result += __shfl_down_sync(0xFFFFFFFF, result, 4);
   result += __shfl_down_sync(0xFFFFFFFF, result, 2);
   result += __shfl_down_sync(0xFFFFFFFF, result, 1);
   // Write result
   if (laneID == 0) outVector[warpID] = result;*/
}
#endif

template< typename OffsetsView,
          typename Index,
          typename Fetch,
          typename Reduction,
          typename ResultKeeper,
          typename Real,
          typename... Args >
void
RowsReductionVectorKernelCaller(
    const OffsetsView& offsets,
    Index first,
    Index last,
    Fetch& fetch,
    const Reduction& reduction,
    ResultKeeper& keeper,
    const Real& zero,
    Args... args )
{
#ifdef HAVE_CUDA
    const Index warpsCount = last - first;
    const size_t threadsCount = warpsCount * TNL::Cuda::getWarpSize();
    dim3 blocksCount, gridsCount, blockSize( 256 );
    TNL::Cuda::setupThreads( blockSize, blocksCount, gridsCount, threadsCount );
    for( int gridIdx = 0; gridIdx < gridsCount.x; gridIdx ++ )
    {
        dim3 gridSize;
        setupGrid( blocksCount, gridsCount, gridIdx, gridSize );
        SpMVCSRVector< Index, Fetch, Redcution, ResultKeeper, Real, Args ><<< gridSize, blockSize >>>(
            gridIdx.x, offsets, first, last, fetch, reduction, keeper, zero, args );
    };

#endif

/*const Index threads = matrix.THREADS_VECTOR; // block size
   size_t neededThreads = matrix.getRowPointers().getSize() * warpSize;
   Index blocks;
   // Execute kernels on device 
   for (Index grid = 0; neededThreads != 0; ++grid) {
      if (MAX_X_DIM * threads >= neededThreads) {
         blocks = roundUpDivision(neededThreads, threads);
         neededThreads = 0;
      } else {
         blocks = MAX_X_DIM;
         neededThreads -= MAX_X_DIM * threads;
      }

      SpMVCSRVector<Real, Index, warpSize><<<blocks, threads>>>(
               inVector,
               outVector,
               matrix.getRowPointers().getData(),
               matrix.getColumnIndexes().getData(),
               matrix.getValues().getData(),
               matrix.getRowPointers().getSize() - 1,
               grid
      );
   }*/
}

#ifdef HAVE_CUDA
template< int ThreadsPerSegment,
          typename Device,
          typename Index,
          typename Fetch,
          typename Reduction,
          typename ResultKeeper,
          typename Real,
          typename... Args >
__global__
void RowsReductionLightKernel(
    int gridIdx,
    const TNL::Containers::VectorView< Index, TNL::Devices::Cuda, Index > offsets,
    Index first,
    Index last,
    Fetch& fetch,
    const Reduction& reduction,
    ResultKeeper& keeper,
    const Real& zero,
    Args... args )
{
    /***
     * We map one warp to each segment
     */
    const Index segmentIdx =  TNL::Cuda::getGlobalThreadIdx( gridIdx ) / TNL::Cuda::getWarpSize() + first;
    if( segmentIdx >= last )
        return;

    const int laneIdx = threadIdx.x & ( ThreadsPerSegment - 1 ); // & is cheaper than %
    Index endIdx = offsets[ segmentIdx + 1] ;

    Index localIdx( laneIdx );
    Real aux = zero;
    for( Index globalIdx = offsets[ segmentIdx ] + localIdx; i < endIdx; i += ThreadsPerSegment )
    {
      aux = reduce( aux, details::FetchLambdaAdapter< IndexType, Fetch >::call( fetch, segmentIdx, localIdx, globalIdx, compute ) );
      localIdx += TNL::Cuda::getWarpSize();
    }

    /****
     * Reduction in each segment.
     */
    if( ThreadsPerSegment == 32 )
        aux += __shfl_down_sync(0xFFFFFFFF, aux, 16);
    if( ThreadsPerSegment >= 16 )
        aux += __shfl_down_sync(0xFFFFFFFF, aux, 8);
    if( ThreadsPerSegment >= 8 )
        aux += __shfl_down_sync(0xFFFFFFFF, aux, 4);
    if( ThreadsPerSegment >= 4 )
        aux += __shfl_down_sync(0xFFFFFFFF, aux, 2);
    if( ThreadsPerSegment >= 2 )
        aux += __shfl_down_sync(0xFFFFFFFF, aux, 1);

   if( laneIdx == 0 )
    keeper( segmentIdx, aux )
}
#endif


template< typename OffsetsView,
          typename Index,
          typename Fetch,
          typename Reduction,
          typename ResultKeeper,
          typename Real,
          typename... Args >
void
RowsReductionLightKernelCaller(
    const Index elementsInSegment,
    const OffsetsView& offsets,
    Index first,
    Index last,
    Fetch& fetch,
    const Reduction& reduction,
    ResultKeeper& keeper,
    const Real& zero,
    Args... args )
{
#ifdef HAVE_CUDA
    const int threadsPerSegment = TNL::min( std::pow( 2, std::floor( std::log2( elementInSegment ) ) ), TNL::Cuda::getWarpSize() );
    TNL::ASSERT_GE( threadsPerSegment, 0 );
    TNL::ASSERT_LE( threadsPerSegment, 32 );
    const size_t threadsCount = threadsPerSegment * ( last - first );
    dim3 blocksCount, gridsCount, blockSize( 256 );
    TNL::Cuda::setupThreads( blockSize, blocksCount, gridsCount, threadsCount );
    for( int gridIdx = 0; gridIdx < gridsCount.x; gridIdx ++ )
    {
        dim3 gridSize;
        setupGrid( blocksCount, gridsCount, gridIdx, gridSize );
        switch( threadsPerSegment )
        {
            case 1:
                SpMVCSRLight<  1, Index, Fetch, Redcution, ResultKeeper, Real, Args ><<< gridSize, blockSize >>>(
                    gridIdx.x, offsets, first, last, fetch, reduction, keeper, zero, args );
                    break;
            case 2:
                SpMVCSRLight<  2, Index, Fetch, Redcution, ResultKeeper, Real, Args ><<< gridSize, blockSize >>>(
                    gridIdx.x, offsets, first, last, fetch, reduction, keeper, zero, args );
                    break;
            case 4:
                SpMVCSRLight<  4, Index, Fetch, Redcution, ResultKeeper, Real, Args ><<< gridSize, blockSize >>>(
                    gridIdx.x, offsets, first, last, fetch, reduction, keeper, zero, args );
                    break;
            case 8:
                SpMVCSRLight<  8, Index, Fetch, Redcution, ResultKeeper, Real, Args ><<< gridSize, blockSize >>>(
                    gridIdx.x, offsets, first, last, fetch, reduction, keeper, zero, args );
                    break;
            case 16:
                SpMVCSRLight< 16, Index, Fetch, Redcution, ResultKeeper, Real, Args ><<< gridSize, blockSize >>>(
                    gridIdx.x, offsets, first, last, fetch, reduction, keeper, zero, args );
                    break;
            case 32:
                SpMVCSRLight< 32, Index, Fetch, Redcution, ResultKeeper, Real, Args ><<< gridSize, blockSize >>>(
                    gridIdx.x, offsets, first, last, fetch, reduction, keeper, zero, args );
                    break;
            default:
                throw std::runtime_error( "Wrong value of threadsPerSegment." );
    };
#endif
}

         } // namespace details
      } // namespace Segments
   }  // namespace Algorithms
} // namespace TNL
