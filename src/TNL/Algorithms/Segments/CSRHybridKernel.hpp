/***************************************************************************
                          CSRHybridKernel.hpp -  description
                             -------------------
    begin                : Jan 23, 2021 -> Joe Biden inauguration
    copyright            : (C) 2021 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Assert.h>
#include <TNL/Cuda/LaunchHelpers.h>
#include <TNL/Containers/VectorView.h>
#include <TNL/Algorithms/ParallelFor.h>
#include <TNL/Algorithms/Segments/detail/LambdaAdapter.h>
#include <TNL/Algorithms/Segments/CSRHybridKernel.h>

namespace TNL {
   namespace Algorithms {
      namespace Segments {

#ifdef HAVE_CUDA
template< int ThreadsPerSegment,
          typename Offsets,
          typename Index,
          typename Fetch,
          typename Reduction,
          typename ResultKeeper,
          typename Real >
__global__
void segmentsReductionCSRHybridVectorKernel(
    int gridIdx,
    const Offsets offsets,
    Index first,
    Index last,
    Fetch fetch,
    const Reduction reduce,
    ResultKeeper keep,
    const Real zero )
{
    const Index segmentIdx =  TNL::Cuda::getGlobalThreadIdx( gridIdx ) / ThreadsPerSegment + first;
    if( segmentIdx >= last )
        return;

    const int laneIdx = threadIdx.x & ( ThreadsPerSegment - 1 ); // & is cheaper than %
    Index endIdx = offsets[ segmentIdx + 1] ;

    Index localIdx( laneIdx );
    Real aux = zero;
    bool compute( true );
    for( Index globalIdx = offsets[ segmentIdx ] + localIdx; globalIdx < endIdx; globalIdx += ThreadsPerSegment )
    {
      aux = reduce( aux, detail::FetchLambdaAdapter< Index, Fetch >::call( fetch, segmentIdx, localIdx, globalIdx, compute ) );
      localIdx += TNL::Cuda::getWarpSize();
    }

    /****
     * Reduction in each segment.
     */
    if( ThreadsPerSegment == 32 )
        aux = reduce( aux, __shfl_down_sync( 0xFFFFFFFF, aux, 16 ) );
    if( ThreadsPerSegment >= 16 )
        aux = reduce( aux, __shfl_down_sync( 0xFFFFFFFF, aux,  8 ) );
    if( ThreadsPerSegment >= 8 )
        aux = reduce( aux, __shfl_down_sync( 0xFFFFFFFF, aux,  4 ) );
    if( ThreadsPerSegment >= 4 )
        aux = reduce( aux, __shfl_down_sync( 0xFFFFFFFF, aux,  2 ) );
    if( ThreadsPerSegment >= 2 )
        aux = reduce( aux, __shfl_down_sync( 0xFFFFFFFF, aux,  1 ) );

    if( laneIdx == 0 )
        keep( segmentIdx, aux );
}

template< int BlockSize,
          int ThreadsPerSegment,
          typename Offsets,
          typename Index,
          typename Fetch,
          typename Reduction,
          typename ResultKeeper,
          typename Real >
__global__
void segmentsReductionCSRHybridMultivectorKernel(
    int gridIdx,
    const Offsets offsets,
    Index first,
    Index last,
    Fetch fetch,
    const Reduction reduce,
    ResultKeeper keep,
    const Real zero )
{
    const Index segmentIdx =  TNL::Cuda::getGlobalThreadIdx( gridIdx ) / ThreadsPerSegment + first;
    if( segmentIdx >= last )
        return;

    __shared__ Real shared[ BlockSize / 32 ];
    if( threadIdx.x < BlockSize / TNL::Cuda::getWarpSize() )
        shared[ threadIdx.x ] = zero;

    const int laneIdx = threadIdx.x & ( ThreadsPerSegment - 1 ); // & is cheaper than %
    const int inWarpLaneIdx = threadIdx.x & ( TNL::Cuda::getWarpSize() - 1 ); // & is cheaper than %
    const Index beginIdx = offsets[ segmentIdx ];
    const Index endIdx   = offsets[ segmentIdx + 1 ] ;

    Real result = zero;
    bool compute( true );
    Index localIdx = laneIdx;
    for( Index globalIdx = beginIdx + laneIdx; globalIdx < endIdx && compute; globalIdx += ThreadsPerSegment )
    {
       result = reduce( result, details::FetchLambdaAdapter< Index, Fetch >::call( fetch, segmentIdx, localIdx, globalIdx, compute ) );
       localIdx += ThreadsPerSegment;
    }
    result += __shfl_down_sync(0xFFFFFFFF, result, 16);
    result += __shfl_down_sync(0xFFFFFFFF, result, 8);
    result += __shfl_down_sync(0xFFFFFFFF, result, 4);
    result += __shfl_down_sync(0xFFFFFFFF, result, 2);
    result += __shfl_down_sync(0xFFFFFFFF, result, 1);

    const Index warpIdx = threadIdx.x / TNL::Cuda::getWarpSize();
    if( inWarpLaneIdx == 0 )
        shared[ warpIdx ] = result;

    __syncthreads();
    // Reduction in shared
    if( warpIdx == 0 && inWarpLaneIdx < 16 )
    {
        //constexpr int totalWarps = BlockSize / WarpSize;
        constexpr int warpsPerSegment = ThreadsPerSegment / TNL::Cuda::getWarpSize();
        if( warpsPerSegment >= 32 )
        {
            shared[ laneIdx ] =  reduce( shared[ laneIdx ], shared[ laneIdx + 16 ] );
            __syncwarp();
        }
        if( warpsPerSegment >= 16 )
        {
            shared[ laneIdx ] =  reduce( shared[ laneIdx ], shared[ laneIdx +  8 ] );
            __syncwarp();
        }
        if( warpsPerSegment >= 8 )
        {
            shared[ laneIdx ] =  reduce( shared[ laneIdx ], shared[ laneIdx +  4 ] );
            __syncwarp();
        }
        if( warpsPerSegment >= 4 )
        {
            shared[ laneIdx ] =  reduce( shared[ laneIdx ], shared[ laneIdx +  2 ] );
            __syncwarp();
        }
        if( warpsPerSegment >= 2 )
        {
            shared[ laneIdx ] =  reduce( shared[ laneIdx ], shared[ laneIdx +  1 ] );
            __syncwarp();
        }
        constexpr int segmentsCount = BlockSize / ThreadsPerSegment;
        if( inWarpLaneIdx < segmentsCount )
        {
            //printf( "Long: segmentIdx %d -> %d \n", segmentIdx, shared[ inWarpLaneIdx ] );
            keep( segmentIdx + inWarpLaneIdx, shared[ inWarpLaneIdx ] );
        }
    }
}
#endif



template< typename Index,
          typename Device,
          int ThreadsInBlock >
    template< typename Offsets >
void
CSRHybridKernel< Index, Device, ThreadsInBlock >::
init( const Offsets& offsets )
{
    const Index segmentsCount = offsets.getSize() - 1;
    const Index elementsInSegment = std::ceil( ( double ) offsets.getElement( segmentsCount ) / ( double ) segmentsCount );
    this->threadsPerSegment = TNL::min( std::pow( 2, std::ceil( std::log2( elementsInSegment ) ) ), ThreadsInBlock ); //TNL::Cuda::getWarpSize() );
    TNL_ASSERT_GE( threadsPerSegment, 0, "" );
    TNL_ASSERT_LE( threadsPerSegment, ThreadsInBlock, "" );
}

template< typename Index,
          typename Device,
          int ThreadsInBlock >
void
CSRHybridKernel< Index, Device, ThreadsInBlock >::
reset()
{
    this->threadsPerSegment = 0;
}

template< typename Index,
          typename Device,
          int ThreadsInBlock >
auto
CSRHybridKernel< Index, Device, ThreadsInBlock >::
getView() -> ViewType
{
    return *this;
}

template< typename Index,
          typename Device,
          int ThreadsInBlock >
TNL::String
CSRHybridKernel< Index, Device, ThreadsInBlock >::
getKernelType()
{
    return "Hybrid";
}

template< typename Index,
          typename Device,
          int ThreadsInBlock >
auto
CSRHybridKernel< Index, Device, ThreadsInBlock >::
getConstView() const -> ConstViewType
{
    return *this;
};


template< typename Index,
          typename Device,
          int ThreadsInBlock >
    template< typename OffsetsView,
              typename Fetch,
              typename Reduction,
              typename ResultKeeper,
              typename Real >
void
CSRHybridKernel< Index, Device, ThreadsInBlock >::
segmentsReduction( const OffsetsView& offsets,
                         Index first,
                         Index last,
                         Fetch& fetch,
                         const Reduction& reduction,
                         ResultKeeper& keeper,
                         const Real& zero ) const
{
    TNL_ASSERT_GE( this->threadsPerSegment, 0, "" );
    TNL_ASSERT_LE( this->threadsPerSegment, ThreadsInBlock, "" );

#ifdef HAVE_CUDA
    const size_t threadsCount = this->threadsPerSegment * ( last - first );
    dim3 blocksCount, gridsCount, blockSize( ThreadsInBlock );
    TNL::Cuda::setupThreads( blockSize, blocksCount, gridsCount, threadsCount );
    //std::cerr << " this->threadsPerSegment = " << this->threadsPerSegment << " offsets = " << offsets << std::endl;
    for( unsigned int gridIdx = 0; gridIdx < gridsCount.x; gridIdx ++ )
    {
        dim3 gridSize;
        TNL::Cuda::setupGrid( blocksCount, gridsCount, gridIdx, gridSize );
        switch( this->threadsPerSegment )
        {
            case 0:      // this means zero/empty matrix
                break;
            case 1:
                segmentsReductionCSRHybridVectorKernel<  1, OffsetsView, Index, Fetch, Reduction, ResultKeeper, Real ><<< gridSize, blockSize >>>(
                    gridIdx, offsets, first, last, fetch, reduction, keeper, zero );
                    break;
            case 2:
                segmentsReductionCSRHybridVectorKernel<  2, OffsetsView, Index, Fetch, Reduction, ResultKeeper, Real ><<< gridSize, blockSize >>>(
                    gridIdx, offsets, first, last, fetch, reduction, keeper, zero );
                    break;
            case 4:
                segmentsReductionCSRHybridVectorKernel<  4, OffsetsView, Index, Fetch, Reduction, ResultKeeper, Real ><<< gridSize, blockSize >>>(
                    gridIdx, offsets, first, last, fetch, reduction, keeper, zero );
                    break;
            case 8:
                segmentsReductionCSRHybridVectorKernel<  8, OffsetsView, Index, Fetch, Reduction, ResultKeeper, Real ><<< gridSize, blockSize >>>(
                    gridIdx, offsets, first, last, fetch, reduction, keeper, zero );
                    break;
            case 16:
                segmentsReductionCSRHybridVectorKernel< 16, OffsetsView, Index, Fetch, Reduction, ResultKeeper, Real ><<< gridSize, blockSize >>>(
                    gridIdx, offsets, first, last, fetch, reduction, keeper, zero );
                    break;
            case 32:
                segmentsReductionCSRHybridVectorKernel< 32, OffsetsView, Index, Fetch, Reduction, ResultKeeper, Real ><<< gridSize, blockSize >>>(
                    gridIdx, offsets, first, last, fetch, reduction, keeper, zero );
                    break;
            case 64:
                segmentsReductionCSRHybridMultivectorKernel< ThreadsInBlock,  64, OffsetsView, Index, Fetch, Reduction, ResultKeeper, Real ><<< gridSize, blockSize >>>(
                    gridIdx, offsets, first, last, fetch, reduction, keeper, zero );
                    break;
            case 128:
                segmentsReductionCSRHybridMultivectorKernel< ThreadsInBlock, 128, OffsetsView, Index, Fetch, Reduction, ResultKeeper, Real ><<< gridSize, blockSize >>>(
                    gridIdx, offsets, first, last, fetch, reduction, keeper, zero );
                    break;
            case 256:
                segmentsReductionCSRHybridMultivectorKernel< ThreadsInBlock, 256, OffsetsView, Index, Fetch, Reduction, ResultKeeper, Real ><<< gridSize, blockSize >>>(
                    gridIdx, offsets, first, last, fetch, reduction, keeper, zero );
                    break;
            default:
                throw std::runtime_error( std::string( "Wrong value of threadsPerSegment: " ) + std::to_string( this->threadsPerSegment ) );
        }
    }
#endif
}

      } // namespace Segments
   }  // namespace Algorithms
} // namespace TNL
