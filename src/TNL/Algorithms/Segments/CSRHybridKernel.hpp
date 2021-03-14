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
#include <TNL/Algorithms/Segments/details/LambdaAdapter.h>
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
          typename Real,
          typename... Args >
__global__
void segmentsReductionCSRHybridKernel(
    int gridIdx,
    const Offsets offsets,
    Index first,
    Index last,
    Fetch fetch,
    const Reduction reduce,
    ResultKeeper keep,
    const Real zero,
    Args... args )
{
    /***
     * We map one warp to each segment
     */
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
      aux = reduce( aux, details::FetchLambdaAdapter< Index, Fetch >::call( fetch, segmentIdx, localIdx, globalIdx, compute ) );
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
#endif



template< typename Index,
          typename Device >
    template< typename Offsets >
void
CSRHybridKernel< Index, Device >::
init( const Offsets& offsets )
{
    const Index segmentsCount = offsets.getSize() - 1;
    const Index elementsInSegment = std::ceil( ( double ) offsets.getElement( segmentsCount ) / ( double ) segmentsCount );
    this->threadsPerSegment = TNL::min( std::pow( 2, std::ceil( std::log2( elementsInSegment ) ) ), TNL::Cuda::getWarpSize() );
    TNL_ASSERT_GE( threadsPerSegment, 0, "" );
    TNL_ASSERT_LE( threadsPerSegment, 32, "" );
}

template< typename Index,
          typename Device >
void
CSRHybridKernel< Index, Device >::
reset()
{
    this->threadsPerSegment = 0;
}

template< typename Index,
          typename Device >
auto
CSRHybridKernel< Index, Device >::
getView() -> ViewType
{
    return *this;
}

template< typename Index,
          typename Device >
TNL::String
CSRHybridKernel< Index, Device >::
getKernelType()
{
    return "Hybrid";
}

template< typename Index,
          typename Device >
auto
CSRHybridKernel< Index, Device >::
getConstView() const -> ConstViewType
{
    return *this;
};


template< typename Index,
          typename Device >
    template< typename OffsetsView,
              typename Fetch,
              typename Reduction,
              typename ResultKeeper,
              typename Real,
              typename... Args >
void
CSRHybridKernel< Index, Device >::
segmentsReduction( const OffsetsView& offsets,
                         Index first,
                         Index last,
                         Fetch& fetch,
                         const Reduction& reduction,
                         ResultKeeper& keeper,
                         const Real& zero,
                         Args... args ) const
{
    TNL_ASSERT_GE( this->threadsPerSegment, 0, "" );
    TNL_ASSERT_LE( this->threadsPerSegment, 32, "" );

#ifdef HAVE_CUDA
    const size_t threadsCount = this->threadsPerSegment * ( last - first );
    dim3 blocksCount, gridsCount, blockSize( 256 );
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
                segmentsReductionCSRHybridKernel<  1, OffsetsView, Index, Fetch, Reduction, ResultKeeper, Real, Args... ><<< gridSize, blockSize >>>(
                    gridIdx, offsets, first, last, fetch, reduction, keeper, zero, args... );
                    break;
            case 2:
                segmentsReductionCSRHybridKernel<  2, OffsetsView, Index, Fetch, Reduction, ResultKeeper, Real, Args... ><<< gridSize, blockSize >>>(
                    gridIdx, offsets, first, last, fetch, reduction, keeper, zero, args... );
                    break;
            case 4:
                segmentsReductionCSRHybridKernel<  4, OffsetsView, Index, Fetch, Reduction, ResultKeeper, Real, Args... ><<< gridSize, blockSize >>>(
                    gridIdx, offsets, first, last, fetch, reduction, keeper, zero, args... );
                    break;
            case 8:
                segmentsReductionCSRHybridKernel<  8, OffsetsView, Index, Fetch, Reduction, ResultKeeper, Real, Args... ><<< gridSize, blockSize >>>(
                    gridIdx, offsets, first, last, fetch, reduction, keeper, zero, args... );
                    break;
            case 16:
                segmentsReductionCSRHybridKernel< 16, OffsetsView, Index, Fetch, Reduction, ResultKeeper, Real, Args... ><<< gridSize, blockSize >>>(
                    gridIdx, offsets, first, last, fetch, reduction, keeper, zero, args... );
                    break;
            case 32:
                segmentsReductionCSRHybridKernel< 32, OffsetsView, Index, Fetch, Reduction, ResultKeeper, Real, Args... ><<< gridSize, blockSize >>>(
                    gridIdx, offsets, first, last, fetch, reduction, keeper, zero, args... );
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
