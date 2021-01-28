/***************************************************************************
                          CSRKernelVector.hpp -  description
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
#include <TNL/Algorithms/Segments/CSRKernelVector.h>

namespace TNL {
   namespace Algorithms {
      namespace Segments {

#ifdef HAVE_CUDA
template< typename Offsets,
          typename Index,
          typename Fetch,
          typename Reduction,
          typename ResultKeeper,
          typename Real,
          typename... Args >
__global__
void segmentsReductionCSRKernelVector(
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
    const Index segmentIdx =  TNL::Cuda::getGlobalThreadIdx( gridIdx ) / TNL::Cuda::getWarpSize() + first;
    if( segmentIdx >= last )
        return;

    const int laneIdx = threadIdx.x & ( TNL::Cuda::getWarpSize() - 1 ); // & is cheaper than %
    TNL_ASSERT_LT( segmentIdx + 1, offsets.getSize(), "" );
    Index endIdx = offsets[ segmentIdx + 1 ];

    Index localIdx( laneIdx );
    Real aux = zero;
    bool compute( true );
    for( Index globalIdx = offsets[ segmentIdx ] + localIdx; globalIdx < endIdx; globalIdx += TNL::Cuda::getWarpSize() )
    {
        TNL_ASSERT_LT( globalIdx, endIdx, "" );
        aux = reduce( aux, details::FetchLambdaAdapter< Index, Fetch >::call( fetch, segmentIdx, localIdx, globalIdx, compute ) );
        localIdx += TNL::Cuda::getWarpSize();
    }

   /****
    * Reduction in each warp which means in each segment.
    */
   aux = reduce( aux, __shfl_down_sync( 0xFFFFFFFF, aux, 16 ) );
   aux = reduce( aux, __shfl_down_sync( 0xFFFFFFFF, aux,  8 ) );
   aux = reduce( aux, __shfl_down_sync( 0xFFFFFFFF, aux,  4 ) );
   aux = reduce( aux, __shfl_down_sync( 0xFFFFFFFF, aux,  2 ) );
   aux = reduce( aux, __shfl_down_sync( 0xFFFFFFFF, aux,  1 ) );

   if( laneIdx == 0 )
     keep( segmentIdx, aux );
}
#endif

template< typename Index,
          typename Device >
    template< typename Offsets >
void
CSRKernelVector< Index, Device >::
init( const Offsets& offsets )
{
}

template< typename Index,
          typename Device >
void
CSRKernelVector< Index, Device >::
reset()
{
}

template< typename Index,
          typename Device >
auto
CSRKernelVector< Index, Device >::
getView() -> ViewType
{
    return *this;
}

template< typename Index,
          typename Device >
auto
CSRKernelVector< Index, Device >::
getConstView() const -> ConstViewType
{
    return *this;
};

template< typename Index,
          typename Device >
TNL::String
CSRKernelVector< Index, Device >::
getKernelType()
{
    return "Vector";
}

template< typename Index,
          typename Device >
    template< typename OffsetsView,
              typename Fetch,
              typename Reduction,
              typename ResultKeeper,
              typename Real,
              typename... Args >
void
CSRKernelVector< Index, Device >::
segmentsReduction( const OffsetsView& offsets,
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
    dim3 gridIdx;
    for( gridIdx.x = 0; gridIdx.x < gridsCount.x; gridIdx.x ++ )
    {
        dim3 gridSize;
        TNL::Cuda::setupGrid( blocksCount, gridsCount, gridIdx, gridSize );
        segmentsReductionCSRKernelVector< OffsetsView, IndexType, Fetch, Reduction, ResultKeeper, Real, Args... >
        <<< gridSize, blockSize >>>(
            gridIdx.x, offsets, first, last, fetch, reduction, keeper, zero, args... );
    };
#endif
}
      } // namespace Segments
   }  // namespace Algorithms
} // namespace TNL
