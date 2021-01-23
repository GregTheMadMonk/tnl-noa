/***************************************************************************
                          CSRKernels.h -  description
                             -------------------
    begin                : Jan 20, 2021 -> Joe Biden inauguration
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
void segmentsReductionCSRLightKernel(
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
struct CSRLightKernel
{
    using IndexType = Index;
    using DeviceType = Device;
    using ViewType = CSRLightKernel< Index, Device >;
    using ConstViewType = CSRLightKernel< Index, Device >;

    template< typename Offsets >
    void init( const Offsets& offsets )
    {
        const Index segmentsCount = offsets.getSize() - 1;
        const Index elementsInSegment = std::ceil( ( double ) offsets.getElement( segmentsCount ) / ( double ) segmentsCount );
        this->threadsPerSegment = TNL::min( std::pow( 2, std::ceil( std::log2( elementsInSegment ) ) ), TNL::Cuda::getWarpSize() );
        TNL_ASSERT_GE( threadsPerSegment, 0, "" );
        TNL_ASSERT_LE( threadsPerSegment, 32, "" );
    };

    void reset() { this->threadsPerSegment = 0; }

    ViewType getView() { return *this; };

    ConstViewType getConstView() const { return *this; };

    template< typename OffsetsView,
              typename Fetch,
              typename Reduction,
              typename ResultKeeper,
              typename Real,
              typename... Args >
    void segmentsReduction( const OffsetsView& offsets,
                        Index first,
                        Index last,
                        Fetch& fetch,
                        const Reduction& reduction,
                        ResultKeeper& keeper,
                        const Real& zero,
                        Args... args ) const
    {
        TNL_ASSERT_GE( threadsPerSegment, 0, "" );
        TNL_ASSERT_LE( threadsPerSegment, 32, "" );

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
                    segmentsReductionCSRLightKernel<  1, OffsetsView, Index, Fetch, Reduction, ResultKeeper, Real, Args... ><<< gridSize, blockSize >>>(
                        gridIdx, offsets, first, last, fetch, reduction, keeper, zero, args... );
                        break;
                case 2:
                    segmentsReductionCSRLightKernel<  2, OffsetsView, Index, Fetch, Reduction, ResultKeeper, Real, Args... ><<< gridSize, blockSize >>>(
                        gridIdx, offsets, first, last, fetch, reduction, keeper, zero, args... );
                        break;
                case 4:
                    segmentsReductionCSRLightKernel<  4, OffsetsView, Index, Fetch, Reduction, ResultKeeper, Real, Args... ><<< gridSize, blockSize >>>(
                        gridIdx, offsets, first, last, fetch, reduction, keeper, zero, args... );
                        break;
                case 8:
                    segmentsReductionCSRLightKernel<  8, OffsetsView, Index, Fetch, Reduction, ResultKeeper, Real, Args... ><<< gridSize, blockSize >>>(
                        gridIdx, offsets, first, last, fetch, reduction, keeper, zero, args... );
                        break;
                case 16:
                    segmentsReductionCSRLightKernel< 16, OffsetsView, Index, Fetch, Reduction, ResultKeeper, Real, Args... ><<< gridSize, blockSize >>>(
                        gridIdx, offsets, first, last, fetch, reduction, keeper, zero, args... );
                        break;
                case 32:
                    segmentsReductionCSRLightKernel< 32, OffsetsView, Index, Fetch, Reduction, ResultKeeper, Real, Args... ><<< gridSize, blockSize >>>(
                        gridIdx, offsets, first, last, fetch, reduction, keeper, zero, args... );
                        break;
                default:
                    throw std::runtime_error( std::string( "Wrong value of threadsPerSegment: " ) + std::to_string( this->threadsPerSegment ) );
            }
        }
#endif
    }

    protected:
        int threadsPerSegment;
};


template< typename Index,
          typename Device >
struct CSRAdaptiveKernelView
{
    using IndexType = Index;
    using DeviceType = Device;
    using ViewType = CSRAdaptiveKernelView< Index, Device >;
    using ConstViewType = CSRAdaptiveKernelView< Index, Device >;

    ViewType getView() { return *this; };

    ConstViewType getConstView() const { return *this; };

    template< typename OffsetsView,
              typename Fetch,
              typename Reduction,
              typename ResultKeeper,
              typename Real,
              typename... Args >
    void segmentsReduction( const OffsetsView& offsets,
                        Index first,
                        Index last,
                        Fetch& fetch,
                        const Reduction& reduction,
                        ResultKeeper& keeper,
                        const Real& zero,
                        Args... args ) const
    {
    }
};

template< typename Index,
          typename Device >
struct CSRAdaptiveKernel
{
    using IndexType = Index;
    using DeviceType = Device;
    using ViewType = CSRAdaptiveKernel< Index, Device >;
    using ConstViewType = CSRAdaptiveKernel< Index, Device >;

    template< typename Offsets >
    void init( const Offsets& offsets )
    {
        /*const Index rows = offsets.getSize();
        Index sum, start = 0, nextStart = 0;

        // Fill blocks
        std::vector<Block<Index>> inBlock;
        inBlock.reserve(rows);

        while (nextStart != rows - 1)
        {
            Type type;
            nextStart = findLimit<Real, Index, Device, KernelType>(
                start, *this, rows, type, sum );

            if (type == Type::LONG)
            {
                Index parts = roundUpDivision(sum, this->SHARED_PER_WARP);
                for (Index index = 0; index < parts; ++index)
                {
                    inBlock.emplace_back(start, Type::LONG, index);
                }
            }
            else
            {
                inBlock.emplace_back(start, type,
                    nextStart,
                    this->rowPointers.getElement(nextStart),
                    this->rowPointers.getElement(start) );
            }
            start = nextStart;
        }
        inBlock.emplace_back(nextStart);

        // Copy values
        this->blocks.setSize(inBlock.size());
        for (size_t i = 0; i < inBlock.size(); ++i)
            this->blocks.setElement(i, inBlock[i]);
        */
    };

    ViewType getView() { return view; };

    ConstViewType getConstView() const { return ConstViewType(); };

    template< typename OffsetsView,
              typename Fetch,
              typename Reduction,
              typename ResultKeeper,
              typename Real,
              typename... Args >
    void segmentsReduction( const OffsetsView& offsets,
                        Index first,
                        Index last,
                        Fetch& fetch,
                        const Reduction& reduction,
                        ResultKeeper& keeper,
                        const Real& zero,
                        Args... args ) const
    {
        view.segmentsReduction( offsets, first, last, fetch, reduction, keeper, zero, args... );
    }

    ViewType view;
};



      } // namespace Segments
   }  // namespace Algorithms
} // namespace TNL
