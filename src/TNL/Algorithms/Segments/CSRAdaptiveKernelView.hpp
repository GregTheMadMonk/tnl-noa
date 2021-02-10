/***************************************************************************
                          CSRAdaptiveKernelView.hpp -  description
                             -------------------
    begin                : Feb 7, 2021 -> Joe Biden inauguration
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
#include <TNL/Algorithms/Segments/CSRScalarKernel.h>
#include <TNL/Algorithms/Segments/CSRAdaptiveKernelView.h>
#include <TNL/Algorithms/Segments/details/CSRAdaptiveKernelBlockDescriptor.h>
#include <TNL/Algorithms/Segments/details/CSRAdaptiveKernelParameters.h>

namespace TNL {
   namespace Algorithms {
      namespace Segments {

#ifdef HAVE_CUDA

template< int warpSize,
          int WARPS,
          int SHARED_PER_WARP,
          int MAX_ELEM_PER_WARP,
          typename BlocksView,
          typename Offsets,
          typename Index,
          typename Fetch,
          typename Reduction,
          typename ResultKeeper,
          typename Real,
          typename... Args >
__global__ void
segmentsReductionCSRAdaptiveKernel( BlocksView blocks,
                                    int gridIdx,
                                    Offsets offsets,
                                    Index first,
                                    Index last,
                                    Fetch fetch,
                                    Reduction reduce,
                                    ResultKeeper keep,
                                    Real zero,
                                    Args... args )
{
   static constexpr int CudaBlockSize = details::CSRAdaptiveKernelParameters< sizeof( Real ) >::CudaBlockSize();
   //constexpr int WarpSize = Cuda::getWarpSize();
   //constexpr int WarpsCount = details::CSRAdaptiveKernelParameters< sizeof( Real ) >::WarpsCount();
   //constexpr size_t StreamedSharedElementsPerWarp  = details::CSRAdaptiveKernelParameters< sizeof( Real ) >::StreamedSharedElementsPerWarp();


   __shared__ Real streamShared[ WARPS ][ SHARED_PER_WARP ];
   __shared__ Real multivectorShared[ CudaBlockSize / warpSize ];
   constexpr size_t MAX_X_DIM = 2147483647;
   const Index index = (gridIdx * MAX_X_DIM) + (blockIdx.x * blockDim.x) + threadIdx.x;
   const Index blockIdx = index / warpSize;
   if( blockIdx >= blocks.getSize() - 1 )
      return;

   if( threadIdx.x < CudaBlockSize / warpSize )
      multivectorShared[ threadIdx.x ] = zero;
   Real result = zero;
   bool compute( true );
   const Index laneIdx = threadIdx.x & 31; // & is cheaper than %
   const details::CSRAdaptiveKernelBlockDescriptor< Index > block = blocks[ blockIdx ];
   const Index& firstSegmentIdx = block.getFirstSegment();
   const Index begin = offsets[ firstSegmentIdx ];

   const auto blockType = block.getType();
   if( blockType == details::Type::STREAM ) // Stream kernel - many short segments per warp
   {
      const Index warpIdx = threadIdx.x / 32;
      const Index end = begin + block.getSize();

      // Stream data to shared memory
      for( Index globalIdx = laneIdx + begin; globalIdx < end; globalIdx += warpSize )
      {
         streamShared[ warpIdx ][ globalIdx - begin ] = //fetch( globalIdx, compute );
            details::FetchLambdaAdapter< Index, Fetch >::call( fetch, -1, -1, globalIdx, compute );
         // TODO:: fix this by template specialization so that we can assume fetch lambda
         // with short parameters
      }

      const Index lastSegmentIdx = firstSegmentIdx + block.getSegmentsInBlock();

      for( Index i = firstSegmentIdx + laneIdx; i < lastSegmentIdx; i += warpSize )
      {
         const Index sharedEnd = offsets[ i + 1 ] - begin; // end of preprocessed data
         result = zero;
         // Scalar reduction
         for( Index sharedIdx = offsets[ i ] - begin; sharedIdx < sharedEnd; sharedIdx++ )
            result = reduce( result, streamShared[ warpIdx ][ sharedIdx ] );
         keep( i, result );
      }
   }
   else if( blockType == details::Type::VECTOR ) // Vector kernel - one segment per warp
   {
      const Index end = begin + block.getSize();
      const Index segmentIdx = block.getFirstSegment();

      for( Index globalIdx = begin + laneIdx; globalIdx < end; globalIdx += warpSize )
         result = reduce( result, details::FetchLambdaAdapter< Index, Fetch >::call( fetch, segmentIdx, -1, globalIdx, compute ) ); // fix local idx

      // Parallel reduction
      result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 16 ) );
      result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result,  8 ) );
      result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result,  4 ) );
      result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result,  2 ) );
      result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result,  1 ) );
      if( laneIdx == 0 )
         keep( segmentIdx, result );
   }
   else // blockType == Type::LONG - several warps per segment
   {
      // Number of elements processed by previous warps
      //const Index offset = //block.index[1] * MAX_ELEM_PER_WARP;
      ///   block.getWarpIdx() * MAX_ELEM_PER_WARP;
      //Index to = begin + (block.getWarpIdx()  + 1) * MAX_ELEM_PER_WARP;
      const Index segmentIdx = block.getFirstSegment();//block.index[0];
      //minID = offsets[block.index[0] ];
      const Index end = offsets[segmentIdx + 1];
      //const int tid = threadIdx.x;
      //const int inBlockWarpIdx = block.getWarpIdx();

      //if( to > end )
      //   to = end;
      TNL_ASSERT_GT( block.getWarpsCount(), 0, "" );
      result = zero;
      //printf( "LONG tid %d warpIdx %d: LONG \n", tid, block.getWarpIdx()  );
      for( Index globalIdx = begin + laneIdx + TNL::Cuda::getWarpSize() * block.getWarpIdx();
           globalIdx < end;
           globalIdx += TNL::Cuda::getWarpSize() * block.getWarpsCount() )
      {
         result = reduce( result, details::FetchLambdaAdapter< Index, Fetch >::call( fetch, segmentIdx, -1, globalIdx, compute ) );
         //if( laneIdx == 0 )
         //   printf( "LONG warpIdx: %d gid: %d begin: %d end: %d -> %d \n", ( int ) block.getWarpIdx(), globalIdx, begin, end,
         //    details::FetchLambdaAdapter< Index, Fetch >::call( fetch, segmentIdx, 0, globalIdx, compute ) );
         //result += values[i] * inVector[columnIndexes[i]];
      }
      //printf( "tid %d -> %d \n", tid, result );

      result += __shfl_down_sync(0xFFFFFFFF, result, 16);
      result += __shfl_down_sync(0xFFFFFFFF, result, 8);
      result += __shfl_down_sync(0xFFFFFFFF, result, 4);
      result += __shfl_down_sync(0xFFFFFFFF, result, 2);
      result += __shfl_down_sync(0xFFFFFFFF, result, 1);

      //if( laneIdx == 0 )
      //   printf( "WARP RESULT: tid %d -> %d \n", tid, result );

      const Index warpID = threadIdx.x / 32;
      if( laneIdx == 0 )
         multivectorShared[ warpID ] = result;

      __syncthreads();
      // Reduction in multivectorShared
      if( block.getWarpIdx() == 0 && laneIdx < 16 )
      {
         constexpr int totalWarps = CudaBlockSize / warpSize;
         if( totalWarps >= 32 )
         {
            multivectorShared[ laneIdx ] =  reduce( multivectorShared[ laneIdx ], multivectorShared[ laneIdx + 16 ] );
            __syncwarp();
         }
         if( totalWarps >= 16 )
         {
            multivectorShared[ laneIdx ] =  reduce( multivectorShared[ laneIdx ], multivectorShared[ laneIdx +  8 ] );
            __syncwarp();
         }
         if( totalWarps >= 8 )
         {
            multivectorShared[ laneIdx ] =  reduce( multivectorShared[ laneIdx ], multivectorShared[ laneIdx +  4 ] );
            __syncwarp();
         }
         if( totalWarps >= 4 )
         {
            multivectorShared[ laneIdx ] =  reduce( multivectorShared[ laneIdx ], multivectorShared[ laneIdx +  2 ] );
            __syncwarp();
         }
         if( totalWarps >= 2 )
         {
            multivectorShared[ laneIdx ] =  reduce( multivectorShared[ laneIdx ], multivectorShared[ laneIdx +  1 ] );
            __syncwarp();
         }
         if( laneIdx == 0 )
         {
            //printf( "Long: segmentIdx %d -> %d \n", segmentIdx, multivectorShared[ 0 ] );
            keep( segmentIdx, multivectorShared[ 0 ] );
         }
      }
   }
}
#endif

/*template< typename Index,
          typename Device >
CSRAdaptiveKernelView< Index, Device >::
CSRAdaptiveKernelView( BlocksType& blocks )
{
   this->blocks.bind( blocks );
}*/

template< typename Index,
          typename Device >
void
CSRAdaptiveKernelView< Index, Device >::
setBlocks( BlocksType& blocks, const int idx )
{
   this->blocksArray[ idx ].bind( blocks );
}

template< typename Index,
          typename Device >
auto
CSRAdaptiveKernelView< Index, Device >::
getView() -> ViewType
{
   return *this;
};

template< typename Index,
          typename Device >
auto
CSRAdaptiveKernelView< Index, Device >::
getConstView() const -> ConstViewType
{
   return *this;
}

template< typename Index,
          typename Device >
TNL::String
CSRAdaptiveKernelView< Index, Device >::
getKernelType()
{
   return "Adaptive";
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
CSRAdaptiveKernelView< Index, Device >::
segmentsReduction( const OffsetsView& offsets,
                   Index first,
                   Index last,
                   Fetch& fetch,
                   const Reduction& reduction,
                   ResultKeeper& keeper,
                   const Real& zero,
                   Args... args ) const
{
#ifdef HAVE_CUDA
   int valueSizeLog = std::ceil( log2f( ( double ) sizeof( Real ) ) );

   if( details::CheckFetchLambda< Index, Fetch >::hasAllParameters() || valueSizeLog > MaxValueSizeLog )
   {
      TNL::Algorithms::Segments::CSRScalarKernel< Index, Device >::
         segmentsReduction( offsets, first, last, fetch, reduction, keeper, zero, args... );
      return;
   }

   static constexpr Index THREADS_ADAPTIVE = details::CSRAdaptiveKernelParameters< sizeof( Real ) >::CudaBlockSize(); //sizeof(Index) == 8 ? 128 : 256;

   /* Max length of row to process one warp for CSR Light, MultiVector */
   //static constexpr Index MAX_ELEMENTS_PER_WARP = 384;

   /* Max length of row to process one warp for CSR Adaptive */
   //static constexpr Index MAX_ELEMENTS_PER_WARP_ADAPT = details::CSRAdaptiveKernelParameters< sizeof( Real ) >::MaxAdaptiveElementsPerWarp();

   /* How many shared memory use per block in CSR Adaptive kernel */
   static constexpr Index SHARED_PER_BLOCK = details::CSRAdaptiveKernelParameters< sizeof( Real ) >::StreamedSharedMemory();

   /* Number of elements in shared memory */
   static constexpr Index SHARED = SHARED_PER_BLOCK/sizeof(Real);

   /* Number of warps in block for CSR Adaptive */
   static constexpr Index WARPS = THREADS_ADAPTIVE / 32;

   /* Number of elements in shared memory per one warp */
   static constexpr Index SHARED_PER_WARP = SHARED / WARPS;

   constexpr int warpSize = 32;

   Index blocksCount;

   const Index threads = THREADS_ADAPTIVE;
   constexpr size_t MAX_X_DIM = 2147483647;

   /* Fill blocks */
   size_t neededThreads = this->blocksArray[ valueSizeLog ].getSize() * warpSize; // one warp per block
   /* Execute kernels on device */
   for (Index gridIdx = 0; neededThreads != 0; gridIdx++ )
   {
      if (MAX_X_DIM * threads >= neededThreads)
      {
         blocksCount = roundUpDivision(neededThreads, threads);
         neededThreads = 0;
      }
      else
      {
         blocksCount = MAX_X_DIM;
         neededThreads -= MAX_X_DIM * threads;
      }

      segmentsReductionCSRAdaptiveKernel<
            warpSize,
            WARPS,
            SHARED_PER_WARP,
            details::CSRAdaptiveKernelParameters< sizeof( Real ) >::MaxAdaptiveElementsPerWarp(),
            BlocksView,
            OffsetsView,
            Index, Fetch, Reduction, ResultKeeper, Real, Args... >
         <<<blocksCount, threads>>>(
            this->blocksArray[ valueSizeLog ],
            gridIdx,
            offsets,
            first,
            last,
            fetch,
            reduction,
            keeper,
            zero,
            args... );
   }
#endif
}

template< typename Index,
          typename Device >
CSRAdaptiveKernelView< Index, Device >&
CSRAdaptiveKernelView< Index, Device >::
operator=( const CSRAdaptiveKernelView< Index, Device >& kernelView )
{
   for( int i = 0; i < MaxValueSizeLog; i++ )
      this->blocksArray[ i ].bind( kernelView.blocksArray[ i ] );
   return *this;
}

template< typename Index,
          typename Device >
void
CSRAdaptiveKernelView< Index, Device >::
printBlocks( int idx ) const
{
   auto& blocks = this->blocksArray[ idx ];
   for( Index i = 0; i < this->blocks.getSize(); i++ )
   {
      auto block = blocks.getElement( i );
      std::cout << "Block " << i << " : " << block << std::endl;
   }

}

      } // namespace Segments
   }  // namespace Algorithms
} // namespace TNL
