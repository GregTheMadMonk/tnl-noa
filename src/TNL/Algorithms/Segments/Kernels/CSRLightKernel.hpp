/***************************************************************************
                          CSRLightKernel.hpp -  description
                             -------------------
    begin                : Jun 9, 2021 -> Joe Biden inauguration
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
#include <TNL/Algorithms/Segments/Kernels/CSRLightKernel.h>

namespace TNL {
   namespace Algorithms {
      namespace Segments {

#ifdef HAVE_CUDA
template< typename Real,
          typename Index,
          typename OffsetsView,
          typename Fetch,
          typename Reduce,
          typename Keep >
__global__
void SpMVCSRLightWithoutAtomic2( OffsetsView offsets,
                                 const Index first,
                                 const Index last,
                                 Fetch fetch,
                                 Reduce reduce,
                                 Keep keep,
                                 const Real zero,
                                 const Index gridID)
{
   const Index segmentIdx =
      first + ( ( gridID * TNL::Cuda::getMaxGridXSize() ) + (blockIdx.x * blockDim.x) + threadIdx.x ) / 2;
   if( segmentIdx >= last )
      return;

   const Index inGroupID = threadIdx.x & 1; // & is cheaper than %
   const Index maxID = offsets[ segmentIdx  + 1];

   Real result = zero;
   bool compute = true;
   for( Index i = offsets[segmentIdx] + inGroupID; i < maxID; i += 2)
      result = reduce( result, fetch( i, compute ) );

   /* Parallel reduction */
   result = reduce( result, __shfl_down_sync(0xFFFFFFFF, result, 1 ) );

   /* Write result */
   if( inGroupID == 0 )
      keep( segmentIdx, result );
}

template< typename Real,
          typename Index,
          typename OffsetsView,
          typename Fetch,
          typename Reduce,
          typename Keep >
__global__
void SpMVCSRLightWithoutAtomic4( OffsetsView offsets,
                                 const Index first,
                                 const Index last,
                                 Fetch fetch,
                                 Reduce reduce,
                                 Keep keep,
                                 const Real zero,
                                 const Index gridID )
{
   const Index segmentIdx =
      first + ((gridID * TNL::Cuda::getMaxGridXSize() ) + (blockIdx.x * blockDim.x) + threadIdx.x) / 4;
   if (segmentIdx >= last)
      return;

   const Index inGroupID = threadIdx.x & 3; // & is cheaper than %
   const Index maxID = offsets[segmentIdx + 1];

   Real result = zero;
   bool compute = true;
   for (Index i = offsets[segmentIdx] + inGroupID; i < maxID; i += 4)
      result = reduce( result, fetch( i, compute ) );

   /* Parallel reduction */
   result = reduce( result, __shfl_down_sync(0xFFFFFFFF, result, 2 ) );
   result = reduce( result, __shfl_down_sync(0xFFFFFFFF, result, 1 ) );

   /* Write result */
   if( inGroupID == 0 )
      keep( segmentIdx, result );

}

template< typename Real,
          typename Index,
          typename OffsetsView,
          typename Fetch,
          typename Reduce,
          typename Keep >
__global__
void SpMVCSRLightWithoutAtomic8( OffsetsView offsets,
                                 const Index first,
                                 const Index last,
                                 Fetch fetch,
                                 Reduce reduce,
                                 Keep keep,
                                 const Real zero,
                                 const Index gridID)
{
   const Index segmentIdx =
      first + ((gridID * TNL::Cuda::getMaxGridXSize() ) + (blockIdx.x * blockDim.x) + threadIdx.x) / 8;
   if (segmentIdx >= last)
      return;

   Index i;
   const Index inGroupID = threadIdx.x & 7; // & is cheaper than %
   const Index maxID = offsets[segmentIdx + 1];

   Real result = zero;
   bool compute = true;
   for (i = offsets[segmentIdx] + inGroupID; i < maxID; i += 8)
      result = reduce( result, fetch( i, compute ) );

   /* Parallel reduction */
   result = reduce( result, __shfl_down_sync(0xFFFFFFFF, result, 4 ) );
   result = reduce( result, __shfl_down_sync(0xFFFFFFFF, result, 2 ) );
   result = reduce( result, __shfl_down_sync(0xFFFFFFFF, result, 1 ) );

   /* Write result */
   if( inGroupID == 0 )
      keep( segmentIdx, result );
}

template< typename Real,
          typename Index,
          typename OffsetsView,
          typename Fetch,
          typename Reduce,
          typename Keep >
__global__
void SpMVCSRLightWithoutAtomic16( OffsetsView offsets,
                                  const Index first,
                                  const Index last,
                                  Fetch fetch,
                                  Reduce reduce,
                                  Keep keep,
                                  const Real zero,
                                  const Index gridID )
{
   const Index segmentIdx =
      first + ((gridID * TNL::Cuda::getMaxGridXSize() ) + (blockIdx.x * blockDim.x) + threadIdx.x ) / 16;
   if( segmentIdx >= last )
      return;

   Index i;
   const Index inGroupID = threadIdx.x & 15; // & is cheaper than %
   const Index maxID = offsets[segmentIdx + 1];

   Real result = zero;
   bool compute = true;
   for( i = offsets[segmentIdx] + inGroupID; i < maxID; i += 16 )
      result = reduce( result, fetch( i, compute ) );

   /* Parallel reduction */
   result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 8 ) );
   result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 4 ) );
   result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 2 ) );
   result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 1 ) );

   /* Write result */
   if( inGroupID == 0 )
      keep( segmentIdx, result );
}

template< typename Real,
          typename Index,
          typename OffsetsView,
          typename Fetch,
          typename Reduce,
          typename Keep >
__global__
void SpMVCSRVector( OffsetsView offsets,
                    const Index first,
                    const Index last,
                    Fetch fetch,
                    Reduce reduce,
                    Keep keep,
                    const Real zero,
                    const Index gridID )
{
   const int warpSize = 32;
   const Index warpID = first + ((gridID * TNL::Cuda::getMaxGridXSize() ) + (blockIdx.x * blockDim.x) + threadIdx.x) / warpSize;
   if (warpID >= last)
      return;

   Real result = zero;
   const Index laneID = threadIdx.x & 31; // & is cheaper than %
   Index endID = offsets[warpID + 1];

   /* Calculate result */
   bool compute = true;
   for (Index i = offsets[warpID] + laneID; i < endID; i += warpSize)
      result = reduce( result, fetch( i, compute ) );

   /* Reduction */
   result = reduce( result, __shfl_down_sync(0xFFFFFFFF, result, 16 ) );
   result = reduce( result, __shfl_down_sync(0xFFFFFFFF, result,  8 ) );
   result = reduce( result, __shfl_down_sync(0xFFFFFFFF, result,  4 ) );
   result = reduce( result, __shfl_down_sync(0xFFFFFFFF, result,  2 ) );
   result = reduce( result, __shfl_down_sync(0xFFFFFFFF, result,  1 ) );
   /* Write result */
   if( laneID == 0 )
      keep( warpID, result );
}
#endif
template< typename Index,
          typename Device,
          typename Fetch,
          typename Reduce,
          typename Keep,
          bool DispatchScalarCSR =
            details::CheckFetchLambda< Index, Fetch >::hasAllParameters() ||
            std::is_same< Device, Devices::Host >::value >
struct CSRLightKernelreduceSegmentsDispatcher;

template< typename Index,
          typename Device,
          typename Fetch,
          typename Reduction,
          typename ResultKeeper >
struct CSRLightKernelreduceSegmentsDispatcher< Index, Device, Fetch, Reduction, ResultKeeper, true >
{

   template< typename Offsets,
             typename Real >
   static void reduce( const Offsets& offsets,
                       Index first,
                       Index last,
                       Fetch& fetch,
                       const Reduction& reduce,
                       ResultKeeper& keep,
                       const Real& zero,
                       const Index threadsPerSegment )
   {
      TNL::Algorithms::Segments::CSRScalarKernel< Index, Device >::
         reduceSegments( offsets, first, last, fetch, reduce, keep, zero );
   }
};

template< typename Index,
          typename Device,
          typename Fetch,
          typename Reduce,
          typename Keep >
struct CSRLightKernelreduceSegmentsDispatcher< Index, Device, Fetch, Reduce, Keep, false >
{
   template< typename OffsetsView,
             typename Real >
   static void reduce( const OffsetsView& offsets,
                       Index first,
                       Index last,
                       Fetch& fetch,
                       const Reduce& reduce,
                       Keep& keep,
                       const Real& zero,
                       const Index threadsPerSegment )
   {
#ifdef HAVE_CUDA
      const int threads = 128;
      Index blocks, groupSize;
      int  neededThreads = threadsPerSegment * ( last - first );

      for (Index grid = 0; neededThreads != 0; ++grid)
      {
         if( TNL::Cuda::getMaxGridXSize() * threads >= neededThreads)
         {
            blocks = roundUpDivision(neededThreads, threads);
            neededThreads = 0;
         }
         else
         {
            blocks = TNL::Cuda::getMaxGridXSize();
            neededThreads -= TNL::Cuda::getMaxGridXSize() * threads;
         }

         if (threadsPerSegment == 2)
            SpMVCSRLightWithoutAtomic2<Real, Index, OffsetsView, Fetch, Reduce, Keep ><<<blocks, threads>>>(
               offsets, first, last, fetch, reduce, keep, zero, grid );
         else if (threadsPerSegment == 4)
            SpMVCSRLightWithoutAtomic4<Real, Index, OffsetsView, Fetch, Reduce, Keep ><<<blocks, threads>>>(
               offsets, first, last, fetch, reduce, keep, zero, grid );
         else if (threadsPerSegment == 8)
            SpMVCSRLightWithoutAtomic8<Real, Index, OffsetsView, Fetch, Reduce, Keep ><<<blocks, threads>>>(
               offsets, first, last, fetch, reduce, keep, zero, grid );
         else if (threadsPerSegment == 16)
            SpMVCSRLightWithoutAtomic16<Real, Index, OffsetsView, Fetch, Reduce, Keep ><<<blocks, threads>>>(
               offsets, first, last, fetch, reduce, keep, zero, grid );
         else // if (threadsPerSegment == 32)
         { // CSR SpMV Light with threadsPerSegment = 32 is CSR Vector
            SpMVCSRVector<Real, Index, OffsetsView, Fetch, Reduce, Keep ><<<blocks, threads>>>(
               offsets, first, last, fetch, reduce, keep, zero, grid );
         }
         /*else
         { // Execute CSR MultiVector
            SpMVCSRMultiVector<Real, Index, warpSize><<<blocks, threads>>>(
                     inVector, outVector, matrix.getoffsets().getData(),
                     matrix.getColumnIndexes().getData(), matrix.getValues().getData(),
                     rows, threadsPerSegment / 32, grid
            );
         }*/

      }
#endif

   }
};


template< typename Index,
          typename Device >
    template< typename Offsets >
void
CSRLightKernel< Index, Device >::
init( const Offsets& offsets )
{
   const Index segmentsCount = offsets.getSize() - 1;
   size_t neededThreads = segmentsCount * 32;//warpSize;

   const Index elementsInSegment = roundUpDivision( offsets.getElement( segmentsCount ), segmentsCount ); // non zeroes per row
   if( elementsInSegment <= 2 )
      this->threadsPerSegment = 2;
   else if( elementsInSegment <= 4 )
      this->threadsPerSegment = 4;
   else if( elementsInSegment <= 8 )
      this->threadsPerSegment = 8;
   else if( elementsInSegment <= 16 )
      this->threadsPerSegment = 16;
   else //if (nnz <= 2 * matrix.MAX_ELEMENTS_PER_WARP)
      this->threadsPerSegment = 32; // CSR Vector
   //else
   //   threadsPerSegment = roundUpDivision(nnz, matrix.MAX_ELEMENTS_PER_WARP) * 32; // CSR MultiVector

   TNL_ASSERT_GE( this->threadsPerSegment, 0, "" );
   TNL_ASSERT_LE( this->threadsPerSegment, 33, "" );

}

template< typename Index,
          typename Device >
void
CSRLightKernel< Index, Device >::
reset()
{
   this->threadsPerSegment = 0;
}

template< typename Index,
          typename Device >
auto
CSRLightKernel< Index, Device >::
getView() -> ViewType
{
    return *this;
}

template< typename Index,
          typename Device >
TNL::String
CSRLightKernel< Index, Device >::
getKernelType()
{
    return "Light";
}

template< typename Index,
          typename Device >
auto
CSRLightKernel< Index, Device >::
getConstView() const -> ConstViewType
{
    return *this;
};


template< typename Index,
          typename Device >
    template< typename OffsetsView,
              typename Fetch,
              typename Reduce,
              typename Keep,
              typename Real >
void
CSRLightKernel< Index, Device >::
reduceSegments( const OffsetsView& offsets,
                Index first,
                Index last,
                Fetch& fetch,
                const Reduce& reduce,
                Keep& keep,
                const Real& zero ) const
{
   TNL_ASSERT_GE( this->threadsPerSegment, 0, "" );
   TNL_ASSERT_LE( this->threadsPerSegment, 33, "" );
   CSRLightKernelreduceSegmentsDispatcher< Index, Device, Fetch, Reduce, Keep >::reduce(
      offsets, first, last, fetch, reduce, keep, zero, this->threadsPerSegment );
}

      } // namespace Segments
   }  // namespace Algorithms
} // namespace TNL
