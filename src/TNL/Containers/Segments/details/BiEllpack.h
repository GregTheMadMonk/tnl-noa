/***************************************************************************
                          BiEllpack.h -  description
                             -------------------
    begin                : Apr 7, 2020
    copyright            : (C) 2020 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <type_traits>
#include <TNL/Containers/Vector.h>
#include <TNL/Containers/Segments/BiEllpackSegmentView.h>
#include <TNL/Containers/Segments/details/CheckLambdas.h>

namespace TNL {
   namespace Containers {
      namespace Segments {
         namespace details {

template< typename Index,
          typename Device,
          bool RowMajorOrder = std::is_same< Device, Devices::Host >::value,
          int WarpSize = 32 >
class BiEllpack
{
   public:

      using DeviceType = Device;
      using IndexType = Index;
      static constexpr bool getRowMajorOrder() { return RowMajorOrder; }
      using OffsetsHolder = Containers::Vector< IndexType, DeviceType, IndexType >;
      using OffsetsHolderView = typename OffsetsHolder::ViewType;
      using SegmentsSizes = OffsetsHolder;
      using SegmentViewType = BiEllpackSegmentView< IndexType, RowMajorOrder >;

      __cuda_callable__ static
      IndexType getSegmentSizeDirect( const OffsetsHolderView& rowPermArray,
                                      const OffsetsHolderView& groupPointers,
                                      const IndexType segmentIdx )
      {
      }

      static
      IndexType getSegmentSize( const OffsetsHolderView& rowPermArray,
                                const OffsetsHolderView& groupPointers,
                                const IndexType segmentIdx )
      {
      }

      __cuda_callable__ static
      IndexType getGlobalIndexDirect( const OffsetsHolderView& rowPermArray,
                                      const OffsetsHolderView& groupPointers,
                                      const IndexType segmentIdx,
                                      const IndexType localIdx )
      {
      }

      static
      IndexType getGlobalIndex( const OffsetsHolderView& rowPermArray,
                                const OffsetsHolderView& groupPointers,
                                const IndexType segmentIdx,
                                const IndexType localIdx )
      {
      }

      static __cuda_callable__
      SegmentViewType getSegmentViewDirect( const OffsetsHolderView& rowPermArray,
                                            const OffsetsHolderView& groupPointers,
                                            const IndexType segmentIdx )
      {
      }

      static __cuda_callable__
      SegmentViewType getSegmentView( const OffsetsHolderView& rowPermArray,
                                      const OffsetsHolderView& groupPointers,
                                      const IndexType segmentIdx )
      {
      }
};

#ifdef HAVE_CUDA
template< typename Index,
          typename Fetch,
          bool HasAllParameters = details::CheckFetchLambda< Index, Fetch >::hasAllParameters(),
          int WarpSize = 32 >
struct BiEllpackSegmentsReductionDispatcher{};

template< typename Index, typename Fetch >
struct BiEllpackSegmentsReductionDispatcher< Index, Fetch, true >
{
   template< typename View,
             typename Reduction,
             typename ResultKeeper,
             typename Real,
             typename... Args >
   __device__
   static void exec( View chunkedEllpack,
                     Index gridIdx,
                     Index first,
                     Index last,
                     Fetch fetch,
                     Reduction reduction,
                     ResultKeeper keeper,
                     Real zero,
                     Args... args )
   {
      chunkedEllpack.segmentsReductionKernelWithAllParameters( gridIdx, first, last, fetch, reduction, keeper, zero, args... );
   }
};

template< typename Index, typename Fetch >
struct BiEllpackSegmentsReductionDispatcher< Index, Fetch, false >
{
   template< typename View,
             typename Reduction,
             typename ResultKeeper,
             typename Real,
             typename... Args >
   __device__
   static void exec( View chunkedEllpack,
                     Index gridIdx,
                     Index first,
                     Index last,
                     Fetch fetch,
                     Reduction reduction,
                     ResultKeeper keeper,
                     Real zero,
                     Args... args )
   {
      chunkedEllpack.segmentsReductionKernel( gridIdx, first, last, fetch, reduction, keeper, zero, args... );
   }
};

template< typename View,
          typename Index,
          typename Fetch,
          typename Reduction,
          typename ResultKeeper,
          typename Real,
          typename... Args >
__global__
void BiEllpackSegmentsReductionKernel( View chunkedEllpack,
                                            Index gridIdx,
                                            Index first,
                                            Index last,
                                            Fetch fetch,
                                            Reduction reduction,
                                            ResultKeeper keeper,
                                            Real zero,
                                            Args... args )
{
   BiEllpackSegmentsReductionDispatcher< Index, Fetch >::exec( chunkedEllpack, gridIdx, first, last, fetch, reduction, keeper, zero, args... );
}
#endif

         } //namespace details
      } //namespace Segments
   } //namespace Containers
} //namepsace TNL
