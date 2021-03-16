/***************************************************************************
                          BiEllpackView.h -  description
                             -------------------
    begin                : Apr 5, 2020
    copyright            : (C) 2020 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <type_traits>

#include <TNL/Containers/Vector.h>
#include <TNL/Algorithms/Segments/ElementsOrganization.h>
#include <TNL/Algorithms/Segments/BiEllpackSegmentView.h>
#include <TNL/Algorithms/Segments/details/BiEllpack.h>

namespace TNL {
   namespace Algorithms {
      namespace Segments {


template< typename Device,
          typename Index,
          ElementsOrganization Organization = Algorithms::Segments::DefaultElementsOrganization< Device >::getOrganization(),
          int WarpSize = 32 >
class BiEllpackView
{
   public:

      using DeviceType = Device;
      using IndexType = std::remove_const_t< Index >;
      using OffsetsView = typename Containers::VectorView< IndexType, DeviceType, IndexType >;
      using ConstOffsetsView = typename OffsetsView::ConstViewType;
      using ViewType = BiEllpackView;
      template< typename Device_, typename Index_ >
      using ViewTemplate = BiEllpackView< Device_, Index_, Organization, WarpSize >;
      using ConstViewType = BiEllpackView< Device, std::add_const_t< Index >, Organization, WarpSize >;
      using SegmentViewType = BiEllpackSegmentView< IndexType, Organization, WarpSize >;

      static constexpr bool havePadding() { return true; };

      __cuda_callable__
      BiEllpackView() = default;

      __cuda_callable__
      BiEllpackView( const IndexType size,
                     const IndexType storageSize,
                     const IndexType virtualRows,
                     const OffsetsView& rowPermArray,
                     const OffsetsView& groupPointers );

      __cuda_callable__
      BiEllpackView( const IndexType size,
                     const IndexType storageSize,
                     const IndexType virtualRows,
                     const OffsetsView&& rowPermArray,
                     const OffsetsView&& groupPointers );

      __cuda_callable__
      BiEllpackView( const BiEllpackView& chunked_ellpack_view );

      __cuda_callable__
      BiEllpackView( const BiEllpackView&& chunked_ellpack_view );

      static String getSerializationType();

      static String getSegmentsType();

      __cuda_callable__
      ViewType getView();

      __cuda_callable__
      const ConstViewType getConstView() const;

      /**
       * \brief Number of segments.
       */
      __cuda_callable__
      IndexType getSegmentsCount() const;

      /***
       * \brief Returns size of the segment number \r segmentIdx
       */
      __cuda_callable__
      IndexType getSegmentSize( const IndexType segmentIdx ) const;

      /***
       * \brief Returns number of elements managed by all segments.
       */
      __cuda_callable__
      IndexType getSize() const;

      /***
       * \brief Returns number of elements that needs to be allocated.
       */
      __cuda_callable__
      IndexType getStorageSize() const;

      __cuda_callable__
      IndexType getGlobalIndex( const Index segmentIdx, const Index localIdx ) const;

      __cuda_callable__
      SegmentViewType getSegmentView( const IndexType segmentIdx ) const;

      /***
       * \brief Go over all segments and for each segment element call
       * function 'f' with arguments 'args'. The return type of 'f' is bool.
       * When its true, the for-loop continues. Once 'f' returns false, the for-loop
       * is terminated.
       */
      template< typename Function >
      void forElements( IndexType first, IndexType last, Function&& f ) const;

      template< typename Function >
      void forEachElement( Function&& f ) const;

      template< typename Function >
      void forSegments( IndexType begin, IndexType end, Function&& f ) const;

      template< typename Function >
      void forEachSegment( Function&& f ) const;

      /***
       * \brief Go over all segments and perform a reduction in each of them.
       */
      template< typename Fetch, typename Reduction, typename ResultKeeper, typename Real, typename... Args >
      void segmentsReduction( IndexType first, IndexType last, Fetch& fetch, const Reduction& reduction, ResultKeeper& keeper, const Real& zero, Args... args ) const;

      template< typename Fetch, typename Reduction, typename ResultKeeper, typename Real, typename... Args >
      void allReduction( Fetch& fetch, const Reduction& reduction, ResultKeeper& keeper, const Real& zero, Args... args ) const;

      BiEllpackView& operator=( const BiEllpackView& view );

      void save( File& file ) const;

      void load( File& file );

      void printStructure( std::ostream& str ) const;

   protected:

      static constexpr int getWarpSize() { return WarpSize; };

      static constexpr int getLogWarpSize() { return std::log2( WarpSize ); };

      IndexType size = 0, storageSize = 0;

      IndexType virtualRows = 0;

      OffsetsView rowPermArray;

      OffsetsView groupPointers;

#ifdef HAVE_CUDA
      template< typename Fetch,
                typename Reduction,
                typename ResultKeeper,
                typename Real,
                int BlockDim,
                typename... Args >
      __device__
      void segmentsReductionKernelWithAllParameters( IndexType gridIdx,
                                                     IndexType first,
                                                     IndexType last,
                                                     Fetch fetch,
                                                     Reduction reduction,
                                                     ResultKeeper keeper,
                                                     Real zero,
                                                     Args... args ) const;

      template< typename Fetch,
                typename Reduction,
                typename ResultKeeper,
                typename Real_,
                int BlockDim,
                typename... Args >
      __device__
      void segmentsReductionKernel( IndexType gridIdx,
                                    IndexType first,
                                    IndexType last,
                                    Fetch fetch,
                                    Reduction reduction,
                                    ResultKeeper keeper,
                                    Real_ zero,
                                    Args... args ) const;

      template< typename View_,
                typename Index_,
                typename Fetch_,
                typename Reduction_,
                typename ResultKeeper_,
                typename Real_,
                int BlockDim,
                typename... Args_ >
      friend __global__
      void BiEllpackSegmentsReductionKernel( View_ chunkedEllpack,
                                             Index_ gridIdx,
                                             Index_ first,
                                             Index_ last,
                                             Fetch_ fetch,
                                             Reduction_ reduction,
                                             ResultKeeper_ keeper,
                                             Real_ zero,
                                             Args_... args );

      template< typename Index_, typename Fetch_, int BlockDim_, int WarpSize_, bool B_ >
      friend struct details::BiEllpackSegmentsReductionDispatcher;
#endif
};
      } // namespace Segments
   }  // namespace Algorithms
} // namespace TNL

#include <TNL/Algorithms/Segments/BiEllpackView.hpp>
