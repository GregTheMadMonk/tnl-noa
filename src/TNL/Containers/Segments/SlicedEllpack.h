/***************************************************************************
                          SlicedEllpack.h -  description
                             -------------------
    begin                : Dec 4, 2019
    copyright            : (C) 2019 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Allocators/Default.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Containers/Segments/SlicedEllpackView.h>
#include <TNL/Containers/Segments/SegmentView.h>

namespace TNL {
   namespace Containers {
      namespace Segments {

template< typename Device,
          typename Index,
          typename IndexAllocator = typename Allocators::Default< Device >::template Allocator< Index >,
          bool RowMajorOrder = std::is_same< Device, Devices::Host >::value,
          int SliceSize = 32 >
class SlicedEllpack
{
   public:

      using DeviceType = Device;
      using IndexType = std::remove_const_t< Index >;
      using OffsetsHolder = Containers::Vector< Index, DeviceType, IndexType, IndexAllocator >;
      static constexpr int getSliceSize() { return SliceSize; }
      static constexpr bool getRowMajorOrder() { return RowMajorOrder; }
      using ViewType = SlicedEllpackView< Device, Index, RowMajorOrder, SliceSize >;
      template< typename Device_, typename Index_ >
      using ViewTemplate = SlicedEllpackView< Device_, Index_, RowMajorOrder, SliceSize >;
      using ConstViewType = SlicedEllpackView< Device, std::add_const_t< Index >, RowMajorOrder, SliceSize >;
      using SegmentViewType = SegmentView< IndexType, RowMajorOrder >;

      SlicedEllpack();

      SlicedEllpack( const Vector< IndexType, DeviceType, IndexType >& sizes );

      SlicedEllpack( const SlicedEllpack& segments );

      SlicedEllpack( const SlicedEllpack&& segments );

      static String getSerializationType();

      static String getSegmentsType();

      ViewType getView();

      const ConstViewType getConstView() const;

      /**
       * \brief Set sizes of particular segments.
       */
      template< typename SizesHolder = OffsetsHolder >
      void setSegmentsSizes( const SizesHolder& sizes );

      __cuda_callable__
      IndexType getSegmentsCount() const;

      __cuda_callable__
      IndexType getSegmentSize( const IndexType segmentIdx ) const;

      /**
       * \brief Number segments.
       */
      __cuda_callable__
      IndexType getSize() const;


      __cuda_callable__
      IndexType getStorageSize() const;

      __cuda_callable__
      IndexType getGlobalIndex( const Index segmentIdx, const Index localIdx ) const;

      __cuda_callable__
      void getSegmentAndLocalIndex( const Index globalIdx, Index& segmentIdx, Index& localIdx ) const;

      __cuda_callable__
      SegmentViewType getSegmentView( const IndexType segmentIdx ) const;

      /***
       * \brief Go over all segments and for each segment element call
       * function 'f' with arguments 'args'. The return type of 'f' is bool.
       * When its true, the for-loop continues. Once 'f' returns false, the for-loop
       * is terminated.
       */
      template< typename Function, typename... Args >
      void forSegments( IndexType first, IndexType last, Function& f, Args... args ) const;

      template< typename Function, typename... Args >
      void forAll( Function& f, Args... args ) const;


      /***
       * \brief Go over all segments and perform a reduction in each of them.
       */
      template< typename Fetch, typename Reduction, typename ResultKeeper, typename Real, typename... Args >
      void segmentsReduction( IndexType first, IndexType last, Fetch& fetch, Reduction& reduction, ResultKeeper& keeper, const Real& zero, Args... args ) const;

      template< typename Fetch, typename Reduction, typename ResultKeeper, typename Real, typename... Args >
      void allReduction( Fetch& fetch, Reduction& reduction, ResultKeeper& keeper, const Real& zero, Args... args ) const;

      SlicedEllpack& operator=( const SlicedEllpack& source ) = default;

      template< typename Device_, typename Index_, typename IndexAllocator_, bool RowMajorOrder_ >
      SlicedEllpack& operator=( const SlicedEllpack< Device_, Index_, IndexAllocator_, RowMajorOrder_, SliceSize >& source );

      void save( File& file ) const;

      void load( File& file );

   protected:

      IndexType size, alignedSize, segmentsCount;

      OffsetsHolder sliceOffsets, sliceSegmentSizes;
};

      } // namespace Segements
   }  // namespace Conatiners
} // namespace TNL

#include <TNL/Containers/Segments/SlicedEllpack.hpp>
