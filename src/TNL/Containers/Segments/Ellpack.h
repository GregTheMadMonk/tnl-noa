/***************************************************************************
                          Ellpack.h -  description
                             -------------------
    begin                : Dec 3, 2019
    copyright            : (C) 2019 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Containers/Vector.h>
#include <TNL/Containers/Segments/EllpackView.h>
#include <TNL/Containers/Segments/SegmentView.h>

namespace TNL {
   namespace Containers {
      namespace Segments {

template< typename Device,
          typename Index,
          typename IndexAllocator = typename Allocators::Default< Device >::template Allocator< Index >,
          bool RowMajorOrder = std::is_same< Device, Devices::Host >::value,
          int Alignment = 32 >
class Ellpack
{
   public:

      using DeviceType = Device;
      using IndexType = std::remove_const_t< Index >;
      static constexpr int getAlignment() { return Alignment; }
      static constexpr bool getRowMajorOrder() { return RowMajorOrder; }
      using OffsetsHolder = Containers::Vector< IndexType, DeviceType, IndexType >;
      using SegmentsSizes = OffsetsHolder;
      template< typename Device_, typename Index_ >
      using ViewTemplate = EllpackView< Device_, Index_, RowMajorOrder, Alignment >;
      using ViewType = EllpackView< Device, Index, RowMajorOrder, Alignment >;
      using ConstViewType = typename ViewType::ConstViewType;
      using SegmentViewType = SegmentView< IndexType, RowMajorOrder >;

      Ellpack();

      Ellpack( const SegmentsSizes& sizes );

      Ellpack( const IndexType segmentsCount, const IndexType segmentSize );

      Ellpack( const Ellpack& segments );

      Ellpack( const Ellpack&& segments );

      static String getSerializationType();

      static String getSegmentsType();

      ViewType getView();

      const ConstViewType getConstView() const;

      /**
       * \brief Set sizes of particular segments.
       */
      template< typename SizesHolder = OffsetsHolder >
      void setSegmentsSizes( const SizesHolder& sizes );

      void setSegmentsSizes( const IndexType segmentsCount, const IndexType segmentSize );
      /**
       * \brief Number segments.
       */
      __cuda_callable__
      IndexType getSegmentsCount() const;

      __cuda_callable__
      IndexType getSegmentSize( const IndexType segmentIdx ) const;

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
      void segmentsReduction( IndexType first, IndexType last, Fetch& fetch, const Reduction& reduction, ResultKeeper& keeper, const Real& zero, Args... args ) const;

      template< typename Fetch, typename Reduction, typename ResultKeeper, typename Real, typename... Args >
      void allReduction( Fetch& fetch, const Reduction& reduction, ResultKeeper& keeper, const Real& zero, Args... args ) const;

      Ellpack& operator=( const Ellpack& source ) = default;

      template< typename Device_, typename Index_, typename IndexAllocator_, bool RowMajorOrder_, int Alignment_ >
      Ellpack& operator=( const Ellpack< Device_, Index_, IndexAllocator_, RowMajorOrder_, Alignment_ >& source );

      void save( File& file ) const;

      void load( File& file );

   protected:

      IndexType segmentSize, size, alignedSize;
};

      } // namespace Segements
   }  // namespace Containers
} // namespace TNL

#include <TNL/Containers/Segments/Ellpack.hpp>
