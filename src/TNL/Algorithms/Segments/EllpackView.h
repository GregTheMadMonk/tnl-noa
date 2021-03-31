/***************************************************************************
                          EllpackView.h -  description
                             -------------------
    begin                : Dec 12, 2019
    copyright            : (C) 2019 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <type_traits>

#include <TNL/Containers/Vector.h>
#include <TNL/Algorithms/Segments/SegmentView.h>
#include <TNL/Algorithms/Segments/ElementsOrganization.h>


namespace TNL {
   namespace Algorithms {
      namespace Segments {

template< typename Device,
          typename Index,
          ElementsOrganization Organization = Segments::DefaultElementsOrganization< Device >::getOrganization(),
          int Alignment = 32 >
class EllpackView
{
   public:

      using DeviceType = Device;
      using IndexType = std::remove_const_t< Index >;
      static constexpr int getAlignment() { return Alignment; }
      static constexpr bool getOrganization() { return Organization; }
      using OffsetsHolder = Containers::Vector< IndexType, DeviceType, IndexType >;
      using SegmentsSizes = OffsetsHolder;
      template< typename Device_, typename Index_ >
      using ViewTemplate = EllpackView< Device_, Index_, Organization, Alignment >;
      using ViewType = EllpackView;
      using ConstViewType = ViewType;
      using SegmentViewType = SegmentView< IndexType, Organization >;

      static constexpr bool havePadding() { return true; };

      __cuda_callable__
      EllpackView();

      __cuda_callable__
      EllpackView( IndexType segmentsCount, IndexType segmentSize, IndexType alignedSize );

      __cuda_callable__
      EllpackView( IndexType segmentsCount, IndexType segmentSize );

      __cuda_callable__
      EllpackView( const EllpackView& ellpackView );

      __cuda_callable__
      EllpackView( const EllpackView&& ellpackView );

      static String getSerializationType();

      static String getSegmentsType();

      __cuda_callable__
      ViewType getView();

      __cuda_callable__
      const ConstViewType getConstView() const;

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
      SegmentViewType getSegmentView( const IndexType segmentIdx ) const;

      /***
       * \brief Go over all segments and for each segment element call
       * function 'f' with arguments 'args'. The return type of 'f' is bool.
       * When its true, the for-loop continues. Once 'f' returns false, the for-loop
       * is terminated.
       */
      template< typename Function >
      void forElements( IndexType begin, IndexType end, Function&& f ) const;

      template< typename Function >
      void forAllElements( Function&& f ) const;

      template< typename Function >
      void forSegments( IndexType begin, IndexType end, Function&& f ) const;

      template< typename Function >
      void forAllSegments( Function&& f ) const;

      /***
       * \brief Go over all segments and perform a reduction in each of them.
       */
      template< typename Fetch, typename Reduction, typename ResultKeeper, typename Real >
      void reduceSegments( IndexType first, IndexType last, Fetch& fetch, const Reduction& reduction, ResultKeeper& keeper, const Real& zero ) const;

      template< typename Fetch, typename Reduction, typename ResultKeeper, typename Real >
      void reduceAllSegments( Fetch& fetch, const Reduction& reduction, ResultKeeper& keeper, const Real& zero ) const;

      EllpackView& operator=( const EllpackView& view );

      void save( File& file ) const;

      void load( File& file );

   protected:

      IndexType segmentSize, segmentsCount, alignedSize;
};

      } // namespace Segments
   }  // namespace Algorithms
} // namespace TNL

#include <TNL/Algorithms/Segments/EllpackView.hpp>
