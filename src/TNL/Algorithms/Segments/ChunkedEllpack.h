/***************************************************************************
                          ChunkedEllpack.h -  description
                             -------------------
    begin                : Mar 21, 2020
    copyright            : (C) 2020 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Allocators/Default.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Algorithms/Segments/ChunkedEllpackView.h>
#include <TNL/Algorithms/Segments/SegmentView.h>

namespace TNL {
   namespace Algorithms {
      namespace Segments {

template< typename Device,
          typename Index,
          typename IndexAllocator = typename Allocators::Default< Device >::template Allocator< Index >,
          ElementsOrganization Organization = Algorithms::Segments::DefaultElementsOrganization< Device >::getOrganization() >
class ChunkedEllpack
{
   public:

      using DeviceType = Device;
      using IndexType = std::remove_const_t< Index >;
      using OffsetsHolder = Containers::Vector< Index, DeviceType, IndexType, IndexAllocator >;
      static constexpr ElementsOrganization getOrganization() { return Organization; }
      using ViewType = ChunkedEllpackView< Device, Index, Organization >;
      template< typename Device_, typename Index_ >
      using ViewTemplate = ChunkedEllpackView< Device_, Index_, Organization >;
      using ConstViewType = ChunkedEllpackView< Device, std::add_const_t< IndexType >, Organization >;
      using SegmentViewType = typename ViewType::SegmentViewType;
      using ChunkedEllpackSliceInfoType = typename ViewType::ChunkedEllpackSliceInfoType; // detail::ChunkedEllpackSliceInfo< IndexType >;
      //TODO: using ChunkedEllpackSliceInfoAllocator = typename IndexAllocatorType::retype< ChunkedEllpackSliceInfoType >;
      using ChunkedEllpackSliceInfoAllocator = typename ViewType::ChunkedEllpackSliceInfoAllocator; // typename Allocators::Default< Device >::template Allocator< ChunkedEllpackSliceInfoType >;
      using ChunkedEllpackSliceInfoContainer = typename ViewType::ChunkedEllpackSliceInfoContainer; // Containers::Array< ChunkedEllpackSliceInfoType, DeviceType, IndexType, ChunkedEllpackSliceInfoAllocator >;

      static constexpr bool havePadding() { return true; };

      ChunkedEllpack() = default;

      ChunkedEllpack( const Containers::Vector< IndexType, DeviceType, IndexType >& sizes );

      ChunkedEllpack( const ChunkedEllpack& segments );

      ChunkedEllpack( const ChunkedEllpack&& segments );

      static String getSerializationType();

      static String getSegmentsType();

      ViewType getView();

      const ConstViewType getConstView() const;

      /**
       * \brief Number of segments.
       */
      __cuda_callable__
      IndexType getSegmentsCount() const;

      /**
       * \brief Set sizes of particular segments.
       */
      template< typename SizesHolder = OffsetsHolder >
      void setSegmentsSizes( const SizesHolder& sizes );

      void reset();

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
      void forAllElements( Function&& f ) const;

      template< typename Function >
      void forSegments( IndexType begin, IndexType end, Function&& f ) const;

      template< typename Function >
      void forAllSegments( Function&& f ) const;

      /***
       * \brief Go over all segments and perform a reduction in each of them.
       */
      template< typename Fetch, typename Reduction, typename ResultKeeper, typename Real, typename... Args >
      void segmentsReduction( IndexType first, IndexType last, Fetch& fetch, const Reduction& reduction, ResultKeeper& keeper, const Real& zero, Args... args ) const;

      template< typename Fetch, typename Reduction, typename ResultKeeper, typename Real, typename... Args >
      void allReduction( Fetch& fetch, const Reduction& reduction, ResultKeeper& keeper, const Real& zero, Args... args ) const;

      ChunkedEllpack& operator=( const ChunkedEllpack& source ) = default;

      template< typename Device_, typename Index_, typename IndexAllocator_, ElementsOrganization Organization_ >
      ChunkedEllpack& operator=( const ChunkedEllpack< Device_, Index_, IndexAllocator_, Organization_ >& source );

      void save( File& file ) const;

      void load( File& file );

      void printStructure( std::ostream& str ); // TODO const;

   protected:

      template< typename SegmentsSizes >
      void resolveSliceSizes( SegmentsSizes& rowLengths );

      template< typename SegmentsSizes >
      bool setSlice( SegmentsSizes& rowLengths,
                     const IndexType sliceIdx,
                     IndexType& elementsToAllocation );

      IndexType size = 0, storageSize = 0;

      IndexType chunksInSlice = 256, desiredChunkSize = 16;

      /**
       * For each segment, this keeps index of the slice which contains the
       * segment.
       */
      OffsetsHolder rowToSliceMapping;

      /**
       * For each row, this keeps index of the first chunk within a slice.
       */
      OffsetsHolder rowToChunkMapping;

      OffsetsHolder chunksToSegmentsMapping;

      /**
       * Keeps index of the first segment index.
       */
      OffsetsHolder rowPointers;

      ChunkedEllpackSliceInfoContainer slices;

      IndexType numberOfSlices = 0;

      template< typename Device_, typename Index_, typename IndexAllocator_, ElementsOrganization Organization_ >
      friend class ChunkedEllpack;
};

      } // namespace Segments
   }  // namespace Algorithms
} // namespace TNL

#include <TNL/Algorithms/Segments/ChunkedEllpack.hpp>
