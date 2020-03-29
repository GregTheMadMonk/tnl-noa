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
#include <TNL/Containers/Segments/ChunkedEllpackView.h>
#include <TNL/Containers/Segments/SegmentView.h>

namespace TNL {
   namespace Containers {
      namespace Segments {

template< typename Device,
          typename Index,
          typename IndexAllocator = typename Allocators::Default< Device >::template Allocator< Index >,
          bool RowMajorOrder = std::is_same< Device, Devices::Host >::value >
class ChunkedEllpack
{
   public:

      using DeviceType = Device;
      using IndexType = Index;
      using OffsetsHolder = Containers::Vector< IndexType, DeviceType, typename std::remove_const< IndexType >::type, IndexAllocator >;
      static constexpr bool getRowMajorOrder() { return RowMajorOrder; }
      using ViewType = ChunkedEllpackView< Device, Index, RowMajorOrder >;
      template< typename Device_, typename Index_ >
      using ViewTemplate = ChunkedEllpackView< Device_, Index_, RowMajorOrder >;
      using ConstViewType = ChunkedEllpackView< Device, std::add_const_t< Index >, RowMajorOrder >;
      using SegmentViewType = SegmentView< IndexType, RowMajorOrder >;
      using ChunkedEllpackSliceInfoType = details::ChunkedEllpackSliceInfo< IndexType >;
      //TODO: using ChunkedEllpackSliceInfoAllocator = typename IndexAllocatorType::retype< ChunkedEllpackSliceInfoType >;
      using ChunkedEllpackSliceInfoAllocator = typename Allocators::Default< Device >::template Allocator< ChunkedEllpackSliceInfoType >;
      using ChunkedEllpackSliceInfoContainer = Containers::Array< ChunkedEllpackSliceInfoType, DeviceType, IndexType, ChunkedEllpackSliceInfoAllocator >;

      ChunkedEllpack() = default;

      ChunkedEllpack( const Vector< IndexType, DeviceType, IndexType >& sizes );

      ChunkedEllpack( const ChunkedEllpack& segments );

      ChunkedEllpack( const ChunkedEllpack&& segments );

      static String getSerializationType();

      static String getSegmentsType();

      ViewType getView();

      ConstViewType getConstView() const;

      /**
       * \brief Set sizes of particular segments.
       */
      template< typename SizesHolder = OffsetsHolder >
      void setSegmentsSizes( const SizesHolder& sizes );

      __cuda_callable__
      IndexType getSegmentsCount() const;

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

      ChunkedEllpack& operator=( const ChunkedEllpack& source ) = default;

      template< typename Device_, typename Index_, typename IndexAllocator_, bool RowMajorOrder_ >
      ChunkedEllpack& operator=( const ChunkedEllpack< Device_, Index_, IndexAllocator_, RowMajorOrder_ >& source );

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

      IndexType numberOfSlices;

      template< typename Device_, typename Index_, typename IndexAllocator_, bool RowMajorOrder_ >
      friend class ChunkedEllpack;
};

      } // namespace Segements
   }  // namespace Conatiners
} // namespace TNL

#include <TNL/Containers/Segments/ChunkedEllpack.hpp>
