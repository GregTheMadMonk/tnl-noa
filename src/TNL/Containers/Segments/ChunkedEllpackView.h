/***************************************************************************
                          ChunkedEllpackView.h -  description
                             -------------------
    begin                : Mar 21, 2020
    copyright            : (C) 2020 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <type_traits>

#include <TNL/Containers/Vector.h>
#include <TNL/Containers/Segments/ChunkedEllpackSegmentView.h>

namespace TNL {
   namespace Containers {
      namespace Segments {

/***
 * In the ChunkedEllpack, the segments are split into slices. This is done
 * in ChunkedEllpack::resolveSliceSizes. All segments elements in each slice
 * are split into chunks. All chunks in one slice have the same size, but the size
 * of chunks can be different in each slice.
 */
template< typename Index >
struct ChunkedEllpackSliceInfo
{
   /**
    * The size of the slice, it means the number of the matrix rows covered by
    * the slice.
    */
   Index size;

   /**
    * The chunk size, i.e. maximal number of non-zero elements that can be stored
    * in the chunk.
    */
   Index chunkSize;

   /**
    * Index of the first segment covered be this slice.
    */
   Index firstSegment;

   /**
    * Position of the first element of this slice.
    */
   Index pointer;
};

template< typename Device,
          typename Index,
          bool RowMajorOrder = std::is_same< Device, Devices::Host >::value >
class ChunkedEllpackView
{
   public:

      using DeviceType = Device;
      using IndexType = Index;
      using OffsetsView = typename Containers::VectorView< IndexType, DeviceType, typename std::remove_const< IndexType >::type >;
      using ConstOffsetsView = typename Containers::Vector< IndexType, DeviceType, typename std::remove_const< IndexType >::type >::ConstViewType;
      using ViewType = ChunkedEllpackView;
      template< typename Device_, typename Index_ >
      using ViewTemplate = ChunkedEllpackView< Device_, Index_ >;
      using ConstViewType = ChunkedEllpackView< Device, std::add_const_t< Index > >;
      using SegmentViewType = ChunkedEllpackSegmentView< IndexType >;
      using ChunkedEllpackSliceInfoType = ChunkedEllpackSliceInfo< IndexType >;
      using ChunkedEllpackSliceInfoAllocator = typename Allocators::Default< Device >::template Allocator< ChunkedEllpackSliceInfoType >;
      using ChunkedEllpackSliceInfoContainer = Containers::Array< ChunkedEllpackSliceInfoType, DeviceType, IndexType, ChunkedEllpackSliceInfoAllocator >;
      using ChunkedEllpackSliceInfoContainerView = typename ChunkedEllpackSliceInfoContainer::ViewType;

      __cuda_callable__
      ChunkedEllpackView() = default;

      __cuda_callable__
      ChunkedEllpackView( const IndexType size,
                          const IndexType storageSize,
                          const IndexType chunksInSlice,
                          const IndexType desiredChunkSize,
                          const OffsetsView& rowToChunkMapping,
                          const OffsetsView& rowToSliceMapping,
                          const OffsetsView& rowPointers,
                          const ChunkedEllpackSliceInfoContainerView& slices,
                          const IndexType numberOfSlices );

      __cuda_callable__
      ChunkedEllpackView( const IndexType size,
                          const IndexType storageSize,
                          const IndexType chunksInSlice,
                          const IndexType desiredChunkSize,
                          const OffsetsView&& rowToChunkMapping,
                          const OffsetsView&& rowToSliceMapping,
                          const OffsetsView&& rowPointers,
                          const ChunkedEllpackSliceInfoContainerView&& slices,
                          const IndexType numberOfSlices );

      __cuda_callable__
      ChunkedEllpackView( const ChunkedEllpackView& chunked_ellpack_view );

      __cuda_callable__
      ChunkedEllpackView( const ChunkedEllpackView&& chunked_ellpack_view );

      static String getSerializationType();

      static String getSegmentsType();

      __cuda_callable__
      ViewType getView();

      __cuda_callable__
      ConstViewType getConstView() const;

      /**
       * \brief Number segments.
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

      ChunkedEllpackView& operator=( const ChunkedEllpackView& view );

      void save( File& file ) const;

      void load( File& file );

      void printStructure( std::ostream& str ) const;

   protected:

      /*IndexType size;

      IndexType chunksInSlice, desiredChunkSize;

      Containers::VectorView< Index, Device, Index > rowToChunkMapping, rowToSliceMapping, rowPointers;

      Containers::ArrayView< ChunkedEllpackSliceInfoType, Device, Index > slices;

      IndexType numberOfSlices;*/

      IndexType size = 0, storageSize = 0;

      IndexType chunksInSlice = 256, desiredChunkSize = 16;

      /**
       * For each segment, this keeps index of the slice which contains the
       * segment.
       */
      OffsetsView rowToSliceMapping;

      /**
       * For each row, this keeps index of the first chunk within a slice.
       */
      OffsetsView rowToChunkMapping;

      /**
       * Keeps index of the first segment index.
       */
      OffsetsView rowPointers;

      ChunkedEllpackSliceInfoContainerView slices;

      IndexType numberOfSlices;
};
      } // namespace Segements
   }  // namespace Conatiners
} // namespace TNL

#include <TNL/Containers/Segments/ChunkedEllpackView.hpp>
