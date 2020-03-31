/***************************************************************************
                          ChunkedEllpack.h -  description
                             -------------------
    begin                : Mar 25, 2020
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
         namespace details {

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
    * The size of the slice, it means the number of the segments covered by
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

         
template< typename Index,
          typename Device,
          bool RowMajorOrder = std::is_same< Device, Devices::Host >::value >
class ChunkedEllpack
{
   public:

      using DeviceType = Device;
      using IndexType = Index;
      static constexpr bool getRowMajorOrder() { return RowMajorOrder; }
      using OffsetsHolder = Containers::Vector< IndexType, DeviceType, IndexType >;
      using OffsetsHolderView = typename OffsetsHolder::ViewType;
      using SegmentsSizes = OffsetsHolder;
      using ChunkedEllpackSliceInfoType = details::ChunkedEllpackSliceInfo< IndexType >;
      using ChunkedEllpackSliceInfoAllocator = typename Allocators::Default< Device >::template Allocator< ChunkedEllpackSliceInfoType >;
      using ChunkedEllpackSliceInfoContainer = Containers::Array< ChunkedEllpackSliceInfoType, DeviceType, IndexType, ChunkedEllpackSliceInfoAllocator >;
      using ChunkedEllpackSliceInfoContainerView = typename ChunkedEllpackSliceInfoContainer::ViewType;
      using SegmentViewType = ChunkedEllpackSegmentView< IndexType, RowMajorOrder >;

      __cuda_callable__ static
      IndexType getSegmentSizeDirect( const OffsetsHolderView& segmentsToSlicesMapping,
                                      const ChunkedEllpackSliceInfoContainerView& slices,
                                      const OffsetsHolderView& segmentsToChunksMapping,
                                      const IndexType segmentIdx )
      {
         const IndexType& sliceIndex = segmentsToSlicesMapping[ segmentIdx ];
         IndexType firstChunkOfSegment( 0 );
         if( segmentIdx != slices[ sliceIndex ].firstSegment )
            firstChunkOfSegment = segmentsToChunksMapping[ segmentIdx - 1 ];

         const IndexType lastChunkOfSegment = segmentsToChunksMapping[ segmentIdx ];
         const IndexType segmentChunksCount = lastChunkOfSegment - firstChunkOfSegment;
         const IndexType chunkSize = slices[ sliceIndex ].chunkSize;
         return chunkSize * segmentChunksCount;
      }

      static
      IndexType getSegmentSize( const OffsetsHolderView& segmentsToSlicesMapping,
                                const ChunkedEllpackSliceInfoContainerView& slices,
                                const OffsetsHolderView& segmentsToChunksMapping,
                                const IndexType segmentIdx )
      {
         const IndexType& sliceIndex = segmentsToSlicesMapping.getElement( segmentIdx );
         IndexType firstChunkOfSegment( 0 );
         if( segmentIdx != slices.getElement( sliceIndex ).firstSegment )
            firstChunkOfSegment = segmentsToChunksMapping.getElement( segmentIdx - 1 );

         const IndexType lastChunkOfSegment = segmentsToChunksMapping.getElement( segmentIdx );
         const IndexType segmentChunksCount = lastChunkOfSegment - firstChunkOfSegment;
         const IndexType chunkSize = slices.getElement( sliceIndex ).chunkSize;
         return chunkSize * segmentChunksCount;
      }

      __cuda_callable__ static
      IndexType getGlobalIndexDirect( const OffsetsHolderView& segmentsToSlicesMapping,
                                      const ChunkedEllpackSliceInfoContainerView& slices,
                                      const OffsetsHolderView& segmentsToChunksMapping,
                                      const IndexType chunksInSlice,
                                      const IndexType segmentIdx,
                                      const IndexType localIdx )
      {
         const IndexType& sliceIndex = segmentsToSlicesMapping[ segmentIdx ];
         IndexType firstChunkOfSegment( 0 );
         if( segmentIdx != slices[ sliceIndex ].firstSegment )
            firstChunkOfSegment = segmentsToChunksMapping[ segmentIdx - 1 ];

         const IndexType lastChunkOfSegment = segmentsToChunksMapping[ segmentIdx ];
         const IndexType segmentChunksCount = lastChunkOfSegment - firstChunkOfSegment;
         const IndexType sliceOffset = slices[ sliceIndex ].pointer;
         const IndexType chunkSize = slices[ sliceIndex ].chunkSize;
         TNL_ASSERT_LE( localIdx, segmentChunksCount * chunkSize, "" );

         if( RowMajorOrder )
            return sliceOffset + firstChunkOfSegment * chunkSize + localIdx;
         else
         {
            const IndexType inChunkOffset = localIdx % chunkSize;
            const IndexType chunkIdx = localIdx / chunkSize;
            return sliceOffset + inChunkOffset * chunksInSlice + firstChunkOfSegment + chunkIdx;
         }
      }

      static
      IndexType getGlobalIndex( const OffsetsHolderView& segmentsToSlicesMapping,
                                const ChunkedEllpackSliceInfoContainerView& slices,
                                const OffsetsHolderView& segmentsToChunksMapping,
                                const IndexType chunksInSlice,
                                const IndexType segmentIdx,
                                const IndexType localIdx )
      {
         const IndexType& sliceIndex = segmentsToSlicesMapping.getElement( segmentIdx );
         IndexType firstChunkOfSegment( 0 );
         if( segmentIdx != slices.getElement( sliceIndex ).firstSegment )
            firstChunkOfSegment = segmentsToChunksMapping.getElement( segmentIdx - 1 );

         const IndexType lastChunkOfSegment = segmentsToChunksMapping.getElement( segmentIdx );
         const IndexType segmentChunksCount = lastChunkOfSegment - firstChunkOfSegment;
         const IndexType sliceOffset = slices.getElement( sliceIndex ).pointer;
         const IndexType chunkSize = slices.getElement( sliceIndex ).chunkSize;
         TNL_ASSERT_LE( localIdx, segmentChunksCount * chunkSize, "" );

         if( RowMajorOrder )
            return sliceOffset + firstChunkOfSegment * chunkSize + localIdx;
         else
         {
            const IndexType inChunkOffset = localIdx % chunkSize;
            const IndexType chunkIdx = localIdx / chunkSize;
            return sliceOffset + inChunkOffset * chunksInSlice + firstChunkOfSegment + chunkIdx;
         }
      }

      static __cuda_callable__
      SegmentViewType getSegmentViewDirect( const OffsetsHolderView& segmentsToSlicesMapping,
                                            const ChunkedEllpackSliceInfoContainerView& slices,
                                            const OffsetsHolderView& segmentsToChunksMapping,
                                            const IndexType chunksInSlice,
                                            const IndexType segmentIdx )
      {
         const IndexType& sliceIndex = segmentsToSlicesMapping[ segmentIdx ];
         IndexType firstChunkOfSegment( 0 );
         if( segmentIdx != slices[ sliceIndex ].firstSegment )
            firstChunkOfSegment = segmentsToChunksMapping[ segmentIdx - 1 ];

         const IndexType lastChunkOfSegment = segmentsToChunksMapping[ segmentIdx ];
         const IndexType segmentChunksCount = lastChunkOfSegment - firstChunkOfSegment;
         const IndexType sliceOffset = slices[ sliceIndex ].pointer;
         const IndexType chunkSize = slices[ sliceIndex ].chunkSize;
         const IndexType segmentSize = segmentChunksCount * chunkSize;

         if( RowMajorOrder )
            return SegmentViewType( sliceOffset + firstChunkOfSegment * chunkSize,
                                    segmentSize,
                                    chunkSize,
                                    chunksInSlice );
         else
            return SegmentViewType( sliceOffset + firstChunkOfSegment,
                                    segmentSize,
                                    chunkSize,
                                    chunksInSlice );
      }

      static __cuda_callable__
      SegmentViewType getSegmentView( const OffsetsHolderView& segmentsToSlicesMapping,
                                      const ChunkedEllpackSliceInfoContainerView& slices,
                                      const OffsetsHolderView& segmentsToChunksMapping,
                                      const IndexType chunksInSlice,
                                      const IndexType segmentIdx )
      {
         const IndexType& sliceIndex = segmentsToSlicesMapping.getElement( segmentIdx );
         IndexType firstChunkOfSegment( 0 );
         if( segmentIdx != slices.getElement( sliceIndex ).firstSegment )
            firstChunkOfSegment = segmentsToChunksMapping.getElement( segmentIdx - 1 );

         const IndexType lastChunkOfSegment = segmentsToChunksMapping.getElement( segmentIdx );
         const IndexType segmentChunksCount = lastChunkOfSegment - firstChunkOfSegment;
         const IndexType sliceOffset = slices.getElement( sliceIndex ).pointer;
         const IndexType chunkSize = slices.getElement( sliceIndex ).chunkSize;
         const IndexType segmentSize = segmentChunksCount * chunkSize;

         if( RowMajorOrder )
            return SegmentViewType( sliceOffset + firstChunkOfSegment * chunkSize,
                                    segmentSize,
                                    chunkSize,
                                    chunksInSlice );
         else
            return SegmentViewType( sliceOffset + firstChunkOfSegment,
                                    segmentSize,
                                    chunkSize,
                                    chunksInSlice );
      }
};
         } //namespace details
      } //namespace Segments
   } //namespace Containers
} //namepsace TNL
