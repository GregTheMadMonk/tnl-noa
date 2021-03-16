/***************************************************************************
                          ChunkedEllpackSegmentView.h -  description
                             -------------------
    begin                : Mar 24, 2020
    copyright            : (C) 2020 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {
   namespace Algorithms {
      namespace Segments {

template< typename Index,
          ElementsOrganization Organization >
class ChunkedEllpackSegmentView;

template< typename Index >
class ChunkedEllpackSegmentView< Index, ColumnMajorOrder >
{
   public:

      using IndexType = Index;

      __cuda_callable__
      ChunkedEllpackSegmentView( const IndexType segmentIdx,
                                 const IndexType offset,
                                 const IndexType size,
                                 const IndexType chunkSize,      // this is only for compatibility with the following specialization
                                 const IndexType chunksInSlice ) // this one as well - both can be replaced when we could use constexprif in C++17
      : segmentIdx( segmentIdx ), segmentOffset( offset ), segmentSize( size ){};

      __cuda_callable__
      ChunkedEllpackSegmentView( const ChunkedEllpackSegmentView& view )
      : segmentIdx( view.segmentIdx ), segmentOffset( view.segmentOffset ), segmentSize( view.segmentSize ){};

      __cuda_callable__
      IndexType getSize() const
      {
         return this->segmentSize;
      };

      __cuda_callable__
      IndexType getGlobalIndex( const IndexType localIndex ) const
      {
         TNL_ASSERT_LT( localIndex, segmentSize, "Local index exceeds segment bounds." );
         return segmentOffset + localIndex;
      };

      __cuda_callable__
      const IndexType& getSegmentIndex() const
      {
         return this->segmentIdx;
      };

      protected:

         IndexType segmentIdx, segmentOffset, segmentSize;
};

template< typename Index >
class ChunkedEllpackSegmentView< Index, RowMajorOrder >
{
   public:

      using IndexType = Index;

      __cuda_callable__
      ChunkedEllpackSegmentView( const IndexType segmentIdx,
                                 const IndexType offset,
                                 const IndexType size,
                                 const IndexType chunkSize,
                                 const IndexType chunksInSlice )
      : segmentIdx( segmentIdx ), segmentOffset( offset ), segmentSize( size ),
        chunkSize( chunkSize ), chunksInSlice( chunksInSlice ){};

      __cuda_callable__
      ChunkedEllpackSegmentView( const ChunkedEllpackSegmentView& view )
      : segmentIdx( view.segmentIdx ), segmentOffset( view.segmentOffset ), segmentSize( view.segmentSize ),
        chunkSize( view.chunkSize ), chunksInSlice( view.chunksInSlice ){};

      __cuda_callable__
      IndexType getSize() const
      {
         return this->segmentSize;
      };

      __cuda_callable__
      IndexType getGlobalIndex( const IndexType localIdx ) const
      {
         TNL_ASSERT_LT( localIdx, segmentSize, "Local index exceeds segment bounds." );
         const IndexType chunkIdx = localIdx / chunkSize;
         const IndexType inChunkOffset = localIdx % chunkSize;
         return segmentOffset + inChunkOffset * chunksInSlice + chunkIdx;
      };

      __cuda_callable__
      const IndexType& getSegmentIndex() const
      {
         return this->segmentIdx;
      };

      __cuda_callable__
      ChunkedEllpackSegmentView& operator = ( const ChunkedEllpackSegmentView& view ) const
      {
         this->segmentIdx = view.segmentIdx;
         this->segmentOffset = view.segmentOffset;
         this->segmentSize = view.segmentSize;
         this->chunkSize = view.chunkSize;
         this->chunksInSlice = view.chunksInSlice;
         return *this;
      }

      protected:

         IndexType segmentIdx, segmentOffset, segmentSize, chunkSize, chunksInSlice;
};

      } //namespace Segments
   } //namespace Algorithms
} //namespace TNL
