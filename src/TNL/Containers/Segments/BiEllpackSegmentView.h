/***************************************************************************
                          BiEllpackSegmentView.h -  description
                             -------------------
    begin                : Apr 7, 2020
    copyright            : (C) 2020 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {
   namespace Containers {
      namespace Segments {

template< typename Index,
          bool RowMajorOrder = false >
class BiEllpackSegmentView;

template< typename Index >
class BiEllpackSegmentView< Index, false >
{
   public:

      using IndexType = Index;

      __cuda_callable__
      BiEllpackSegmentView( const IndexType offset,
                                 const IndexType size,
                                 const IndexType chunkSize,      // this is only for compatibility with the following specialization
                                 const IndexType chunksInSlice ) // this one as well - both can be replaced when we could use constexprif in C++17
      : segmentOffset( offset ), segmentSize( size ){};

      __cuda_callable__
      BiEllpackSegmentView( const BiEllpackSegmentView& view )
      : segmentOffset( view.segmentOffset ), segmentSize( view.segmentSize ){};

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

      protected:
         
         IndexType segmentOffset, segmentSize;
};

template< typename Index >
class BiEllpackSegmentView< Index, true >
{
   public:

      using IndexType = Index;

      __cuda_callable__
      BiEllpackSegmentView( const IndexType offset,
                                 const IndexType size,
                                 const IndexType chunkSize,
                                 const IndexType chunksInSlice )
      : segmentOffset( offset ), segmentSize( size ),
        chunkSize( chunkSize ), chunksInSlice( chunksInSlice ){};

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

      protected:
         
         IndexType segmentOffset, segmentSize, chunkSize, chunksInSlice;
};

      } //namespace Segments
   } //namespace Containers
} //namespace TNL
