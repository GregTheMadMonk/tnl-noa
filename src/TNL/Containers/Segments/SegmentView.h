/***************************************************************************
                          SegmentView.h -  description
                             -------------------
    begin                : Dec 28, 2019
    copyright            : (C) 2019 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {
   namespace Containers {
      namespace Segments {

template< typename Index,
          bool RowMajorOrder = false >
class SegmentView;

template< typename Index >
class SegmentView< Index, false >
{
   public:

      using IndexType = Index;

      __cuda_callable__
      SegmentView( const IndexType offset,
                   const IndexType size,
                   const IndexType step )
      : segmentOffset( offset ), segmentSize( size ), step( step )
      {
         printf( "--- size = %d \n", size );
      };

      __cuda_callable__
      SegmentView( const SegmentView& view )
      : segmentOffset( view.segmentOffset ), segmentSize( view.segmentSize ), step( view.step )
      {
      };
      __cuda_callable__
      IndexType getSize() const
      {
         return this->segmentSize;
      };

      __cuda_callable__
      IndexType getGlobalIndex( const IndexType localIndex ) const
      {
         TNL_ASSERT_LT( localIndex, segmentSize, "Local index exceeds segment bounds." );
         return segmentOffset + localIndex * step;
      };

      protected:
         
         IndexType segmentOffset, segmentSize, step;
};

template< typename Index >
class SegmentView< Index, true >
{
   public:

      using IndexType = Index;

      __cuda_callable__
      SegmentView( const IndexType offset,
                   const IndexType size,
                   const IndexType step = 1 ) // For compatibility with previous specialization
      : segmentOffset( offset ), segmentSize( size ){};

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

      } //namespace Segments
   } //namespace Containers
} //namespace TNL
