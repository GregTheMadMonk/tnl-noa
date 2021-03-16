/***************************************************************************
                          SegmentView.h -  description
                             -------------------
    begin                : Dec 28, 2019
    copyright            : (C) 2019 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Algorithms/Segments/ElementsOrganization.h>

namespace TNL {
   namespace Algorithms {
      namespace Segments {

template< typename Index,
          ElementsOrganization Organization >
class SegmentView;

template< typename Index >
class SegmentView< Index, ColumnMajorOrder >
{
   public:

      using IndexType = Index;

      __cuda_callable__
      SegmentView( const IndexType segmentIdx,
                   const IndexType offset,
                   const IndexType size,
                   const IndexType step )
      : segmentIdx( segmentIdx ), segmentOffset( offset ), segmentSize( size ), step( step ){};

      __cuda_callable__
      SegmentView( const SegmentView& view )
      : segmentIdx( view.segmentIdx ), segmentOffset( view.segmentOffset ), segmentSize( view.segmentSize ), step( view.step ){};

      __cuda_callable__
      const IndexType& getSize() const
      {
         return this->segmentSize;
      };

      __cuda_callable__
      IndexType getGlobalIndex( const IndexType localIndex ) const
      {
         TNL_ASSERT_LT( localIndex, segmentSize, "Local index exceeds segment bounds." );
         return segmentOffset + localIndex * step;
      };

      __cuda_callable__
      const IndexType& getSegmentIndex() const
      {
         return this->segmentIdx;
      };

      protected:

         IndexType segmentIdx, segmentOffset, segmentSize, step;
};

template< typename Index >
class SegmentView< Index, RowMajorOrder >
{
   public:

      using IndexType = Index;

      __cuda_callable__
      SegmentView( const IndexType segmentIdx,
                   const IndexType offset,
                   const IndexType size,
                   const IndexType step = 1 ) // For compatibility with previous specialization
      : segmentIdx( segmentIdx ), segmentOffset( offset ), segmentSize( size ){};

      __cuda_callable__
      const IndexType& getSize() const
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

      } //namespace Segments
   } //namespace Algorithms
} //namespace TNL
