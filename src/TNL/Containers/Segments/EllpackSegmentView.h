/***************************************************************************
                          EllpackSegmentView.h -  description
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

template< typename Index >
class EllpackSegmentView
{
   public:

      using IndexType = Index;

      __cuda_callable__
      EllpackSegmentView( const IndexType offset,
                          const IndexType size,
                          const IndexType step )
      : segmentOffset( offset ), segmentSize( size ), step( step ){};

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
      } //namespace Segments
   } //namespace Containers
} //namespace TNL
