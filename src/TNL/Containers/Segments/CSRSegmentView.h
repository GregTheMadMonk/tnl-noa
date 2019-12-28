/***************************************************************************
                          CSRSegmentView.h -  description
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
class CSRSegmentView
{
   public:

      using IndexType = Index;

      __cuda_callable__
      CSRSegmentView( const IndexType offset, const IndexType size )
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