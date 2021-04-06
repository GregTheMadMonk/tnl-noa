/***************************************************************************
                          SegmentElement.h -  description
                             -------------------
    begin                : Apr 5, 2021
    copyright            : (C) 2021 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <ostream>

#include <TNL/Cuda/CudaCallable.h>

namespace TNL {
   namespace Algorithms {
      namespace Segments {


template< typename Index >
class SegmentElement
{
   public:

      using IndexType = Index;

      __cuda_callable__
      SegmentElement( const IndexType& segmentIdx,
                      const IndexType& localIdx,
                      const IndexType globalIdx )
      : segmentIdx( segmentIdx ), localIdx( localIdx ), globalIdx( globalIdx ) {};

      __cuda_callable__
      const IndexType& segmentIndex() const { return segmentIdx; };

      __cuda_callable__
      const IndexType& localIndex() const { return localIdx; };

      __cuda_callable__
      const IndexType& globalIndex() const { return globalIdx; };

   protected:

      const IndexType& segmentIdx;

      const IndexType& localIdx;

      const IndexType globalIdx;


};

      } // namespace Segments
   } // namespace Algorithms
} // namespace TNL
