 /***************************************************************************
                          SegmentViewIterator.h -  description
                             -------------------
    begin                : Apr 5, 2021
    copyright            : (C) 2021 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <ostream>

#include <TNL/Cuda/CudaCallable.h>
#include <TNL/Algorithms/Segments/SegmentElement.h>

namespace TNL {
   namespace Algorithms {
      namespace Segments {

template< typename SegmentView >
class SegmentViewIterator
{
   public:

      /**
       * \brief Type of SegmentView
       */
      using SegmentViewType = SegmentView;

      /**
       * \brief The type used for matrix elements indexing.
       */
      using IndexType = typename SegmentViewType::IndexType;

      /**
       * \brief The type of related matrix element.
       */
      using SegmentElementType = SegmentElement< IndexType >;

      __cuda_callable__
      SegmentViewIterator( const SegmentViewType& segmentView,
                           const IndexType& localIdx );

      /**
       * \brief Comparison of two matrix Segment iterators.
       *
       * \param other is another matrix Segment iterator.
       * \return \e true if both iterators points at the same point of the same matrix, \e false otherwise.
       */
      __cuda_callable__
      bool operator==( const SegmentViewIterator& other ) const;

      /**
       * \brief Comparison of two matrix Segment iterators.
       *
       * \param other is another matrix Segment iterator.
       * \return \e false if both iterators points at the same point of the same matrix, \e true otherwise.
       */
      __cuda_callable__
      bool operator!=( const SegmentViewIterator& other ) const;

      __cuda_callable__
      SegmentViewIterator& operator++();

      __cuda_callable__
      SegmentViewIterator& operator--();

      __cuda_callable__
      const SegmentElementType operator*() const;

   protected:

      const SegmentViewType& segmentView;

      IndexType localIdx = 0;
};

      } // namespace Segments
   } // namespace Algorithms
} // namespace TNL

#include <TNL/Algorithms/Segments/SegmentViewIterator.hpp>
