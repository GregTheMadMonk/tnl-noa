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
#include <TNL/Algorithms/Segments/SegmentViewIterator.h>

namespace TNL {
   namespace Algorithms {
      namespace Segments {

/**
 * \brief Data structure for accessing particular segment.
 *
 * \tparam Index is type for indexing elements in related segments.
 *
 * See the template specializations \ref TNL::Algorithms::Segments::SegmentView< Index, ColumnMajorOrder >
 *  and \ref TNL::Algorithms::Segments::SegmentView< Index, RowMajorOrder > for column-major
 * and row-major elements organization respectively. They have equivalent interface.
 */
template< typename Index,
          ElementsOrganization Organization >
class SegmentView;


/**
 * \brief Data structure for accessing particular segment.
 *
 * \tparam Index is type for indexing elements in related segments.
 */
template< typename Index >
class SegmentView< Index, ColumnMajorOrder >
{
   public:

      /**
       * \brief Type for indexing elements in related segments.
       */
      using IndexType = Index;

      /**
       * \brief Type of iterator for iterating over elements of the segment.
       */
      using IteratorType = SegmentViewIterator< SegmentView >;

      /**
       * \brief Conctructor with all parameters.
       *
       * \param segmentIdx is an index of segment the segment view will point to.
       * \param offset is an offset of the segment in the parent segments.
       * \param size is a size of the segment.
       * \param step is stepping between neighbouring elements in the segment.
       */
      __cuda_callable__
      SegmentView( const IndexType segmentIdx,
                   const IndexType offset,
                   const IndexType size,
                   const IndexType step )
      : segmentIdx( segmentIdx ), segmentOffset( offset ), segmentSize( size ), step( step ){};

      /**
       * \brief Copy constructor.
       *
       * \param view is the source view.
       */
      __cuda_callable__
      SegmentView( const SegmentView& view )
      : segmentIdx( view.segmentIdx ), segmentOffset( view.segmentOffset ), segmentSize( view.segmentSize ), step( view.step ){};

      /**
       * \brief Get the size of the segment, i.e. number of elements in the segment.
       *
       * \return number of elements in the segment.
       */
      __cuda_callable__
      const IndexType& getSize() const
      {
         return this->segmentSize;
      };

      /**
       * \brief Get global index of an element with rank \e localIndex in the segment.
       *
       * \param localIndex is the rank of the element in the segment.
       * \return global index of the element.
       */
      __cuda_callable__
      IndexType getGlobalIndex( const IndexType localIndex ) const
      {
         TNL_ASSERT_LT( localIndex, segmentSize, "Local index exceeds segment bounds." );
         return segmentOffset + localIndex * step;
      };

      /**
       * \brief Get index of the segment.
       *
       * \return index of the segment.
       */
      __cuda_callable__
      const IndexType& getSegmentIndex() const
      {
         return this->segmentIdx;
      };

      /**
       * \brief Returns iterator pointing at the beginning of the segment.
       *
       * \return iterator pointing at the beginning.
       */
      __cuda_callable__
      IteratorType begin() const { return IteratorType( *this, 0 ); };

      /**
       * \brief Returns iterator pointing at the end of the segment.
       *
       * \return iterator pointing at the end.
       */
      __cuda_callable__
      IteratorType end() const { return IteratorType( *this, this->getSize() ); };

      /**
       * \brief Returns constant iterator pointing at the beginning of the segment.
       *
       * \return iterator pointing at the beginning.
       */
      __cuda_callable__
      const IteratorType cbegin() const { return IteratorType( *this, 0 ); };

      /**
       * \brief Returns constant iterator pointing at the end of the segment.
       *
       * \return iterator pointing at the end.
       */
      __cuda_callable__
      const IteratorType cend() const { return IteratorType( *this, this->getSize() ); };

      protected:

         IndexType segmentIdx, segmentOffset, segmentSize, step;
};

template< typename Index >
class SegmentView< Index, RowMajorOrder >
{
   public:

      /**
       * \brief Type for indexing elements in related segments.
       */
      using IndexType = Index;

      /**
       * \brief Type of iterator for iterating over elements of the segment.
       */
      using IteratorType = SegmentViewIterator< SegmentView >;

      /**
       * \brief Conctructor with all parameters.
       *
       * \param segmentIdx is an index of segment the segment view will point to.
       * \param offset is an offset of the segment in the parent segments.
       * \param size is a size of the segment.
       * \param step is stepping between neighbouring elements in the segment.
       */
      __cuda_callable__
      SegmentView( const IndexType segmentIdx,
                   const IndexType offset,
                   const IndexType size,
                   const IndexType step = 1 ) // For compatibility with previous specialization
      : segmentIdx( segmentIdx ), segmentOffset( offset ), segmentSize( size ){};

      /**
       * \brief Copy constructor.
       *
       * \param view is the source view.
       */
      __cuda_callable__
      SegmentView( const SegmentView& view )
      : segmentIdx( view.segmentIdx ), segmentOffset( view.segmentOffset ), segmentSize( view.segmentSize ) {};

      /**
       * \brief Get the size of the segment, i.e. number of elements in the segment.
       *
       * \return number of elements in the segment.
       */
      __cuda_callable__
      const IndexType& getSize() const
      {
         return this->segmentSize;
      };

      /**
       * \brief Get global index of an element with rank \e localIndex in the segment.
       *
       * \param localIndex is the rank of the element in the segment.
       * \return global index of the element.
       */
      __cuda_callable__
      IndexType getGlobalIndex( const IndexType localIndex ) const
      {
         TNL_ASSERT_LT( localIndex, segmentSize, "Local index exceeds segment bounds." );
         return segmentOffset + localIndex;
      };

      /**
       * \brief Get index of the segment.
       *
       * \return index of the segment.
       */
      __cuda_callable__
      const IndexType& getSegmentIndex() const
      {
         return this->segmentIdx;
      };

      /**
       * \brief Returns iterator pointing at the beginning of the segment.
       *
       * \return iterator pointing at the beginning.
       */
      __cuda_callable__
      IteratorType begin() const { return IteratorType( *this, 0 ); };

      /**
       * \brief Returns iterator pointing at the end of the segment.
       *
       * \return iterator pointing at the end.
       */
      __cuda_callable__
      IteratorType end() const { return IteratorType( *this, this->getSize() ); };

      /**
       * \brief Returns constant iterator pointing at the beginning of the segment.
       *
       * \return iterator pointing at the beginning.
       */
      __cuda_callable__
      const IteratorType cbegin() const { return IteratorType( *this, 0 ); };

      /**
       * \brief Returns constant iterator pointing at the end of the segment.
       *
       * \return iterator pointing at the end.
       */
      __cuda_callable__
      const IteratorType cend() const { return IteratorType( *this, this->getSize() ); };

      protected:

         IndexType segmentIdx, segmentOffset, segmentSize;
};

      } //namespace Segments
   } //namespace Algorithms
} //namespace TNL
