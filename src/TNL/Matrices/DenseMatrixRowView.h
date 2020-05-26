/***************************************************************************
                          DenseMatrixRowView.h -  description
                             -------------------
    begin                : Jan 3, 2020
    copyright            : (C) 2020 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {
   namespace Matrices {

/**
 * \brief RowView is a simple structure for accessing rows of dense matrix.
 * 
 * \tparam SegmentView is a segment view of segments representing the matrix format.
 * \tparam ValuesView is a vector view storing the matrix elements values.
 * 
 * See \ref DenseMatrix and \ref DenseMatrixView.
 * 
 * \par Example
 * \include Matrices/DenseMatrixExample_getRow.cpp
 * \par Output
 * \include DenseMatrixExample_getRow.out
 * 
 * \par Example
 * \include Matrices/DenseMatrixViewExample_getRow.cpp
 * \par Output
 * \include DenseMatrixViewExample_getRow.out
 */
template< typename SegmentView,
          typename ValuesView >
class DenseMatrixRowView
{
   public:

      /**
       * \brief The type of matrix elements.
       */
      using RealType = typename ValuesView::RealType;

      /**
       * \brief The type used for matrix elements indexing.
       */
      using IndexType = typename SegmentView::IndexType;

      /**
       * \brief Type representing matrix row format.
       */
      using SegmentViewType = SegmentView;

      /**
       * \brief Type of container view used for storing matrix elements values.
       */
      using ValuesViewType = ValuesView;

      /**
       * \brief Constructor with \e segmentView and \e values
       * 
       * \param segmentView instance of SegmentViewType representing matrix row.
       * \param values is a container view for storing the matrix elements values.
       */
      __cuda_callable__
      DenseMatrixRowView( const SegmentViewType& segmentView,
                          const ValuesViewType& values );

      /**
       * \brief Returns size of the matrix row, i.e. number of matrix elements in this row.
       * 
       * \return Size of the matrix row.
       */
      __cuda_callable__
      IndexType getSize() const;

      /**
       * \brief Returns constants reference to an element with given column index.
       * 
       * \param column is column index of the matrix element.
       * 
       * \return constant reference to the matrix element.
       */
      __cuda_callable__
      const RealType& getElement( const IndexType column ) const;

      /**
       * \brief Returns non-constants reference to an element with given column index.
       * 
       * \param column is a column index of the matrix element.
       * 
       * \return non-constant reference to the matrix element.
       */
      __cuda_callable__
      RealType& getElement( const IndexType column );

      /**
       * \brief Sets value of matrix element with given column index
       * .
       * \param column is a column index of the matrix element.
       * \param value is a value the matrix element will be set to.
       */
      __cuda_callable__
      void setElement( const IndexType column,
                       const RealType& value );
   protected:

      SegmentViewType segmentView;

      ValuesViewType values;
};
   } // namespace Matrices
} // namespace TNL

#include <TNL/Matrices/DenseMatrixRowView.hpp>
