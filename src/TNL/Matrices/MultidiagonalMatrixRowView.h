/***************************************************************************
                          MultidiagonalMatrixRowView.h  -  description
                             -------------------
    begin                : Jan 11, 2020
    copyright            : (C) 2020 by oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {
namespace Matrices {

/**
 * \brief RowView is a simple structure for accessing rows of multidiagonal matrix.
 *
 * \tparam ValuesView is a vector view storing the matrix elements values.
 * \tparam Indexer is type of object responsible for indexing and organization of
 *    matrix elements.
 * \tparam DiagonalsOffsetsView_ is a container view holding offsets of
 *    diagonals of multidiagonal matrix.
 *
 * See \ref MultidiagonalMatrix and \ref MultidiagonalMatrixView.
 *
 * \par Example
 * \include Matrices/MultidiagonalMatrix/MultidiagonalMatrixExample_getRow.cpp
 * \par Output
 * \include MultidiagonalatrixExample_getRow.out
 *
 * \par Example
 * \include Matrices/MultidiagonalMatrix/MultidiagonalMatrixViewExample_getRow.cpp
 * \par Output
 * \include MultidiagonalMatrixViewExample_getRow.out
 */
template< typename ValuesView,
          typename Indexer,
          typename DiagonalsOffsetsView_ >
class MultidiagonalMatrixRowView
{
   public:

      /**
       * \brief The type of matrix elements.
       */
      using RealType = typename ValuesView::RealType;

      /**
       * \brief The type used for matrix elements indexing.
       */
      using IndexType = typename ValuesView::IndexType;

      /**
       * \brief Type of container view used for storing the matrix elements values.
       */
      using ValuesViewType = ValuesView;

      /**
       * \brief Type of object responsible for indexing and organization of
       * matrix elements.
       */
      using IndexerType = Indexer;

      /**
       * \brief Type of a container view holding offsets of
       * diagonals of multidiagonal matrix.
       */
      using DiagonalsOffsetsView = DiagonalsOffsetsView_;

      /**
       * \brief Type of constant container view used for storing the matrix elements values.
       */
      using ConstValuesViewType = typename ValuesViewType::ConstViewType;

      /**
       * \brief Type of constant container view used for storing the column indexes of the matrix elements.
       */
      using ConstDiagonalsOffsetsViewType = typename DiagonalsOffsetsView::ConstViewType;

      /**
       * \brief Type of constant indexer view.
       */
      using ConstIndexerViewType = typename Indexer::ConstType;

      /**
       * \brief Type of constant sparse matrix row view.
       */
      using ConstViewType = MultidiagonalMatrixRowView< ConstValuesViewType, ConstIndexerViewType, ConstDiagonalsOffsetsViewType >;

      /**
       * \brief Constructor with all necessary data.
       *
       * \param rowIdx is index of the matrix row this RowView refer to.
       * \param diagonalsOffsets is a vector view holding offsets of matrix diagonals,
       * \param values is a vector view holding values of matrix elements.
       * \param indexer is object responsible for indexing and organization of matrix elements
       */
      __cuda_callable__
      MultidiagonalMatrixRowView( const IndexType rowIdx,
                                  const DiagonalsOffsetsView& diagonalsOffsets,
                                  const ValuesViewType& values,
                                  const IndexerType& indexer );

      /**
       * \brief Returns number of diagonals of the multidiagonal matrix.
       *
       * \return number of diagonals of the multidiagonal matrix.
       */
      __cuda_callable__
      IndexType getSize() const;

      /**
       * \brief Returns the matrix row index.
       *
       * \return matrix row index.
       */
      __cuda_callable__
      const IndexType& getRowIndex() const;

      /**
       * \brief Computes column index of matrix element on given subdiagonal.
       *
       * \param localIdx is an index of the subdiagonal.
       *
       * \return column index of matrix element on given subdiagonal.
       */
      __cuda_callable__
      const IndexType getColumnIndex( const IndexType localIdx ) const;

      /**
       * \brief Returns value of matrix element on given subdiagonal.
       *
       * \param localIdx is an index of the subdiagonal.
       *
       * \return constant reference to matrix element value.
       */
      __cuda_callable__
      const RealType& getValue( const IndexType localIdx ) const;

      /**
       * \brief Returns value of matrix element on given subdiagonal.
       *
       * \param localIdx is an index of the subdiagonal.
       *
       * \return non-constant reference to matrix element value.
       */
      __cuda_callable__
      RealType& getValue( const IndexType localIdx );

      /**
       * \brief Changes value of matrix element on given subdiagonal.
       *
       * \param localIdx is an index of the matrix subdiagonal.
       * \param value is the new value of the matrix element.
       */
      __cuda_callable__
      void setElement( const IndexType localIdx,
                       const RealType& value );
   protected:

      IndexType rowIdx;

      DiagonalsOffsetsView diagonalsOffsets;

      ValuesViewType values;

      Indexer indexer;
};

} // namespace Matrices
} // namespace TNL

#include <TNL/Matrices/MultidiagonalMatrixRowView.hpp>
