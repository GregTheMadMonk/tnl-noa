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

template< typename ValuesView,
          typename Indexer,
          typename DiagonalsOffsetsView_ >
class MultidiagonalMatrixRowView
{
   public:

      using RealType = typename ValuesView::RealType;
      using IndexType = typename ValuesView::IndexType;
      using ValuesViewType = ValuesView;
      using IndexerType = Indexer;
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

      __cuda_callable__
      MultidiagonalMatrixRowView( const IndexType rowIdx,
                                  const DiagonalsOffsetsView& diagonalsOffsets,
                                  const ValuesViewType& values,
                                  const IndexerType& indexer);

      __cuda_callable__
      IndexType getSize() const;

      __cuda_callable__
      const IndexType getColumnIndex( const IndexType localIdx ) const;

      __cuda_callable__
      const RealType& getValue( const IndexType localIdx ) const;

      __cuda_callable__
      RealType& getValue( const IndexType localIdx );

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
