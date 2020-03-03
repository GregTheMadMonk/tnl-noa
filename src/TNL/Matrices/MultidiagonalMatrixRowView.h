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
          typename DiagonalsShiftsView_ >
class MultidiagonalMatrixRowView
{
   public:

      using RealType = typename ValuesView::RealType;
      using IndexType = typename ValuesView::IndexType;
      using ValuesViewType = ValuesView;
      using IndexerType = Indexer;
      using DiagonalsShiftsView = DiagonalsShiftsView_;

      __cuda_callable__
      MultidiagonalMatrixRowView( const IndexType rowIdx,
                                  const DiagonalsShiftsView& diagonalsShifts,
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

      DiagonalsShiftsView diagonalsShifts;

      ValuesViewType values;

      Indexer indexer;
};

} // namespace Matrices
} // namespace TNL

#include <TNL/Matrices/MultidiagonalMatrixRowView.hpp>
