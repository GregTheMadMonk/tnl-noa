/***************************************************************************
                          TridiagonalMatrixRowView.h  -  description
                             -------------------
    begin                : Dec 31, 2014
    copyright            : (C) 2014 by oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {
namespace Matrices {   

template< typename ValuesView,
          typename Indexer >
class TridiagonalMatrixRowView
{
   public:

      using RealType = typename ValuesView::RealType;
      using IndexType = typename ValuesView::IndexType;
      using ValuesViewType = ValuesView;
      using IndexerType = Indexer;

      __cuda_callable__
      TridiagonalMatrixRowView( const IndexType rowIdx,
                                const ValuesViewType& values,
                                const IndexerType& indexer );

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

      ValuesViewType values;

      Indexer indexer;
};

} // namespace Matrices
} // namespace TNL

#include <TNL/Matrices/TridiagonalMatrixRowView.hpp>
