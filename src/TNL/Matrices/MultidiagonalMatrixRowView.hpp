/***************************************************************************
                          MultidiagonalMatrixRowView.hpp  -  description
                             -------------------
    begin                : Jan 11, 2020
    copyright            : (C) 2020 by oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {
namespace Matrices {   

template< typename ValuesView, typename Indexer >
__cuda_callable__
MultidiagonalMatrixRowView< ValuesView, Indexer >::
MultidiagonalMatrixRowView( const IndexType rowIdx,
                          const ValuesViewType& values,
                          const IndexerType& indexer )
: rowIdx( rowIdx ), values( values ), indexer( indexer )
{
}

template< typename ValuesView, typename Indexer >
__cuda_callable__
auto
MultidiagonalMatrixRowView< ValuesView, Indexer >::
getSize() const -> IndexType
{
   return indexer.getRowSize();
}

template< typename ValuesView, typename Indexer >
__cuda_callable__
auto
MultidiagonalMatrixRowView< ValuesView, Indexer >::
getColumnIndex( const IndexType localIdx ) const -> const IndexType
{
   TNL_ASSERT_GE( localIdx, 0, "" );
   TNL_ASSERT_LT( localIdx, 3, "" );
   return rowIdx + localIdx - 1;
}

template< typename ValuesView, typename Indexer >
__cuda_callable__
auto
MultidiagonalMatrixRowView< ValuesView, Indexer >::
getValue( const IndexType localIdx ) const -> const RealType&
{
   return this->values[ this->indexer.getGlobalIndex( rowIdx, localIdx ) ];
}

template< typename ValuesView, typename Indexer >
__cuda_callable__
auto
MultidiagonalMatrixRowView< ValuesView, Indexer >::
getValue( const IndexType localIdx ) -> RealType&
{
   return this->values[ this->indexer.getGlobalIndex( rowIdx, localIdx ) ];
}

template< typename ValuesView, typename Indexer >
__cuda_callable__
void 
MultidiagonalMatrixRowView< ValuesView, Indexer >::
setElement( const IndexType localIdx,
            const RealType& value )
{
   this->values[ indexer.getGlobalIndex( rowIdx, localIdx ) ] = value;
}

} // namespace Matrices
} // namespace TNL
