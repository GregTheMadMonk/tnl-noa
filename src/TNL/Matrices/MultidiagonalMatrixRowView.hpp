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

template< typename ValuesView, typename Indexer, typename DiagonalsOffsetsView >
__cuda_callable__
MultidiagonalMatrixRowView< ValuesView, Indexer, DiagonalsOffsetsView >::
MultidiagonalMatrixRowView( const IndexType rowIdx,
                            const DiagonalsOffsetsView& diagonalsOffsets,
                            const ValuesViewType& values,
                            const IndexerType& indexer )
: rowIdx( rowIdx ), diagonalsOffsets( diagonalsOffsets ), values( values ), indexer( indexer )
{
}

template< typename ValuesView, typename Indexer, typename DiagonalsOffsetsView >
__cuda_callable__
auto
MultidiagonalMatrixRowView< ValuesView, Indexer, DiagonalsOffsetsView >::
getSize() const -> IndexType
{
   return diagonalsOffsets.getSize();//indexer.getRowSize( rowIdx );
}

template< typename ValuesView, typename Indexer, typename DiagonalsOffsetsView >
__cuda_callable__
auto
MultidiagonalMatrixRowView< ValuesView, Indexer, DiagonalsOffsetsView >::
getColumnIndex( const IndexType localIdx ) const -> const IndexType
{
   TNL_ASSERT_GE( localIdx, 0, "" );
   TNL_ASSERT_LT( localIdx, indexer.getDiagonals(), "" );
   return rowIdx + diagonalsOffsets[ localIdx ];
}

template< typename ValuesView, typename Indexer, typename DiagonalsOffsetsView >
__cuda_callable__
auto
MultidiagonalMatrixRowView< ValuesView, Indexer, DiagonalsOffsetsView >::
getValue( const IndexType localIdx ) const -> const RealType&
{
   return this->values[ this->indexer.getGlobalIndex( rowIdx, localIdx ) ];
}

template< typename ValuesView, typename Indexer, typename DiagonalsOffsetsView >
__cuda_callable__
auto
MultidiagonalMatrixRowView< ValuesView, Indexer, DiagonalsOffsetsView >::
getValue( const IndexType localIdx ) -> RealType&
{
   return this->values[ this->indexer.getGlobalIndex( rowIdx, localIdx ) ];
}

template< typename ValuesView, typename Indexer, typename DiagonalsOffsetsView >
__cuda_callable__
void 
MultidiagonalMatrixRowView< ValuesView, Indexer, DiagonalsOffsetsView >::
setElement( const IndexType localIdx,
            const RealType& value )
{
   this->values[ indexer.getGlobalIndex( rowIdx, localIdx ) ] = value;
}

} // namespace Matrices
} // namespace TNL
