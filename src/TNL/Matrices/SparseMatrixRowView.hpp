/***************************************************************************
                          SparseMatrixRowView.hpp -  description
                             -------------------
    begin                : Dec 28, 2019
    copyright            : (C) 2019 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Matrices/SparseMatrixRowView.h>
#include <TNL/Assert.h>

namespace TNL {
namespace Matrices {

template< typename SegmentView,
          typename ValuesView,
          typename ColumnsIndexesView,
          bool isBinary_ >
__cuda_callable__
SparseMatrixRowView< SegmentView, ValuesView, ColumnsIndexesView, isBinary_ >::
SparseMatrixRowView( const SegmentViewType& segmentView,
                     const ValuesViewType& values,
                     const ColumnsIndexesViewType& columnIndexes )
 : segmentView( segmentView ), values( values ), columnIndexes( columnIndexes )
{
}

template< typename SegmentView,
          typename ValuesView,
          typename ColumnsIndexesView,
          bool isBinary_ >
__cuda_callable__ auto
SparseMatrixRowView< SegmentView, ValuesView, ColumnsIndexesView, isBinary_ >::
getSize() const -> IndexType
{
   return segmentView.getSize();
}

template< typename SegmentView,
          typename ValuesView,
          typename ColumnsIndexesView,
          bool isBinary_ >
__cuda_callable__
auto
SparseMatrixRowView< SegmentView, ValuesView, ColumnsIndexesView, isBinary_ >::
getRowIndex() const -> const IndexType&
{
   return segmentView.getSegmentIndex();
}

template< typename SegmentView,
          typename ValuesView,
          typename ColumnsIndexesView,
          bool isBinary_ >
__cuda_callable__ auto
SparseMatrixRowView< SegmentView, ValuesView, ColumnsIndexesView, isBinary_ >::
getColumnIndex( const IndexType localIdx ) const -> const IndexType&
{
   TNL_ASSERT_LT( localIdx, this->getSize(), "Local index exceeds matrix row capacity." );
   return columnIndexes[ segmentView.getGlobalIndex( localIdx ) ];
}

template< typename SegmentView,
          typename ValuesView,
          typename ColumnsIndexesView,
          bool isBinary_ >
__cuda_callable__ auto
SparseMatrixRowView< SegmentView, ValuesView, ColumnsIndexesView, isBinary_ >::
getColumnIndex( const IndexType localIdx ) -> IndexType&
{
   TNL_ASSERT_LT( localIdx, this->getSize(), "Local index exceeds matrix row capacity." );
   return columnIndexes[ segmentView.getGlobalIndex( localIdx ) ];
}

template< typename SegmentView,
          typename ValuesView,
          typename ColumnsIndexesView,
          bool isBinary_ >
__cuda_callable__ auto
SparseMatrixRowView< SegmentView, ValuesView, ColumnsIndexesView, isBinary_ >::
getValue( const IndexType localIdx ) const -> const RealType&
{
   TNL_ASSERT_LT( localIdx, this->getSize(), "Local index exceeds matrix row capacity." );
   TNL_ASSERT_FALSE( isBinary(), "Cannot call this method for binary matrix row." );
   return values[ segmentView.getGlobalIndex( localIdx ) ];
}

template< typename SegmentView,
          typename ValuesView,
          typename ColumnsIndexesView,
          bool isBinary_ >
__cuda_callable__ auto
SparseMatrixRowView< SegmentView, ValuesView, ColumnsIndexesView, isBinary_ >::
getValue( const IndexType localIdx ) -> RealType&
{
   TNL_ASSERT_LT( localIdx, this->getSize(), "Local index exceeds matrix row capacity." );
   TNL_ASSERT_FALSE( isBinary(), "Cannot call this method for binary matrix row." );
   return values[ segmentView.getGlobalIndex( localIdx ) ];
}

template< typename SegmentView,
          typename ValuesView,
          typename ColumnsIndexesView,
          bool isBinary_ >
__cuda_callable__ void
SparseMatrixRowView< SegmentView, ValuesView, ColumnsIndexesView, isBinary_ >::
setValue( const IndexType localIdx,
          const RealType& value )
{
   TNL_ASSERT_LT( localIdx, this->getSize(), "Local index exceeds matrix row capacity." );
   if( ! isBinary() ) {
      const IndexType globalIdx = segmentView.getGlobalIndex( localIdx );
      values[ globalIdx ] = value;
   }
}

template< typename SegmentView,
          typename ValuesView,
          typename ColumnsIndexesView,
          bool isBinary_ >
__cuda_callable__ void
SparseMatrixRowView< SegmentView, ValuesView, ColumnsIndexesView, isBinary_ >::
setColumnIndex( const IndexType localIdx,
                const IndexType& columnIndex )
{
   TNL_ASSERT_LT( localIdx, this->getSize(), "Local index exceeds matrix row capacity." );
   const IndexType globalIdx = segmentView.getGlobalIndex( localIdx );
   this->columnIndexes[ globalIdx ] = columnIndex;
}

template< typename SegmentView,
          typename ValuesView,
          typename ColumnsIndexesView,
          bool isBinary_ >
__cuda_callable__ void
SparseMatrixRowView< SegmentView, ValuesView, ColumnsIndexesView, isBinary_ >::
setElement( const IndexType localIdx,
            const IndexType column,
            const RealType& value )
{
   TNL_ASSERT_LT( localIdx, this->getSize(), "Local index exceeds matrix row capacity." );
   const IndexType globalIdx = segmentView.getGlobalIndex( localIdx );
   columnIndexes[ globalIdx ] = column;
   if( ! isBinary() )
      values[ globalIdx ] = value;
}

template< typename SegmentView,
          typename ValuesView,
          typename ColumnsIndexesView,
          bool isBinary_ >
   template< typename _SegmentView,
             typename _ValuesView,
             typename _ColumnsIndexesView,
             bool _isBinary >
__cuda_callable__
bool
SparseMatrixRowView< SegmentView, ValuesView, ColumnsIndexesView, isBinary_ >::
operator==( const SparseMatrixRowView< _SegmentView, _ValuesView, _ColumnsIndexesView, _isBinary >& other ) const
{
   IndexType i = 0;
   while( i < getSize() && i < other.getSize() ) {
      if( getColumnIndex( i ) != other.getColumnIndex( i ) )
         return false;
      if( ! _isBinary && getValue( i ) != other.getValue( i ) )
         return false;
      ++i;
   }
   for( IndexType j = i; j < getSize(); j++ )
      // TODO: use ... != getPaddingIndex()
      if( getColumnIndex( j ) >= 0 )
         return false;
   for( IndexType j = i; j < other.getSize(); j++ )
      // TODO: use ... != getPaddingIndex()
      if( other.getColumnIndex( j ) >= 0 )
         return false;
   return true;
}

template< typename SegmentView,
          typename ValuesView,
          typename ColumnsIndexesView,
          bool isBinary_ >
std::ostream& operator<<( std::ostream& str, const SparseMatrixRowView< SegmentView, ValuesView, ColumnsIndexesView, isBinary_ >& row )
{
   using NonConstIndex = std::remove_const_t< typename SparseMatrixRowView< SegmentView, ValuesView, ColumnsIndexesView, isBinary_ >::IndexType >;
   for( NonConstIndex i = 0; i < row.getSize(); i++ )
      if( isBinary_ )
         // TODO: check getPaddingIndex(), print only the column indices of non-zeros but not the values
         str << " [ " << row.getColumnIndex( i ) << " ] = " << (row.getColumnIndex( i ) >= 0) << ", ";
      else
         str << " [ " << row.getColumnIndex( i ) << " ] = " << row.getValue( i ) << ", ";
   return str;
}

} // namespace Matrices
} // namespace TNL
