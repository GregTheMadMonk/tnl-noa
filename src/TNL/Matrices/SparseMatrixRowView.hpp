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

namespace TNL {
   namespace Matrices {

template< typename Real,
          typename SegmentView >
__cuda_callable__
SparseMatrixRowView< Real, SegmentView >::
SparseMatrixRowView( const SegmentView& segmentView,
                     const ValuesView& values,
                     const ColumnIndexesView& columnIndexes )
 : segmentView( segmentView ), values( values ), columnIndexes( columnIndexes )
{
}

template< typename Real,
          typename SegmentView >
__cuda_callable__ auto
SparseMatrixRowView< Real, SegmentView >::
getSize() const -> IndexType
{
   return segmentView.getSize();
}

template< typename Real,
          typename SegmentView >
__cuda_callable__ auto
SparseMatrixRowView< Real, SegmentView >::
getColumnIndex( const IndexType localIdx ) const -> const IndexType&
{
   TNL_ASSERT_LT( localIdx, this->getSize(), "Local index exceeds matrix row capacity." );
   return columnIndexes[ segmentView.getGlobalIndex( localIdx ) ];
}

template< typename Real,
          typename SegmentView >
__cuda_callable__ auto
SparseMatrixRowView< Real, SegmentView >::
getColumnIndex( const IndexType localIdx ) -> IndexType&
{
   TNL_ASSERT_LT( localIdx, this->getSize(), "Local index exceeds matrix row capacity." );
   return columnIndexes[ segmentView.getGlobalIndex( localIdx ) ];
}

template< typename Real,
          typename SegmentView >
__cuda_callable__ auto
SparseMatrixRowView< Real, SegmentView >::
getValue( const IndexType localIdx ) const -> const RealType&
{
   TNL_ASSERT_LT( localIdx, this->getSize(), "Local index exceeds matrix row capacity." );
   return values[ segmentView.getGlobalIndex( localIdx ) ];
}

template< typename Real,
          typename SegmentView >
__cuda_callable__ auto
SparseMatrixRowView< Real, SegmentView >::
getValue( const IndexType localIdx ) -> RealType&
{
   TNL_ASSERT_LT( localIdx, this->getSize(), "Local index exceeds matrix row capacity." );
   return values[ segmentView.getGlobalIndex( localIdx ) ];
}

template< typename Real,
          typename SegmentView >
__cuda_callable__ void 
SparseMatrixRowView< Real, SegmentView >::
setElement( const IndexType localIdx,
            const IndexType column,
            const RealType& value )
{
   TNL_ASSERT_LT( localIdx, this->getSize(), "Local index exceeds matrix row capacity." );
   const IndexType globalIdx = segmentView.getGlobalIndex( localIdx );
   columnIndexes[ globalIdx ] = column;
   values[ globalIdx ] = value;
}


   } // namespace Matrices
} // namespace TNL
