/***************************************************************************
                          DenseMatrixRowView.hpp -  description
                             -------------------
    begin                : Jan 3, 2020
    copyright            : (C) 2020 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Matrices/DenseMatrixRowView.h>

namespace TNL {
   namespace Matrices {

template< typename SegmentView,
          typename ValuesView >
__cuda_callable__
DenseMatrixRowView< SegmentView, ValuesView >::
DenseMatrixRowView( const SegmentViewType& segmentView,
                     const ValuesViewType& values )
 : segmentView( segmentView ), values( values )
{
}

template< typename SegmentView,
          typename ValuesView >
__cuda_callable__ auto
DenseMatrixRowView< SegmentView, ValuesView >::
getSize() const -> IndexType
{
   return segmentView.getSize();
}

template< typename SegmentView,
          typename ValuesView >
__cuda_callable__ auto
DenseMatrixRowView< SegmentView, ValuesView >::
getRowIndex() const -> const IndexType&
{
   return segmentView.getSegmentIndex();
}

template< typename SegmentView,
          typename ValuesView >
__cuda_callable__ auto
DenseMatrixRowView< SegmentView, ValuesView >::
getValue( const IndexType column ) const -> const RealType&
{
   TNL_ASSERT_LT( column, this->getSize(), "Column index exceeds matrix row size." );
   return values[ segmentView.getGlobalIndex( column ) ];
}

template< typename SegmentView,
          typename ValuesView >
__cuda_callable__ auto
DenseMatrixRowView< SegmentView, ValuesView >::
getValue( const IndexType column ) -> RealType&
{
   TNL_ASSERT_LT( column, this->getSize(), "Column index exceeds matrix row size." );
   return values[ segmentView.getGlobalIndex( column ) ];
}

template< typename SegmentView,
          typename ValuesView >
__cuda_callable__ auto
DenseMatrixRowView< SegmentView, ValuesView >::
getColumnIndex( const IndexType localIdx ) const -> IndexType
{
   TNL_ASSERT_LT( localIdx, this->getSize(), "Column index exceeds matrix row size." );
   return localIdx;
}


template< typename SegmentView,
          typename ValuesView >
__cuda_callable__ void
DenseMatrixRowView< SegmentView, ValuesView >::
setValue( const IndexType column,
          const RealType& value )
{
   TNL_ASSERT_LT( column, this->getSize(), "Column index exceeds matrix row size." );
   const IndexType globalIdx = segmentView.getGlobalIndex( column );
   values[ globalIdx ] = value;
}

template< typename SegmentView,
          typename ValuesView >
__cuda_callable__ void
DenseMatrixRowView< SegmentView, ValuesView >::
setElement( const IndexType localIdx,
            const IndexType column,
            const RealType& value )
{
   TNL_ASSERT_LT( column, this->getSize(), "Column index exceeds matrix row size." );
   const IndexType globalIdx = segmentView.getGlobalIndex( column );
   values[ globalIdx ] = value;
}

template< typename SegmentView,
          typename ValuesView >
__cuda_callable__ auto
DenseMatrixRowView< SegmentView, ValuesView >::
begin() -> IteratorType
{
   return IteratorType( *this, 0 );
}

template< typename SegmentView,
          typename ValuesView >
__cuda_callable__ auto
DenseMatrixRowView< SegmentView, ValuesView >::
end() -> IteratorType
{
   return IteratorType( *this, this->getSize() );
}

template< typename SegmentView,
          typename ValuesView >
__cuda_callable__ auto
DenseMatrixRowView< SegmentView, ValuesView >::
cbegin() const -> const IteratorType
{
   return IteratorType( *this, 0 );
}

template< typename SegmentView,
          typename ValuesView >
__cuda_callable__ auto
DenseMatrixRowView< SegmentView, ValuesView >::
cend() const -> const IteratorType
{
   return IteratorType( *this, this->getSize() );
}

   } // namespace Matrices
} // namespace TNL
