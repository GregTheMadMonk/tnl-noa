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
getElement( const IndexType column ) const -> const RealType&
{
   TNL_ASSERT_LT( column, this->getSize(), "Column index exceeds matrix row size." );
   return values[ segmentView.getGlobalIndex( column ) ];
}

template< typename SegmentView,
          typename ValuesView >
__cuda_callable__ auto
DenseMatrixRowView< SegmentView, ValuesView >::
getElement( const IndexType column ) -> RealType&
{
   TNL_ASSERT_LT( column, this->getSize(), "Column index exceeds matrix row size." );
   return values[ segmentView.getGlobalIndex( column ) ];
}

template< typename SegmentView,
          typename ValuesView >
__cuda_callable__ void
DenseMatrixRowView< SegmentView, ValuesView >::
setElement( const IndexType column,
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

   } // namespace Matrices
} // namespace TNL
