/***************************************************************************
                          LambdaMatrixRowView.hpp -  description
                             -------------------
    begin                : Mar 21, 2021
    copyright            : (C) 2021 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Matrices/LambdaMatrixRowView.h>
#include <TNL/Assert.h>

namespace TNL {
namespace Matrices {

template< typename RowView >
__cuda_callable__
LambdaMatrixRowViewIterator< RowView >::
LambdaMatrixRowViewIterator( const RowViewType& rowView,
                             const IndexType& localIdx )
: rowView( rowView ), localIdx( localIdx )
{
}

template< typename RowView >
__cuda_callable__ bool
LambdaMatrixRowViewIterator< RowView >::
operator==( const LambdaMatrixRowViewIterator& other ) const
{
   if( &this->rowView == &other.rowView &&
       localIdx == other.localIdx )
      return true;
   return false;
}

template< typename RowView >
__cuda_callable__ bool
LambdaMatrixRowViewIterator< RowView >::
operator!=( const LambdaMatrixRowViewIterator& other ) const
{
   return ! ( other == *this );
}

template< typename RowView >
__cuda_callable__
LambdaMatrixRowViewIterator< RowView >&
LambdaMatrixRowViewIterator< RowView >::
operator++()
{
   if( localIdx < rowView.getSize() )
      localIdx ++;
   return *this;
}

template< typename RowView >
__cuda_callable__
LambdaMatrixRowViewIterator< RowView >&
LambdaMatrixRowViewIterator< RowView >::
operator--()
{
   if( localIdx > 0 )
      localIdx --;
   return *this;
}

template< typename RowView >
__cuda_callable__ auto
LambdaMatrixRowViewIterator< RowView >::
operator*() -> MatrixElementType
{
   return MatrixElementType(
      this->rowView.getValue( this->localIdx ),
      this->rowView.getRowIndex(),
      this->rowView.getColumnIndex( this->localIdx ),
      this->localIdx );
}

template< typename RowView >
__cuda_callable__ auto
LambdaMatrixRowViewIterator< RowView >::
operator*() const -> const MatrixElementType
{
   return MatrixElementType(
      this->rowView.getValue( this->localIdx ),
      this->rowView.getRowIndex(),
      this->rowView.getColumnIndex( this->localIdx ),
      this->localIdx );
}


   } // namespace Matrices
} // namespace TNL
