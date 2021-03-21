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

template< typename RowView >
__cuda_callable__
SparseMatrixRowViewIterator< RowView >::
SparseMatrixRowViewIterator( RowViewType& rowView,
                             const IndexType& localIdx )
: rowView( rowView ), localIdx( localIdx )
{
}

template< typename RowView >
__cuda_callable__ bool
SparseMatrixRowViewIterator< RowView >::
operator==( const SparseMatrixRowViewIterator& other ) const
{
   if( &this->rowView == &other.rowView &&
       localIdx == other.localIdx )
      return true;
   return false;
}

template< typename RowView >
__cuda_callable__ bool
SparseMatrixRowViewIterator< RowView >::
operator!=( const SparseMatrixRowViewIterator& other ) const
{
   return ! ( other == *this );
}

template< typename RowView >
__cuda_callable__
SparseMatrixRowViewIterator< RowView >&
SparseMatrixRowViewIterator< RowView >::
operator++()
{
   if( localIdx < rowView.getSize() )
      localIdx ++;
   return *this;
}

template< typename RowView >
__cuda_callable__
SparseMatrixRowViewIterator< RowView >&
SparseMatrixRowViewIterator< RowView >::
operator--()
{
   if( localIdx > 0 )
      localIdx --;
   return *this;
}

template< typename RowView >
__cuda_callable__ auto
SparseMatrixRowViewIterator< RowView >::
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
SparseMatrixRowViewIterator< RowView >::
operator*() const -> const MatrixElementType
{
   return MatrixElementType(
      this->rowView.getValue( this->localIdx ),
      this->rowView.getRowIndex( this->localIdx ),
      this->rowView.getColumnIndex( this->localIdx ),
      this->localIdx );
}


   } // namespace Matrices
} // namespace TNL
