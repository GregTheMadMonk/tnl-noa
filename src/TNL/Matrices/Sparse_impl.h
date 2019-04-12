/***************************************************************************
                          Sparse_impl.h  -  description
                             -------------------
    begin                : Dec 21, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include "Sparse.h"
#include <TNL/Exceptions/NotImplementedError.h>

namespace TNL {
namespace Matrices {

template< typename Real,
          typename Device,
          typename Index >
Sparse< Real, Device, Index >::Sparse()
: maxRowLength( 0 )
{
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Real2,
             typename Device2,
             typename Index2 >
void Sparse< Real, Device, Index >::setLike( const Sparse< Real2, Device2, Index2 >& matrix )
{
   Matrix< Real, Device, Index >::setLike( matrix );
   this->allocateMatrixElements( matrix.getNumberOfMatrixElements() );
}

template< typename Real,
          typename Device,
          typename Index >
Index Sparse< Real, Device, Index >::getNumberOfMatrixElements() const
{
   return this->values.getSize();
}

template< typename Real,
          typename Device,
          typename Index >
Index Sparse< Real, Device, Index >::getNumberOfNonzeroMatrixElements() const
{
   IndexType nonzeroElements( 0 );
   for( IndexType i = 0; i < this->values.getSize(); i++ )
      if( this->columnIndexes.getElement( i ) != this-> columns &&
          this->values.getElement( i ) != 0.0 )
         nonzeroElements++;
   return nonzeroElements;
}

template< typename Real,
          typename Device,
          typename Index >
Index
Sparse< Real, Device, Index >::
getMaxRowLength() const
{
   return this->maxRowLength;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
Index Sparse< Real, Device, Index >::getPaddingIndex() const
{
   return this->getColumns();
}

template< typename Real,
          typename Device,
          typename Index >
void Sparse< Real, Device, Index >::reset()
{
   Matrix< Real, Device, Index >::reset();
   this->values.reset();
   this->columnIndexes.reset();
}

template< typename Real,
          typename Device,
          typename Index >
void Sparse< Real, Device, Index >::save( File& file ) const
{
   Matrix< Real, Device, Index >::save( file );
   this->values.save( file );
   this->columnIndexes.save( file );
}

template< typename Real,
          typename Device,
          typename Index >
void Sparse< Real, Device, Index >::load( File& file )
{
   Matrix< Real, Device, Index >::load( file );
   this->values.load( file );
   this->columnIndexes.load( file );
}

template< typename Real,
          typename Device,
          typename Index >
void Sparse< Real, Device, Index >::allocateMatrixElements( const IndexType& numberOfMatrixElements )
{
   this->values.setSize( numberOfMatrixElements );
   this->columnIndexes.setSize( numberOfMatrixElements );

   /****
    * Setting a column index to this->columns means that the
    * index is undefined.
    */
   if( numberOfMatrixElements > 0 )
      this->columnIndexes.setValue( this->columns );
}

template< typename Real,
          typename Device,
          typename Index >
void Sparse< Real, Device, Index >::printStructure( std::ostream& str ) const
{
   throw Exceptions::NotImplementedError("Sparse::printStructure is not implemented yet.");
}

} // namespace Matrices
} // namespace TNL
