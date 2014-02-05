/***************************************************************************
                          tnlMatrix_impl.h  -  description
                             -------------------
    begin                : Dec 18, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef TNLMATRIX_IMPL_H_
#define TNLMATRIX_IMPL_H_

#include <matrices/tnlMatrix.h>

template< typename Real,
          typename Device,
          typename Index >
tnlMatrix< Real, Device, Index >::tnlMatrix()
: rows( 0 ),
  columns( 0 )
{
}

template< typename Real,
          typename Device,
          typename Index >
 bool tnlMatrix< Real, Device, Index >::setDimensions( const IndexType rows,
                                                       const IndexType columns )
{
   tnlAssert( rows > 0 && columns > 0,
            cerr << " rows = " << rows << " columns = " << columns );
   this->rows = rows;
   this->columns = columns;
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
void tnlMatrix< Real, Device, Index >::getRowLengths( tnlVector< IndexType, DeviceType, IndexType >& rowLengths ) const
{
   rowLengths.setSize( this->getRows() );
   for( IndexType row = 0; row < this->getRows(); row++ )
      rowLengths.setElement( row, this->getRowLength( row ) );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Real2,
             typename Device2,
             typename Index2 >
bool tnlMatrix< Real, Device, Index >::setLike( const tnlMatrix< Real2, Device2, Index2 >& matrix )
{
   return setDimensions( matrix.getRows(), matrix.getColumns() );
}

template< typename Real,
          typename Device,
          typename Index >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
Index tnlMatrix< Real, Device, Index >::getRows() const
{
   return this->rows;
}

template< typename Real,
          typename Device,
          typename Index >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
Index tnlMatrix< Real, Device, Index >::getColumns() const
{
   return this->columns;
}

template< typename Real,
          typename Device,
          typename Index >
void tnlMatrix< Real, Device, Index >::reset()
{
   this->rows = 0;
   this->columns = 0;
}

template< typename Real,
          typename Device,
          typename Index >
tnlMatrix< Real, Device, Index >& tnlMatrix< Real, Device, Index >::operator = ( const tnlMatrix< RealType, DeviceType, IndexType >& m )
{
   this->setLike( m );

   tnlVector< IndexType, DeviceType, IndexType > rowLengths;
   m.getRowLengths( rowLengths );
   this->setRowLengths( rowLengths );

   tnlVector< RealType, DeviceType, IndexType > rowValues;
   tnlVector< IndexType, DeviceType, IndexType > rowColumns;
   const IndexType maxRowLength = rowLengths.max();
   rowValues.setSize( maxRowLength );
   rowColumns.setSize( maxRowLength );
   for( IndexType row = 0; row < this->getRows(); row++ )
   {
      m.getRowFast( row,
                    rowColumns.getData(),
                    rowValues.getData() );
      this->setRowFast( row,
                        rowColumns.getData(),
                        rowValues.getData(),
                        rowLengths.getElement( row ) );
   }
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlMatrix< Real, Device, Index >::save( tnlFile& file ) const
{
#ifdef HAVE_NOT_CXX11
   if( ! tnlObject::save( file ) ||
       ! file.write< IndexType, tnlHost, Index >( &this->rows, 1 ) ||
       ! file.write< IndexType, tnlHost, Index >( &this->columns, 1 ) ||
       ! this->values.save( file ) )
      return false;
#else   
   if( ! tnlObject::save( file ) ||
       ! file.write( &this->rows ) ||
       ! file.write( &this->columns ) ||
       ! this->values.save>( file ) )
      return false;
#endif      
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlMatrix< Real, Device, Index >::load( tnlFile& file )
{
#ifdef HAVE_NOT_CXX11
   if( ! tnlObject::load( file ) ||
       ! file.read< IndexType, tnlHost, Index >( &this->rows, 1 ) ||
       ! file.read< IndexType, tnlHost, Index >( &this->columns, 1 ) ||
       ! this->values.load( file ) )
      return false;
#else   
   if( ! tnlObject::load( file ) ||
       ! file.read( &this->rows ) ||
       ! file.read( &this->columns ) ||
       ! this->values.load( file ) )
      return false;
#endif      
   return true;
}

#endif /* TNLMATRIX_IMPL_H_ */
