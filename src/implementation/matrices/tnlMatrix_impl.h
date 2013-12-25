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
Index tnlMatrix< Real, Device, Index >::getRows() const
{
   return this->rows;
}

template< typename Real,
          typename Device,
          typename Index >
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
bool tnlMatrix< Real, Device, Index >::save( tnlFile& file ) const
{
   if( ! tnlObject::save( file ) ||
       ! file.write( &this->rows ) ||
       ! file.write( &this->columns ) )
      return false;
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlMatrix< Real, Device, Index >::load( tnlFile& file )
{
   if( ! tnlObject::load( file ) ||
       ! file.read( &this->rows ) ||
       ! file.read( &this->columns ) )
      return false;
   return true;
}

#endif /* TNLMATRIX_IMPL_H_ */
