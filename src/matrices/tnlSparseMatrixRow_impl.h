/***************************************************************************
                          tnlSparseMatrixRow_impl.h  -  description
                             -------------------
    begin                : Dec 19, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
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

#ifndef TNLSPARSEMATRIXROW_IMPL_H_
#define TNLSPARSEMATRIXROW_IMPL_H_

template< typename Real, typename Index >
__cuda_callable__
tnlSparseMatrixRow< Real, Index >::
tnlSparseMatrixRow()
: values( 0 ),
  columns( 0 ),
  length( 0 ),
  step( 0 )
{
}

template< typename Real, typename Index >
__cuda_callable__
tnlSparseMatrixRow< Real, Index >::
tnlSparseMatrixRow( const Index* columns,
					const Real* values,
                    const Index length,
                    const Index step )
: values( values ),
  columns( columns ),
  length( length ),
  step( step )
{
}

template< typename Real, typename Index >
__cuda_callable__
void
tnlSparseMatrixRow< Real, Index >::
bind( Index* columns,
      Real* values,
      const Index length,
      const Index step )
{
   this->columns = columns;
   this-> values = values;
   this->length = length;
   this->step = step;
}

template< typename Real, typename Index >
__cuda_callable__
void
tnlSparseMatrixRow< Real, Index >::
setElement( const Index& elementIndex,
            const Index& column,
            const Real& value )
{
   tnlAssert( this->columns, );
   tnlAssert( this->values, );
   tnlAssert( this->step > 0,);
   //printf( "elementIndex = %d length = %d \n", elementIndex, this->length );
   tnlAssert( elementIndex >= 0 && elementIndex < this->length,
              cerr << "elementIndex = " << elementIndex << " this->length = " << this->length );

   tnlAssert(false, ); //Disabled
   //this->columns[ elementIndex * step ] = column;
   //this->values[ elementIndex * step ] = value;
}

#endif /* TNLSPARSEMATRIXROW_IMPL_H_ */
