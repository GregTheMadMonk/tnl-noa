/***************************************************************************
                          tnlMatrixRow_impl.h  -  description
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

#ifndef TNLMATRIXROW_IMPL_H_
#define TNLMATRIXROW_IMPL_H_

template< typename Real, typename Index >
#ifdef HAVE_CUDA
      __device__ __host__
#endif
tnlMatrixRow< Real, Index >::
tnlMatrixRow()
: values( 0 ),
  columns( 0 ),
  length( 0 ),
  step( 0 )
{
}

template< typename Real, typename Index >
#ifdef HAVE_CUDA
      __device__ __host__
#endif
tnlMatrixRow< Real, Index >::
tnlMatrixRow( Index* columns,
              Real* values,
              const Index length,
              const Index step )
: values( values ),
  columns( columns ),
  length( length ),
  step( step )
{
}

template< typename Real, typename Index >
#ifdef HAVE_CUDA
      __device__ __host__
#endif
void
tnlMatrixRow< Real, Index >::
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
#ifdef HAVE_CUDA
      __device__ __host__
#endif
void
tnlMatrixRow< Real, Index >::
setElement( const Index& elementIndex,
            const Index& column,
            const Real& value )
{
   tnlAssert( this->columns, );
   tnlAssert( this->values, );
   tnlAssert( this->step > 0,);
   tnlAssert( elementIndex >= 0 && elementIndex < this->length,
              cerr << "elementIndex = " << elementIndex << " this->length = " << this->length );

   this->columns[ elementIndex * step ] = column;
   this->values[ elementIndex * step ] = value;
}

#endif /* TNLMATRIXROW_IMPL_H_ */
