/***************************************************************************
                          tnlDenseMatrixRow_impl.h  -  description
                             -------------------
    begin                : Dec 24, 2014
    copyright            : (C) 2014 by oberhuber
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

#ifndef TNLDENSEMATRIXROW_IMPL_H_
#define TNLDENSEMATRIXROW_IMPL_H_

template< typename Real, typename Index >
#ifdef HAVE_CUDA
      __device__ __host__
#endif
tnlDenseMatrixRow< Real, Index >::
tnlDenseMatrixRow()
: values( 0 ),
  columns( 0 ),
  step( 0 )
{
}

template< typename Real, typename Index >
#ifdef HAVE_CUDA
      __device__ __host__
#endif
tnlDenseMatrixRow< Real, Index >::
tnlDenseMatrixRow( Real* values,
                   const Index columns,
                   const Index step )
: values( values ),
  columns( columns ),
  step( step )
{
}

template< typename Real, typename Index >
#ifdef HAVE_CUDA
      __device__ __host__
#endif
void
tnlDenseMatrixRow< Real, Index >::
bind( Real* values,
      const Index columns,
      const Index step )
{
   this->values = values;
   this->columns = columns;
   this->step = step;
}

template< typename Real, typename Index >
#ifdef HAVE_CUDA
      __device__ __host__
#endif
void
tnlDenseMatrixRow< Real, Index >::
setElement( const Index& elementIndex,
            const Index& column,
            const Real& value )
{
   tnlAssert( this->values, );
   tnlAssert( this->step > 0,);
   tnlAssert( column >= 0 && column < this->columns,
              cerr << "column = " << column << " this->columns = " << this->columns );

   this->values[ column * this->step ] = value;
}


#endif /* TNLDENSEMATRIXROW_IMPL_H_ */
