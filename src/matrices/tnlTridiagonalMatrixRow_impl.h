/***************************************************************************
                          tnlTridiagonalMatrixRow_impl.h  -  description
                             -------------------
    begin                : Dec 31, 2014
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

#ifndef TNLTRIDIAGONALMATRIXROW_IMPL_H_
#define TNLTRIDIAGONALMATRIXROW_IMPL_H_

template< typename Real, typename Index >
#ifdef HAVE_CUDA
      __device__ __host__
#endif
tnlTridiagonalMatrixRow< Real, Index >::
tnlTridiagonalMatrixRow()
: values( 0 ),
  row( 0 ),
  columns( 0 ),
  step( 0 )
{
}

template< typename Real, typename Index >
#ifdef HAVE_CUDA
      __device__ __host__
#endif
tnlTridiagonalMatrixRow< Real, Index >::
tnlTridiagonalMatrixRow( Real* values,
                         const Index row,
                         const Index columns,
                         const Index step )
: values( values ),
  row( row ),
  columns( columns ),
  step( step )
{
}

template< typename Real, typename Index >
#ifdef HAVE_CUDA
      __device__ __host__
#endif
void
tnlTridiagonalMatrixRow< Real, Index >::
bind( Real* values,
      const Index row,
      const Index columns,
      const Index step )
{
   this->values = values;
   this->row = row;
   this->columns = columns;
   this->step = step;
}

template< typename Real, typename Index >
#ifdef HAVE_CUDA
      __device__ __host__
#endif
void
tnlTridiagonalMatrixRow< Real, Index >::
setElement( const Index& elementIndex,
            const Index& column,
            const Real& value )
{
   tnlAssert( this->values, );
   tnlAssert( this->step > 0,);
   tnlAssert( column >= 0 && column < this->columns,
              cerr << "column = " << columns << " this->columns = " << this->columns );
   tnlAssert( abs( column - row ) <= 1,
              cerr << "column = " << column << " row =  " << row );

   /****
    * this->values stores an adress of the diagonal element
    */
   this->values[ ( column - row ) * this->step ] = value;
}



#endif /* TNLTRIDIAGONALMATRIXROW_IMPL_H_ */
