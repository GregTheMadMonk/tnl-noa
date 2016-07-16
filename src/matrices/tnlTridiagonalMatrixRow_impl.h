/***************************************************************************
                          tnlTridiagonalMatrixRow_impl.h  -  description
                             -------------------
    begin                : Dec 31, 2014
    copyright            : (C) 2014 by oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNLTRIDIAGONALMATRIXROW_IMPL_H_
#define TNLTRIDIAGONALMATRIXROW_IMPL_H_

template< typename Real, typename Index >
__cuda_callable__
tnlTridiagonalMatrixRow< Real, Index >::
tnlTridiagonalMatrixRow()
: values( 0 ),
  row( 0 ),
  columns( 0 ),
  step( 0 )
{
}

template< typename Real, typename Index >
__cuda_callable__
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
__cuda_callable__
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
__cuda_callable__
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
