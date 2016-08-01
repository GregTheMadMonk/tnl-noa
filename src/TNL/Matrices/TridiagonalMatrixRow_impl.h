/***************************************************************************
                          TridiagonalMatrixRow_impl.h  -  description
                             -------------------
    begin                : Dec 31, 2014
    copyright            : (C) 2014 by oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {
namespace Matrices {   

template< typename Real, typename Index >
__cuda_callable__
TridiagonalMatrixRow< Real, Index >::
TridiagonalMatrixRow()
: values( 0 ),
  row( 0 ),
  columns( 0 ),
  step( 0 )
{
}

template< typename Real, typename Index >
__cuda_callable__
TridiagonalMatrixRow< Real, Index >::
TridiagonalMatrixRow( Real* values,
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
TridiagonalMatrixRow< Real, Index >::
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
TridiagonalMatrixRow< Real, Index >::
setElement( const Index& elementIndex,
            const Index& column,
            const Real& value )
{
   Assert( this->values, );
   Assert( this->step > 0,);
   Assert( column >= 0 && column < this->columns,
              std::cerr << "column = " << columns << " this->columns = " << this->columns );
   Assert( abs( column - row ) <= 1,
              std::cerr << "column = " << column << " row =  " << row );

   /****
    * this->values stores an adress of the diagonal element
    */
   this->values[ ( column - row ) * this->step ] = value;
}

} // namespace Matrices
} // namespace TNL
