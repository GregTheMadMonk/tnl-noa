/***************************************************************************
                          tnlDenseMatrixRow_impl.h  -  description
                             -------------------
    begin                : Dec 24, 2014
    copyright            : (C) 2014 by oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {

template< typename Real, typename Index >
__cuda_callable__
tnlDenseMatrixRow< Real, Index >::
tnlDenseMatrixRow()
: values( 0 ),
  columns( 0 ),
  step( 0 )
{
}

template< typename Real, typename Index >
__cuda_callable__
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
__cuda_callable__
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
__cuda_callable__
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

} // namespace TNL
