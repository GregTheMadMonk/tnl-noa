/***************************************************************************
                          DenseRow_impl.h  -  description
                             -------------------
    begin                : Dec 24, 2014
    copyright            : (C) 2014 by oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {
namespace Matrices {   

template< typename Real, typename Index >
__cuda_callable__
DenseRow< Real, Index >::
DenseRow()
: values( 0 ),
  columns( 0 ),
  step( 0 )
{
}

template< typename Real, typename Index >
__cuda_callable__
DenseRow< Real, Index >::
DenseRow( Real* values,
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
DenseRow< Real, Index >::
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
DenseRow< Real, Index >::
setElement( const Index& elementIndex,
            const Index& column,
            const Real& value )
{
   Assert( this->values, );
   Assert( this->step > 0,);
   Assert( column >= 0 && column < this->columns,
              std::cerr << "column = " << column << " this->columns = " << this->columns );

   this->values[ column * this->step ] = value;
}

} // namespace Matrices
} // namespace TNL
