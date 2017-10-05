/***************************************************************************
                          SparseRow_impl.h  -  description
                             -------------------
    begin                : Dec 19, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Matrices/SparseRow.h>

namespace TNL {
namespace Matrices {   

template< typename Real, typename Index >
__cuda_callable__
SparseRow< Real, Index >::
SparseRow()
: values( 0 ),
  columns( 0 ),
  length( 0 ),
  step( 0 )
{
}

template< typename Real, typename Index >
__cuda_callable__
SparseRow< Real, Index >::
SparseRow( Index* columns,
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
__cuda_callable__
void
SparseRow< Real, Index >::
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
SparseRow< Real, Index >::
setElement( const Index& elementIndex,
            const Index& column,
            const Real& value )
{
   TNL_ASSERT( this->columns, );
   TNL_ASSERT( this->values, );
   TNL_ASSERT( this->step > 0,);
   //printf( "elementIndex = %d length = %d \n", elementIndex, this->length );
   TNL_ASSERT( elementIndex >= 0 && elementIndex < this->length,
              std::cerr << "elementIndex = " << elementIndex << " this->length = " << this->length );

   this->columns[ elementIndex * step ] = column;
   this->values[ elementIndex * step ] = value;
}

template< typename Real, typename Index >
__cuda_callable__
const Index&
SparseRow< Real, Index >::
getElementColumn( const Index& elementIndex ) const
{
   TNL_ASSERT( elementIndex >= 0 && elementIndex < this->length,
              std::cerr << "elementIndex = " << elementIndex << " this->length = " << this->length );

   return this->columns[ elementIndex * step ];
}

template< typename Real, typename Index >
__cuda_callable__
const Real&
SparseRow< Real, Index >::
getElementValue( const Index& elementIndex ) const
{
   TNL_ASSERT( elementIndex >= 0 && elementIndex < this->length,
              std::cerr << "elementIndex = " << elementIndex << " this->length = " << this->length );

   return this->values[ elementIndex * step ];
}

template< typename Real, typename Index >
void
SparseRow< Real, Index >::
print( std::ostream& str ) const
{
   using NonConstIndex = typename std::remove_const< Index >::type;
   NonConstIndex pos( 0 );
   for( NonConstIndex i = 0; i < length; i++ )
   {
      str << " [ " << columns[ pos ] << " ] = " << values[ pos ] << ", ";
      pos += step;
   }
}

} // namespace Matrices
} // namespace TNL
