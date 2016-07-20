/***************************************************************************
                          tnlSparseMatrixRow_impl.h  -  description
                             -------------------
    begin                : Dec 19, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {

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
tnlSparseMatrixRow( Index* columns,
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
   Assert( this->columns, );
   Assert( this->values, );
   Assert( this->step > 0,);
   //printf( "elementIndex = %d length = %d \n", elementIndex, this->length );
   Assert( elementIndex >= 0 && elementIndex < this->length,
              std::cerr << "elementIndex = " << elementIndex << " this->length = " << this->length );

   this->columns[ elementIndex * step ] = column;
   this->values[ elementIndex * step ] = value;
}

template< typename Real, typename Index >
void
tnlSparseMatrixRow< Real, Index >::
print( std::ostream& str ) const
{
   Index pos( 0 );
   for( Index i = 0; i < length; i++ )
   {
      str << " [ " << columns[ pos ] << " ] = " << values[ pos ] << ", ";
      pos += step;
   }
}

} // namespace TNL
