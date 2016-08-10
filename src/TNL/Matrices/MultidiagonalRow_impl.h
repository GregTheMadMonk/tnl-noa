/***************************************************************************
                          MultidiagonalRow_impl.h  -  description
                             -------------------
    begin                : Jan 2, 2015
    copyright            : (C) 2015 by oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {
namespace Matrices {   

template< typename Real, typename Index >
__cuda_callable__
MultidiagonalRow< Real, Index >::
MultidiagonalRow()
: values( 0 ),
  diagonals( 0 ),
  row( 0 ),
  columns( 0 ),
  maxRowLength( 0 ),
  step( 0 )
{
}

template< typename Real, typename Index >
__cuda_callable__
MultidiagonalRow< Real, Index >::
MultidiagonalRow( Real* values,
                           Index* diagonals,
                           const Index maxRowLength,
                           const Index row,
                           const Index columns,
                           const Index step )
: values( values ),
  diagonals( diagonals ),
  row( row ),
  columns( columns ),
  maxRowLength( maxRowLength ),
  step( step )
{
}

template< typename Real, typename Index >
__cuda_callable__
void
MultidiagonalRow< Real, Index >::
bind( Real* values,
      Index* diagonals,
      const Index maxRowLength,
      const Index row,
      const Index columns,
      const Index step )
{
   this->values = values;
   this->diagonals = diagonals;
   this->row = row;
   this->columns = columns;
   this->maxRowLength = maxRowLength;
   this->step = step;
}

template< typename Real, typename Index >
__cuda_callable__
void
MultidiagonalRow< Real, Index >::
setElement( const Index& elementIndex,
            const Index& column,
            const Real& value )
{
   Assert( this->values, );
   Assert( this->step > 0,);
   Assert( column >= 0 && column < this->columns,
              std::cerr << "column = " << columns << " this->columns = " << this->columns );
   Assert( elementIndex >= 0 && elementIndex < this->maxRowLength,
              std::cerr << "elementIndex = " << elementIndex << " this->maxRowLength =  " << this->maxRowLength );

   Index aux = elementIndex;
   while( row + this->diagonals[ aux ] < column ) aux++;
   Assert( row + this->diagonals[ aux ] == column,
              std::cerr << "row = " << row
                   << " aux = " << aux
                   << " this->diagonals[ aux ] = " << this->diagonals[ aux]
                   << " row + this->diagonals[ aux ] " << row + this->diagonals[ aux ]
                   << " column = " << column );

   //printf( "Setting element %d column %d value %f \n", aux * this->step, column, value );
   this->values[ aux * this->step ] = value;
}

} // namespace Matrices
} // namespace TNL
