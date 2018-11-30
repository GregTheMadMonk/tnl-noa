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
#include <TNL/ParallelFor.h>

// Following includes are here to enable usage of std::vector and std::cout. To avoid having to include Device type (HOW would this be done anyway)
#include <iostream>
#include <vector>

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
__cuda_callable__
Index
SparseRow< Real, Index >::
getLength() const
{
   return length;
}

template< typename Real, typename Index >
__cuda_callable__
Index
SparseRow< Real, Index >::
getNonZeroElementsCount() const
{
    using NonConstIndex = typename std::remove_const< Index >::type;
    // using DeviceType = typename TNL::Matrices::Matrix::DeviceType;
    
    NonConstIndex elementCount ( 0 );
    
//    auto computeNonzeros = [this, &elementCount] /*__cuda_callable__*/ ( NonConstIndex i ) mutable
//    {
//        if( getElementValue( i ) != ( Real ) 0 )
//            elementCount++;
//    };
   
//    ParallelFor< Device >::exec( ( NonConstIndex ) 0, length, computeNonzeros );
//    The ParallelFor::exec() function needs a < DeviceType >, how to get this into SparseRow?
    /*
     
     */
    
    // std::vector< Real > vls = values; // Size of values should be something like: (sizeof(this->values)/sizeof(*this->values)) from https://stackoverflow.com/questions/4108313/how-do-i-find-the-length-of-an-array
   
    for( NonConstIndex i = 0; i < length; i++ ) // this->values doesn't have anything similar to getSize().
        if( this->values[ i * step ] != 0.0 ) // This returns the same amount of elements in a row as does getRowLength(). WHY?
            elementCount++;
    
    return elementCount;
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
