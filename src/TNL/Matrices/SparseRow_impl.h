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
    
    // If this is static, it will trigger a illegal memory address
    // How to get it into the lambda function?
    NonConstIndex elementCount ( 0 );
    
    
//    using CudaType = typename TNL::Devices::Cuda;
//    using HostType = typename TNL::Devices::Host;
//    
//    
//    // elementCount = 0; // Only if it is static. Make sure it is reset. Without this seemingly useless step, it returned incorrect values.
//    
//    // PROBLEM: Lambda function with __cuda_callable__ CANNOT pass values by reference!!
//    // INCORRECT ASSUMPTION!! PROBLEM: Lambda function which takes in anything via capture list, cannot return anything. (Maybe dont capture anything? pass this->values by parameter and return count?)
//        // WRONG: https://stackoverflow.com/questions/38835154/lambda-function-capture-a-variable-vs-return-value?fbclid=IwAR0ybDD83LRWxkJsrcoSmGW2mbsMfhywmdZQkleqyjU-NOIwqkz8woihfXs
//    auto computeNonZeros = [=] __cuda_callable__ ( NonConstIndex i /*, NonConstIndex *elementCount*/ ) mutable
//    {
//        //std::cout << "this->values[ i * step ] = " << this->values[ i * step ] << " != 0.0/n";
//        if( this->values[ i * step ] != 0.0 )
//            elementCount++;//*elementCount++;
//        
//        //std::cout << "End of lambda elementCount = " << elementCount << "/n";
//        //return elementCount;
//    };
//    
//    
//    // Decide which ParallelFor will be executed, either Host or Cuda.
//    if( deviceType == TNL::String( "Devices::Host" ) )
//    {
//        ParallelFor< HostType >::exec( ( NonConstIndex ) 0, length, computeNonZeros /*, &elementCount*/ );
//    }
//    
//    else if( deviceType == TNL::String( "Cuda" ) )
//    {
//        ParallelFor< CudaType >::exec( ( NonConstIndex ) 0, length, computeNonZeros /*, &elementCount*/ );
//    }
   
    
//    // THE FOLLOWING doesn't work on GPU
    for( NonConstIndex i = 0; i < length; i++ )
    {
        std::cout << "this->values[ i * step ] = " << this->values[ i * step ] << " != 0.0" << std::endl;
        if( this->values[ i * step ] != 0.0 ) // Returns the same amount of elements in a row as does getRowLength() in ChunkedEllpack. WHY?
            elementCount++;
    }
    
     std::cout << "Element Count = " << elementCount << "\n";
    
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
