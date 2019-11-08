/***************************************************************************
                          MemoryOperationsSequential.hpp  -  description
                             -------------------
    begin                : Apr 8, 2019
    copyright            : (C) 2019 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Algorithms/MemoryOperations.h>

namespace TNL {
namespace Algorithms {

template< typename Element >
__cuda_callable__
void
MemoryOperations< Devices::Sequential >::
setElement( Element* data,
            const Element& value )
{
   *data = value;
}

template< typename Element >
__cuda_callable__
Element
MemoryOperations< Devices::Sequential >::
getElement( const Element* data )
{
   return *data;
}

template< typename Element, typename Index >
__cuda_callable__
void
MemoryOperations< Devices::Sequential >::
set( Element* data,
     const Element& value,
     const Index size )
{
   for( Index i = 0; i < size; i ++ )
      data[ i ] = value;
}

template< typename DestinationElement,
          typename SourceElement,
          typename Index >
__cuda_callable__
void
MemoryOperations< Devices::Sequential >::
copy( DestinationElement* destination,
      const SourceElement* source,
      const Index size )
{
   for( Index i = 0; i < size; i ++ )
      destination[ i ] = source[ i ];
}

template< typename DestinationElement,
          typename Index,
          typename SourceIterator >
void
MemoryOperations< Devices::Sequential >::
copyFromIterator( DestinationElement* destination,
                  Index destinationSize,
                  SourceIterator first,
                  SourceIterator last )
{
   Index i = 0;
   while( i < destinationSize && first != last )
      destination[ i++ ] = *first++;
   if( first != last )
      throw std::length_error( "Source iterator is larger than the destination array." );
}

template< typename Element1,
          typename Element2,
          typename Index >
__cuda_callable__
bool
MemoryOperations< Devices::Sequential >::
compare( const Element1* destination,
         const Element2* source,
         const Index size )
{
   for( Index i = 0; i < size; i++ )
      if( ! ( destination[ i ] == source[ i ] ) )
         return false;
   return true;
}

template< typename Element,
          typename Index >
__cuda_callable__
bool
MemoryOperations< Devices::Sequential >::
containsValue( const Element* data,
               const Index size,
               const Element& value )
{
   if( size == 0 ) return false;
   TNL_ASSERT_TRUE( data, "Attempted to check data through a nullptr." );
   TNL_ASSERT_GE( size, 0, "" );

   for( Index i = 0; i < size; i++ )
      if( data[ i ] == value )
         return true;
   return false;
}

template< typename Element,
          typename Index >
__cuda_callable__
bool
MemoryOperations< Devices::Sequential >::
containsOnlyValue( const Element* data,
                   const Index size,
                   const Element& value )
{
   if( size == 0 ) return false;
   TNL_ASSERT_TRUE( data, "Attempted to check data through a nullptr." );
   TNL_ASSERT_GE( size, 0, "" );

   for( Index i = 0; i < size; i++ )
      if( ! ( data[ i ] == value ) )
         return false;
   return true;
}

} // namespace Algorithms
} // namespace TNL
