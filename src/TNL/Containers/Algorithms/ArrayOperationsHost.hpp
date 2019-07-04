/***************************************************************************
                          ArrayOperationsHost_impl.h  -  description
                             -------------------
    begin                : Jul 16, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <type_traits>
#include <string.h>

#include <TNL/ParallelFor.h>
#include <TNL/Containers/Algorithms/ArrayOperations.h>
#include <TNL/Containers/Algorithms/Reduction.h>
#include <TNL/Containers/Algorithms/ReductionOperations.h>

namespace TNL {
namespace Containers {
namespace Algorithms {

template< typename Element, typename Index >
void
ArrayOperations< Devices::Host >::
allocateMemory( Element*& data,
                const Index size )
{
   data = new Element[ size ];
   // According to the standard, new either throws, or returns non-nullptr.
   // Some (old) compilers don't comply:
   // https://stackoverflow.com/questions/550451/will-new-return-null-in-any-case
   TNL_ASSERT_TRUE( data, "Operator 'new' returned a nullptr. This should never happen - there is "
                          "either a bug or the compiler does not comply to the standard." );
}

template< typename Element >
void
ArrayOperations< Devices::Host >::
freeMemory( Element* data )
{
   delete[] data;
}

template< typename Element >
void
ArrayOperations< Devices::Host >::
setMemoryElement( Element* data,
                  const Element& value )
{
   TNL_ASSERT_TRUE( data, "Attempted to set data through a nullptr." );
   *data = value;
}

template< typename Element >
Element
ArrayOperations< Devices::Host >::
getMemoryElement( const Element* data )
{
   TNL_ASSERT_TRUE( data, "Attempted to get data through a nullptr." );
   return *data;
}

template< typename Element, typename Index >
void
ArrayOperations< Devices::Host >::
setMemory( Element* data,
           const Element& value,
           const Index size )
{
   TNL_ASSERT_TRUE( data, "Attempted to set data through a nullptr." );
   auto kernel = [data, value]( Index i )
   {
      data[ i ] = value;
   };
   ParallelFor< Devices::Host >::exec( (Index) 0, size, kernel );
}

template< typename DestinationElement,
          typename SourceElement,
          typename Index >
void
ArrayOperations< Devices::Host >::
copyMemory( DestinationElement* destination,
            const SourceElement* source,
            const Index size )
{
   if( std::is_same< DestinationElement, SourceElement >::value &&
       ( std::is_fundamental< DestinationElement >::value ||
         std::is_pointer< DestinationElement >::value ) )
   {
      // GCC 8.1 complains that we bypass a non-trivial copy-constructor
      // (in C++17 we could use constexpr if to avoid compiling this branch in that case)
      #if defined(__GNUC__) && ( __GNUC__ > 8 || ( __GNUC__ == 8 && __GNUC_MINOR__ > 0 ) ) && !defined(__clang__)
         #pragma GCC diagnostic push
         #pragma GCC diagnostic ignored "-Wclass-memaccess"
      #endif
      memcpy( destination, source, size * sizeof( DestinationElement ) );
      #if defined(__GNUC__) && !defined(__clang__) && !defined(__NVCC__)
         #pragma GCC diagnostic pop
      #endif
   }
   else
   {
      auto kernel = [destination, source]( Index i )
      {
         destination[ i ] = source[ i ];
      };
      ParallelFor< Devices::Host >::exec( (Index) 0, size, kernel );
   }
}

template< typename DestinationElement,
          typename SourceElement >
void
ArrayOperations< Devices::Host >::
copySTLList( DestinationElement* destination,
             const std::list< SourceElement >& source )
{
   std::size_t i = 0;
   for( const SourceElement& e : source )
      destination[ i++ ] = e;
}


template< typename DestinationElement,
          typename SourceElement,
          typename Index >
bool
ArrayOperations< Devices::Host >::
compareMemory( const DestinationElement* destination,
               const SourceElement* source,
               const Index size )
{
   TNL_ASSERT_TRUE( destination, "Attempted to compare data through a nullptr." );
   TNL_ASSERT_TRUE( source, "Attempted to compare data through a nullptr." );
   if( std::is_same< DestinationElement, SourceElement >::value &&
       ( std::is_fundamental< DestinationElement >::value ||
         std::is_pointer< DestinationElement >::value ) )
   {
      if( memcmp( destination, source, size * sizeof( DestinationElement ) ) != 0 )
         return false;
   }
   else
      for( Index i = 0; i < size; i++ )
         if( ! ( destination[ i ] == source[ i ] ) )
            return false;
   return true;
}

template< typename Element,
          typename Index >
bool
ArrayOperations< Devices::Host >::
containsValue( const Element* data,
               const Index size,
               const Element& value )
{
   TNL_ASSERT_TRUE( data, "Attempted to check data through a nullptr." );
   TNL_ASSERT_GE( size, 0, "" );

   for( Index i = 0; i < size; i++ )
      if( data[ i ] == value )
         return true;
   return false;
}

template< typename Element,
          typename Index >
bool
ArrayOperations< Devices::Host >::
containsOnlyValue( const Element* data,
                   const Index size,
                   const Element& value )
{
   TNL_ASSERT_TRUE( data, "Attempted to check data through a nullptr." );
   TNL_ASSERT_GE( size, 0, "" );

   if( size == 0 ) return false;

   for( Index i = 0; i < size; i++ )
      if( ! ( data[ i ] == value ) )
         return false;
   return true;
}

} // namespace Algorithms
} // namespace Containers
} // namespace TNL
