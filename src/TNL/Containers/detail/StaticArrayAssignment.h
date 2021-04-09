/***************************************************************************
                          StaticArrayAssignment.h  -  description
                             -------------------
    begin                : Aug 29, 2019
    copyright            : (C) 2019 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/TypeTraits.h>
#include <TNL/Algorithms/unrolledFor.h>

namespace TNL {
namespace Containers {
namespace detail {

template< typename StaticArray,
          typename T,
          bool isStaticArrayType = IsStaticArrayType< T >::value >
struct StaticArrayAssignment;

/**
 * \brief Specialization for array-array assignment.
 */
template< typename StaticArray,
          typename T >
struct StaticArrayAssignment< StaticArray, T, true >
{
   static constexpr void assign( StaticArray& a, const T& v )
   {
      static_assert( StaticArray::getSize() == T::getSize(),
                     "Cannot assign static arrays with different size." );
      Algorithms::unrolledFor< int, 0, StaticArray::getSize() >(
         [&] ( int i ) mutable {
            a[ i ] = v[ i ];
         }
      );
   }
};

/**
 * \brief Specialization for array-value assignment for other types. We assume
 * that T is convertible to StaticArray::ValueType.
 */
template< typename StaticArray,
          typename T >
struct StaticArrayAssignment< StaticArray, T, false >
{
   static constexpr void assign( StaticArray& a, const T& v )
   {
      Algorithms::unrolledFor< int, 0, StaticArray::getSize() >(
         [&] ( int i ) mutable {
            a[ i ] = v;
         }
      );
   }
};

} // namespace detail
} // namespace Containers
} // namespace TNL
