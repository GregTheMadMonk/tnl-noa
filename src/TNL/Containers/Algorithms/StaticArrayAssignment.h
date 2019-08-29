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
#include <TNL/StaticFor.h>

namespace TNL {
namespace Containers {
namespace Algorithms {

   namespace detail {
      ////
      // Functors used together with StaticFor for static loop unrolling in the
      // implementation of the StaticArray
      template< typename LeftValue, typename RightValue = LeftValue >
      struct assignArrayFunctor
      {
         __cuda_callable__ void operator()( int i, LeftValue* data, const RightValue* v ) const
         {
            data[ i ] = v[ i ];
         }
      };

      template< typename LeftValue, typename RightValue = LeftValue >
      struct assignValueFunctor
      {
         __cuda_callable__ void operator()( int i, LeftValue* data, const RightValue v ) const
         {
            data[ i ] = v;
         }
      };
   } //namespace detail

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
   static void assign( StaticArray& a, const T& t )
   {
      static_assert( StaticArray::getSize() == T::getSize(), "Cannot assign static arrays with different size." );
      StaticFor< 0, StaticArray::getSize() >::exec( detail::assignArrayFunctor< StaticArray, T >{}, a, t );
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
   static void assign( StaticArray& a, const T& t )
   {
      StaticFor< 0, StaticArray::getSize() >::exec( detail::assignValueFunctor< StaticArray, T >{}, a, t );

   }
};

} // namespace Algorithms
} // namespace Containers
} // namespace TNL
