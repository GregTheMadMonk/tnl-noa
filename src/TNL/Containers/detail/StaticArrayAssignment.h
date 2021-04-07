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
#include <TNL/Algorithms/UnrolledFor.h>

namespace TNL {
namespace Containers {
namespace detail {

struct AssignArrayFunctor
{
   template< typename LeftValue, typename RightValue >
   __cuda_callable__
   void operator()( int i, LeftValue* data, const RightValue* v ) const
   {
      data[ i ] = v[ i ];
   }
};

struct AssignValueFunctor
{
   template< typename LeftValue, typename RightValue >
   __cuda_callable__
   void operator()( int i, LeftValue* data, const RightValue& v ) const
   {
      data[ i ] = v;
   }
};

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
   __cuda_callable__
   static void assign( StaticArray& a, const T& v )
   {
      static_assert( StaticArray::getSize() == T::getSize(), "Cannot assign static arrays with different size." );
      Algorithms::UnrolledFor< 0, StaticArray::getSize() >::exec( AssignArrayFunctor{}, a.getData(), v.getData() );
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
   __cuda_callable__
   static void assign( StaticArray& a, const T& v )
   {
      Algorithms::UnrolledFor< 0, StaticArray::getSize() >::exec( AssignValueFunctor{}, a.getData(), v );
   }
};

} // namespace detail
} // namespace Containers
} // namespace TNL
