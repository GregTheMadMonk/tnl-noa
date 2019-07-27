/***************************************************************************
                          ArrayAssignment.h  -  description
                             -------------------
    begin                : Apr 4, 2019
    copyright            : (C) 2019 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/TypeTraits.h>
#include <TNL/Containers/Algorithms/ArrayOperations.h>

namespace TNL {
namespace Containers {
namespace Algorithms {

template< typename Array,
          typename T,
          bool isArrayType = IsArrayType< T >::value >
struct ArrayAssignment;

/**
 * \brief Specialization for array-array assignment with containers implementing
 * getArrayData method.
 */
template< typename Array,
          typename T >
struct ArrayAssignment< Array, T, true >
{
   static void resize( Array& a, const T& t )
   {
      a.setSize( t.getSize() );
   }

   static void assign( Array& a, const T& t )
   {
      TNL_ASSERT_EQ( a.getSize(), t.getSize(), "The sizes of the arrays must be equal." );
      if( t.getSize() > 0 ) // we allow even assignment of empty arrays
         ArrayOperations< typename Array::DeviceType, typename T::DeviceType >::template
            copy< typename Array::ValueType, typename T::ValueType, typename Array::IndexType >
            ( a.getArrayData(), t.getArrayData(), t.getSize() );
   }
};

/**
 * \brief Specialization for array-value assignment for other types. We assume
 * that T is convertible to Array::ValueType.
 */
template< typename Array,
          typename T >
struct ArrayAssignment< Array, T, false >
{
   static void resize( Array& a, const T& t )
   {
   }

   static void assign( Array& a, const T& t )
   {
      TNL_ASSERT_FALSE( a.empty(), "Cannot assign value to empty array." );
      ArrayOperations< typename Array::DeviceType >::template
         set< typename Array::ValueType, typename Array::IndexType >
         ( a.getArrayData(), ( typename Array::ValueType ) t, a.getSize() );
   }
};

} // namespace Algorithms
} // namespace Containers
} // namespace TNL
