/***************************************************************************
                          ArrayAssignment.h  -  description
                             -------------------
    begin                : Apr 4, 2019
    copyright            : (C) 2019 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <type_traits>
#include <utility>
#include <TNL/Containers/Algorithms/ArrayOperations.h>

namespace TNL {
namespace Containers {
namespace Algorithms {

namespace detail {
/**
 * SFINAE for checking if T has getArrayData method
 */
template< typename T >
class HasGetArrayData
{
private:
    typedef char YesType[1];
    typedef char NoType[2];

    template< typename C > static YesType& test( decltype(std::declval< C >().getArrayData()) );
    template< typename C > static NoType& test(...);

public:
    static constexpr bool value = ( sizeof( test< T >(0) ) == sizeof( YesType ) );
};
} // namespace detail

template< typename Array,
          typename T,
          bool hasGetArrayData = detail::HasGetArrayData< T >::value >
struct ArrayAssignment{};

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
            copyMemory< typename Array::ValueType, typename T::ValueType, typename Array::IndexType >
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
         setMemory< typename Array::ValueType, typename Array::IndexType >
         ( a.getArrayData(), ( typename Array::ValueType ) t, a.getSize() );
   }
};

} // namespace Algorithms
} // namespace Containers
} // namespace TNL
