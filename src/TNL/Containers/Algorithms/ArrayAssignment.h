/***************************************************************************
                          ArrayOperations.h  -  description
                             -------------------
    begin                : Jul 15, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Containers/Algorithms/ArrayOperations.h>

namespace TNL {
namespace Containers {
namespace Algorithms {

namespace Details {
/**
 * SFINAE for checking if T has getArrayData method
 */
template< typename T >
class HasGetArrayData
{
private:
    typedef char YesType[1];
    typedef char NoType[2];

    template< typename C > static YesType& test( decltype(&C::getArrayData) ) ;
    template< typename C > static NoType& test(...);

public:
    enum { value = sizeof( test< T >(0) ) == sizeof( YesType ) };
};
} // namespace Details

template< typename Array,
          typename T,
          bool hasGetArrayData = Details::HasGetArrayData< T >::value >
struct ArrayAssignment{};

/**
 * \brief Specialization for array-array assignment with containers implementing
 * getArrayData method.
 */
template< typename Array,
          typename T >
struct ArrayAssignment< Array, T, true >
{
   static void assign( Array& a, const T& t )
   {
      ArrayOperations< typename Array::DeviceType, typename T::DeviceType >::template
         copyMemory< typename Array::ValueType, typename T::ValueType, typename Array::IndexType >
         ( a.getArrayData(), t.getArrayData(), t.getSize() );
   };
};

/**
 * \brief Specialization for array-value assignment for other types. We assume
 * thet T is convertible to Array::ValueType.
 */
template< typename Array,
          typename T >
struct ArrayAssignment< Array, T, false >
{
   static void assign( Array& a, const T& t )
   {
      ArrayOperations< typename Array::DeviceType >::template
         setMemory< typename Array::ValueType, typename Array::IndexType >
         ( a.getArrayData(), ( typename Array::ValueType ) t, a.getSize() );
   };

};



} // namespace Algorithms
} // namespace Containers
} // namespace TNL
