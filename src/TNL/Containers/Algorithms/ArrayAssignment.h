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

/**
 * \brief Specialization for array-array assignment
 */
template< typename Array,
          typename T >
struct ArrayAssignment
{
   static void assign( Array& a, const T& t, const typename T::ValueType* )
   {
      /*a.setSize( t.getSize() );
      ArrayOperations< typename Array::DeviceType, typename T::DeviceType >::template
         copyMemory< typename Array::ValueType, typename T::ValueType, typename T::IndexType >
         ( a.getData(), t.getData(), t.getSize() );*/
   };
   
   static void assign( Array& a, const T& t, const void* )
   {
      
   };
};

/**
 * \brief Specialization for array-value assignment
 */
/*template< typename Array,
          typename T >
struct ArrayAssignment< Array, T, void >
{
   static void assign( Array& a, const T& t )
   {
      ArrayOperations< typename Array::DeviceType >::template
         setMemory< typename Array::ValueType, typename Array::IndexType >
         ( a.getData(), ( typename Array::ValueType ) t, t.getSize() );
   };

};*/


} // namespace Algorithms
} // namespace Containers
} // namespace TNL
