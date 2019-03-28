/***************************************************************************
                          ArrayOperations.h  -  description
                             -------------------
    begin                : Jul 15, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Containers/ArrayOperations.h>

namespace TNL {
namespace Containers {
namespace Algorithms {


template< typename Array,
   typename T,
   bool isArray = TNL::IsArray< T > >
struct ArraysAsignment
{};

/**
 * \brief Specialization for array-array assignment
 */
template< typename Array,
   typename T >
struct ArraysAsignment< Array, T, true >
{
   void assign( Array& a, T& t )
   {
      a.setSize( t.getSize() );
      ArrayOperations< typename Array::DeviceType, typename T::DeviceType >::
         copyMemory< typename Array::ValueType, typename T::ValueType, typename T::IndexType >
         ( a.getData(), t.getData(), t.getSize() );
   };
};

/**
 * \brief Specialization for array-value assignment
 */
template< typename Array,
   typename T >
struct ArraysAsignment< Array, T, false >
{
   void assign( Array& a, T& t )
   {
      ArrayOperations< typename Array::DeviceType >::
         setMemory< typename Array::ValueType, typename Array::IndexType >
         ( a.getData(), ( typename Array::ValueType ) t, t.getSize() );
   };

};


} // namespace Algorithms
} // namespace Containers
} // namespace TNL
