/***************************************************************************
                          SharedPointer.h  -  description
                             -------------------
    begin                : May 6, 2016
    copyright            : (C) 2016 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Tomas Oberhuber, Jakub Klinkovsky

#pragma once

#include <cstring>
#include <type_traits>
#include <TNL/Assert.h>
#include <TNL/TypeInfo.h>

//#define TNL_DEBUG_SHARED_POINTERS

namespace TNL {
namespace Pointers {

template< typename Object,
          typename Device = typename Object::DeviceType >
class SharedPointer
{
   static_assert( ! std::is_same< Device, void >::value, "The device cannot be void. You need to specify the device explicitly in your code." );
};

} // namespace Pointers

#ifndef NDEBUG
namespace Assert {

template< typename Object, typename Device >
struct Formatter< Pointers::SharedPointer< Object, Device > >
{
   static std::string
   printToString( const Pointers::SharedPointer< Object, Device >& value )
   {
      ::std::stringstream ss;
      ss << "(" + getType< Pointers::SharedPointer< Object, Device > >()
         << " > object at " << &value << ")";
      return ss.str();
   }
};

} // namespace Assert
#endif

} // namespace TNL

#include <TNL/Pointers/SharedPointerHost.h>
#include <TNL/Pointers/SharedPointerCuda.h>
