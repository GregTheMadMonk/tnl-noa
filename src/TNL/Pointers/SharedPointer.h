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

//#define TNL_DEBUG_SHARED_POINTERS

#ifdef TNL_DEBUG_SHARED_POINTERS
   #include <typeinfo>
   #include <cxxabi.h>
   #include <iostream>
   #include <string>
   #include <memory>
   #include <cstdlib>

   inline
   std::string demangle(const char* mangled)
   {
      int status;
      std::unique_ptr<char[], void (*)(void*)> result(
         abi::__cxa_demangle(mangled, 0, 0, &status), std::free);
      return result.get() ? std::string(result.get()) : "error occurred";
   }
#endif


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
      ss << "(SharedPointer< " << Object::getType() << ", " << Device::getType()
         << " > object at " << &value << ")";
      return ss.str();
   }
};

} // namespace Assert
#endif

} // namespace TNL

#include <TNL/Pointers/SharedPointerHost.h>
#include <TNL/Pointers/SharedPointerCuda.h>
