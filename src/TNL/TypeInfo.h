/***************************************************************************
                          TypeInfo.h  -  description
                             -------------------
    begin                : Aug 20, 2019
    copyright            : (C) 2019 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <typeinfo>
#include <string>

#if defined( __has_include )
   #if __has_include(<cxxabi.h>)
      #define TNL_HAS_CXXABI_H
   #endif
#elif defined( __GLIBCXX__ ) || defined( __GLIBCPP__ )
   #define TNL_HAS_CXXABI_H
#endif

#if defined( TNL_HAS_CXXABI_H )
   #include <cxxabi.h>  // abi::__cxa_demangle
   #include <memory>  // std::unique_ptr
   #include <cstdlib>  // std::free
#endif

#include <TNL/TypeTraits.h>
#include <TNL/String.h>

namespace TNL {
namespace __getType_impl {

inline std::string
demangle( const char* name )
{
#if defined( TNL_HAS_CXXABI_H )
   int status = 0;
   std::size_t size = 0;
   std::unique_ptr<char[], void (*)(void*)> result(
      abi::__cxa_demangle( name, NULL, &size, &status ),
      std::free
   );
   if( result.get() )
      return result.get();
#endif
   return name;
}

} // namespace __getType_impl

/**
 * \brief Returns a human-readable string representation of given type.
 *
 * Note that since we use the \ref typeid operator internally, the top-level
 * cv-qualifiers are always ignored. See https://stackoverflow.com/a/8889143
 * for details.
 */
template< typename T >
String getType()
{
   return __getType_impl::demangle( typeid(T).name() );
}

/**
 * \brief Returns a human-readable string representation of given object's type.
 *
 * Note that since we use the \ref typeid operator internally, the top-level
 * cv-qualifiers are always ignored. See https://stackoverflow.com/a/8889143
 * for details.
 */
template< typename T >
String getType( T&& obj )
{
   return __getType_impl::demangle( typeid(obj).name() );
}

/**
 * \brief Returns a string identifying a type for the purpose of serialization.
 *
 * By default, this function returns the same string as \ref getType. However,
 * if a user-defined class has a static \e getSerializationType method, it is
 * called instead. This is useful for overriding the default \ref typeid name,
 * which may be necessary e.g. for class templates which should have the same
 * serialization type for multiple devices.
 */
template< typename T,
          std::enable_if_t< ! HasStaticGetSerializationType< T >::value, bool > = true >
String getSerializationType()
{
   return getType< T >();
}

/**
 * \brief Specialization of \ref getSerializationType for types which provide a
 *        static \e getSerializationType method to override the default behaviour.
 */
template< typename T,
          std::enable_if_t< HasStaticGetSerializationType< T >::value, bool > = true >
String getSerializationType()
{
   return T::getSerializationType();
}

} // namespace TNL
