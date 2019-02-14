/***************************************************************************
                          param-types.h  -  description
                             -------------------
    begin                : 2009/07/29
    copyright            : (C) 2009 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <vector>
#include <type_traits>

#include <TNL/Experimental/Arithmetics/Real.h>
#include <TNL/String.h>

namespace TNL {

namespace __getType_impl {

template< typename T,
          bool isEnum = std::is_enum< T >::value >
struct getTypeHelper
{
   static String get() { return T::getType(); }
};

template<> struct getTypeHelper< void,                 false >{ static String get() { return String( "void" ); }; };
template<> struct getTypeHelper< bool,                 false >{ static String get() { return String( "bool" ); }; };

template<> struct getTypeHelper< char,                 false >{ static String get() { return String( "char" ); }; };
template<> struct getTypeHelper< short int,            false >{ static String get() { return String( "short int" ); }; };
template<> struct getTypeHelper< int,                  false >{ static String get() { return String( "int" ); }; };
template<> struct getTypeHelper< long int,             false >{ static String get() { return String( "long int" ); }; };

template<> struct getTypeHelper< unsigned char,        false >{ static String get() { return String( "unsigned char" ); }; };
template<> struct getTypeHelper< unsigned short,       false >{ static String get() { return String( "unsigned short" ); }; };
template<> struct getTypeHelper< unsigned int,         false >{ static String get() { return String( "unsigned int" ); }; };
template<> struct getTypeHelper< unsigned long,        false >{ static String get() { return String( "unsigned long" ); }; };

template<> struct getTypeHelper< signed char,          false >{ static String get() { return String( "signed char" ); }; };

template<> struct getTypeHelper< float,                false >{ static String get() { return String( "float" ); }; };
template<> struct getTypeHelper< double,               false >{ static String get() { return String( "double" ); }; };
template<> struct getTypeHelper< long double,          false >{ static String get() { return String( "long double" ); }; };
template<> struct getTypeHelper< tnlFloat,             false >{ static String get() { return String( "tnlFloat" ); }; };
template<> struct getTypeHelper< tnlDouble,            false >{ static String get() { return String( "tnlDouble" ); }; };

// const specializations
template<> struct getTypeHelper< const void,           false >{ static String get() { return String( "const void" ); }; };
template<> struct getTypeHelper< const bool,           false >{ static String get() { return String( "const bool" ); }; };

template<> struct getTypeHelper< const char,           false >{ static String get() { return String( "const char" ); }; };
template<> struct getTypeHelper< const short int,      false >{ static String get() { return String( "const short int" ); }; };
template<> struct getTypeHelper< const int,            false >{ static String get() { return String( "const int" ); }; };
template<> struct getTypeHelper< const long int,       false >{ static String get() { return String( "const long int" ); }; };

template<> struct getTypeHelper< const unsigned char,  false >{ static String get() { return String( "const unsigned char" ); }; };
template<> struct getTypeHelper< const unsigned short, false >{ static String get() { return String( "const unsigned short" ); }; };
template<> struct getTypeHelper< const unsigned int,   false >{ static String get() { return String( "const unsigned int" ); }; };
template<> struct getTypeHelper< const unsigned long,  false >{ static String get() { return String( "const unsigned long" ); }; };

template<> struct getTypeHelper< const signed char,    false >{ static String get() { return String( "const signed char" ); }; };

template<> struct getTypeHelper< const float,          false >{ static String get() { return String( "const float" ); }; };
template<> struct getTypeHelper< const double,         false >{ static String get() { return String( "const double" ); }; };
template<> struct getTypeHelper< const long double,    false >{ static String get() { return String( "const long double" ); }; };
template<> struct getTypeHelper< const tnlFloat,       false >{ static String get() { return String( "const tnlFloat" ); }; };
template<> struct getTypeHelper< const tnlDouble,      false >{ static String get() { return String( "const tnlDouble" ); }; };

template< typename T >
struct getTypeHelper< T, true >
{
   static String get() { return getTypeHelper< typename std::underlying_type< T >::type, false >::get(); };
};

// wrappers for STL containers
template< typename T >
struct getTypeHelper< std::vector< T >, false >
{
   static String get() { return String( "std::vector< " ) + getTypeHelper< T >::get() + " >"; }
};

} // namespace __getType_impl

template< typename T >
String getType() { return __getType_impl::getTypeHelper< T >::get(); }

} // namespace TNL
