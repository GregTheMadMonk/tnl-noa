/***************************************************************************
                          param-types.h  -  description
                             -------------------
    begin                : 2009/07/29
    copyright            : (C) 2009 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once 

#include <TNL/Experimental/Arithmetics/Real.h>
#include <TNL/String.h>

namespace TNL {

template< typename T >
String getType() { return T::getType(); };

// non-const specializations
template<> inline String getType< void >() { return String( "void" ); };
template<> inline String getType< bool >() { return String( "bool" ); };

template<> inline String getType< char >() { return String( "char" ); };
template<> inline String getType< short int >() { return String( "short int" ); };
template<> inline String getType< int >() { return String( "int" ); };
template<> inline String getType< long int >() { return String( "long int" ); };

template<> inline String getType< unsigned char >() { return String( "unsigned char" ); };
template<> inline String getType< unsigned short >() { return String( "unsigned short" ); };
template<> inline String getType< unsigned int >() { return String( "unsigned int" ); };
template<> inline String getType< unsigned long >() { return String( "unsigned long" ); };

template<> inline String getType< signed char >() { return String( "signed char" ); };

template<> inline String getType< float >() { return String( "float" ); };
template<> inline String getType< double >() { return String( "double" ); };
template<> inline String getType< long double >() { return String( "long double" ); };
template<> inline String getType< tnlFloat >() { return String( "tnlFloat" ); };
template<> inline String getType< tnlDouble> () { return String( "tnlDouble" ); };

// const specializations
template<> inline String getType< const void >() { return String( "const void" ); };
template<> inline String getType< const bool >() { return String( "const bool" ); };

template<> inline String getType< const char >() { return String( "const char" ); };
template<> inline String getType< const short int >() { return String( "const short int" ); };
template<> inline String getType< const int >() { return String( "const int" ); };
template<> inline String getType< const long int >() { return String( "const long int" ); };

template<> inline String getType< const unsigned char >() { return String( "const unsigned char" ); };
template<> inline String getType< const unsigned short >() { return String( "const unsigned short" ); };
template<> inline String getType< const unsigned int >() { return String( "const unsigned int" ); };
template<> inline String getType< const unsigned long >() { return String( "const unsigned long" ); };

template<> inline String getType< const signed char >() { return String( "const signed char" ); };

template<> inline String getType< const float >() { return String( "const float" ); };
template<> inline String getType< const double >() { return String( "const double" ); };
template<> inline String getType< const long double >() { return String( "const long double" ); };
template<> inline String getType< const tnlFloat >() { return String( "const tnlFloat" ); };
template<> inline String getType< const tnlDouble> () { return String( "const tnlDouble" ); };

} // namespace TNL
