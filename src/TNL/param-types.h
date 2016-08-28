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
String getType() { return T :: getType(); };

template<> inline String getType< bool >() { return String( "bool" ); };
template<> inline String getType< short int >() { return String( "short int" ); };
template<> inline String getType< int >() { return String( "int" ); };
template<> inline String getType< long int >() { return String( "long int" ); };
template<> inline String getType< unsigned int >() { return String( "unsigned int" ); };
template<> inline String getType< char >() { return String( "char" ); };
template<> inline String getType< float >() { return String( "float" ); };
template<> inline String getType< double >() { return String( "double" ); };
template<> inline String getType< long double >() { return String( "long double" ); };
template<> inline String getType< tnlFloat >() { return String( "tnlFloat" ); };
template<> inline String getType< tnlDouble> () { return String( "tnlDouble" ); };

} // namespace TNL
