/***************************************************************************
                          param-types.h  -  description
                             -------------------
    begin                : 2009/07/29
    copyright            : (C) 2009 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once 

#include <TNL/core/tnlReal.h>
#include <TNL/core/tnlString.h>

namespace TNL {

template< typename T >
tnlString getType() { return T :: getType(); };

template<> inline tnlString getType< bool >() { return tnlString( "bool" ); };
template<> inline tnlString getType< short int >() { return tnlString( "short int" ); };
template<> inline tnlString getType< int >() { return tnlString( "int" ); };
template<> inline tnlString getType< long int >() { return tnlString( "long int" ); };
template<> inline tnlString getType< unsigned int >() { return tnlString( "unsigned int" ); };
template<> inline tnlString getType< char >() { return tnlString( "char" ); };
template<> inline tnlString getType< float >() { return tnlString( "float" ); };
template<> inline tnlString getType< double >() { return tnlString( "double" ); };
template<> inline tnlString getType< long double >() { return tnlString( "long double" ); };
template<> inline tnlString getType< tnlFloat >() { return tnlString( "tnlFloat" ); };
template<> inline tnlString getType< tnlDouble> () { return tnlString( "tnlDouble" ); };

} // namespace TNL
