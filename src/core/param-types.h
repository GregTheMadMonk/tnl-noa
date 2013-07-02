/***************************************************************************
                          param-types.h  -  description
                             -------------------
    begin                : 2009/07/29
    copyright            : (C) 2009 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef param_typesH
#define param_typesH

#include <core/tnlReal.h>
#include <core/tnlList.h>
#include <core/tnlString.h>

template< typename T >
tnlString getParameterType() { return T :: getType(); };

template<> inline tnlString getParameterType< bool >() { return tnlString( "bool" ); };
template<> inline tnlString getParameterType< int >() { return tnlString( "int" ); };
template<> inline tnlString getParameterType< long int >() { return tnlString( "long int" ); };
template<> inline tnlString getParameterType< char >() { return tnlString( "char" ); };
template<> inline tnlString getParameterType< float >() { return tnlString( "float" ); };
template<> inline tnlString getParameterType< double >() { return tnlString( "double" ); };
template<> inline tnlString getParameterType< long double >() { return tnlString( "long double" ); };
template<> inline tnlString getParameterType< tnlFloat >() { return tnlString( "tnlFloat" ); };
template<> inline tnlString getParameterType< tnlDouble> () { return tnlString( "tnlDouble" ); };

#endif
