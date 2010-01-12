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

#include "tnlList.h"
#include "tnlString.h"

inline tnlString GetParameterType( bool ) { return tnlString( "bool" ); };
inline tnlString GetParameterType( int ) { return tnlString( "int" ); };
inline tnlString GetParameterType( char ) { return tnlString( "char" ); };
inline tnlString GetParameterType( const float& ) { return tnlString( "float" ); };
inline tnlString GetParameterType( const double& ) { return tnlString( "double" ); };

#endif
