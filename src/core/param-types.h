/***************************************************************************
                          param-types.h  -  description
                             -------------------
    begin                : 2009/07/29
    copyright            : (C) 2009 by Tomá¹ Oberhuber
    email                : oberhuber@seznam.cz
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

#include "mList.h"
#include "mString.h"

inline mString GetParameterType( bool ) { return mString( "bool" ); };
inline mString GetParameterType( int ) { return mString( "int" ); };
inline mString GetParameterType( char ) { return mString( "char" ); };
inline mString GetParameterType( const float& ) { return mString( "float" ); };
inline mString GetParameterType( const double& ) { return mString( "double" ); };

#endif
