/***************************************************************************
                          debug.h  -  description
                             -------------------
    begin                : 2005/07/02
    copyright            : (C) 2005 by Tomas Oberhuber
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

#ifndef debugH
#define debugH

//#include <mdiff-config.h>

#ifdef HAVE_MDIFF_POUND_H
#include <pound.h>
#else
   #define dbgInit( file_name )
   #define dbgFunctionName( _class, _func )
   #define dbgCout( args )
   #define dbgExpr( expr )
   #define DBG_PRINTF( expr )
   #define DBG_COND_COUT( condition, args )
   #define DBG_COND_EXPR( condition, expr )
   #define DBG_COND_PRINTF( condition, expr )
   #define DBG_MPI_BARRIER
#endif

#endif
