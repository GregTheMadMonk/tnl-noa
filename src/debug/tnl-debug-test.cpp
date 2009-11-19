/***************************************************************************
                          tnl-debug-test.cpp  -  description
                             -------------------
    begin                : 2005/07/05
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

#include <iostream.h>
#define HAVE_TNLDEBUG_H
#define DEBUG
#include "tnlDebug.h"

//--------------------------------------------------------------------------
int main( int argc, char* argv[] )
{
   dbgFunctionName( "", "main" );
   dbgInit( "debug.dbg" );

   cout << "tnlDebug test ... " << endl;
   return 0;

}
