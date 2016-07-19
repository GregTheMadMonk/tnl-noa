/***************************************************************************
                          tnl-debug-test.cpp  -  description
                             -------------------
    begin                : 2005/07/05
    copyright            : (C) 2005 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <iostream>
#define HAVE_TNLDEBUG_H
#define DEBUG
#include "tnlDebug.h"

using namespace std;

//--------------------------------------------------------------------------
int main( int argc, char* argv[] )
{
   dbgFunctionName( "", "main" );
   //dbgInit( "debug.dbg" );

   std::cout << "tnlDebug test ... " << std::endl;
   return 0;

}
