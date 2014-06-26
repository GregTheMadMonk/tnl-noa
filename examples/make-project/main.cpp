/***************************************************************************
                          main.cpp  -  description
                             -------------------
    begin                : Jan 12, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
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

#include "program-name-conf.h"
#include <config/tnlConfigDescription.h>
#include <config/tnlParameterContainer.h>

int main( int argc, char* argv[] )
{
   tnlParameterContainer parameters;
   tnlConfigDescription conf_desc;
   if( conf_desc.parseConfigDescription( CONFIG_FILE ) != 0 )
      return EXIT_FAILURE;
   if( ! ParseCommandLine( argc, argv, conf_desc, parameters ) )
   {
      conf_desc.printUsage( argv[ 0 ] );
      return EXIT_FAILURE;
   }

   /****
    * Write your code here
    */
   return EXIT_SUCCESS;
}


