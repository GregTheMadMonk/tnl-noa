/***************************************************************************
                          tnl-grid-setup.cpp  -  description
                             -------------------
    begin                : Nov 20, 2013
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

#include "tnl-grid-setup.h"
#include "tnlConfig.h"
#include <config/tnlParameterContainer.h>

const char configFile[] = TNL_CONFIG_DIRECTORY "tnl-grid-setup.cfg.desc";

int main( int argc, char* argv[] )
{
   tnlParameterContainer parameters;
   tnlConfigDescription conf_desc;
   if( conf_desc. ParseConfigDescription( configFile ) != 0 )
      return EXIT_FAILURE;
   if( ! ParseCommandLine( argc, argv, conf_desc, parameters ) )
   {
      conf_desc. PrintUsage( argv[ 0 ] );
      return EXIT_FAILURE;
   }
   if( ! resolveRealType( parameters ) )
      return EXIT_FAILURE;
   return EXIT_SUCCESS;
}


