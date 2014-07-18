/***************************************************************************
                          tnl-mesh-convert.cpp  -  description
                             -------------------
    begin                : Feb 19, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
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

#ifndef HAVE_NOT_CXX11
#include "tnl-mesh-convert.h"
#endif
#include "tnlConfig.h"
#include <config/tnlParameterContainer.h>

const char configFile[] = TNL_CONFIG_DIRECTORY "tnl-mesh-convert.cfg.desc";

int main( int argc, char* argv[] )
{
   tnlParameterContainer parameters;
   tnlConfigDescription conf_desc;
   if( conf_desc.parseConfigDescription( configFile ) != 0 )
      return EXIT_FAILURE;
   if( ! ParseCommandLine( argc, argv, conf_desc, parameters ) )
   {
      conf_desc.printUsage( argv[ 0 ] );
      return EXIT_FAILURE;
   }
#ifndef HAVE_NOT_CXX11
   if( ! convertMesh( parameters ) )
      return EXIT_FAILURE;
#endif
   return EXIT_SUCCESS;
}


