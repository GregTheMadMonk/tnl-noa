/***************************************************************************
                          tnl-view.cpp  -  description
                             -------------------
    begin                : Jan 21, 2013
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

#include "tnl-view.h"
#include <cstdlib>
#include <core/tnlFile.h>
#include <debug/tnlDebug.h>
#include <config/tnlConfigDescription.h>
#include <config/tnlParameterContainer.h>
#include <mesh/tnlDummyMesh.h>

#include "tnlConfig.h"
const char configFile[] = TNL_CONFIG_DIRECTORY "tnl-view.cfg.desc";

int main( int argc, char* argv[] )
{
   tnlParameterContainer parameters;
   tnlConfigDescription conf_desc;
   if( conf_desc. ParseConfigDescription( configFile ) != 0 )
      return 1;
   if( ! ParseCommandLine( argc, argv, conf_desc, parameters ) )
   {
      conf_desc. PrintUsage( argv[ 0 ] );
      return 1;
   }

   int verbose = parameters. GetParameter< int >( "verbose" );
   tnlString meshFile = parameters. GetParameter< tnlString >( "mesh" );
   if( meshFile == "" )
   {
      if( ! processFiles< tnlDummyMesh< double, tnlHost, int > >( parameters ) )
         return EXIT_FAILURE;
      return EXIT_SUCCESS;
   }
   tnlString meshType;
   if( getObjectType( meshFile, meshType ) )
   {
      cerr << "I am not able to detect the mesh type from the file " << meshFile << "." << endl;
      return EXIT_FAILURE;
   }
   cout << meshType << " detected in " << meshFile << " file." << endl;
   if( meshType == "tnlGrid< 2, double, tnlHost, int >" )
   {
      if( ! processFiles< tnlGrid< 2, double, tnlHost, int > >( parameters ) )
         return EXIT_FAILURE;
      return EXIT_SUCCESS;
   }
   return EXIT_FAILURE;
}
