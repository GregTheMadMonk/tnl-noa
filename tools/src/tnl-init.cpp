/***************************************************************************
                          tnl-init.cpp  -  description
                             -------------------
    begin                : Nov 23, 2013
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

#include "tnl-init.h"

#include <core/tnlFile.h>
#include <debug/tnlDebug.h>
#include <config/tnlConfigDescription.h>
#include <config/tnlParameterContainer.h>
#include <mesh/tnlDummyMesh.h>
#include <mesh/tnlGrid.h>

#include "tnlConfig.h"
const char configFile[] = TNL_CONFIG_DIRECTORY "tnl-discrete.cfg.desc";

int main( int argc, char* argv[] )
{
   tnlParameterContainer parameters;
   tnlConfigDescription conf_desc;
   if( conf_desc.parseConfigDescription( configFile ) != 0 )
      return 1;
   if( ! ParseCommandLine( argc, argv, conf_desc, parameters ) )
      return EXIT_FAILURE;

   tnlString meshFile = parameters. GetParameter< tnlString >( "mesh" );
   tnlString meshType;
   if( ! getObjectType( meshFile, meshType ) )
   {
      cerr << "I am not able to detect the mesh type from the file " << meshFile << "." << endl;
      return EXIT_FAILURE;
   }
   cout << meshType << " detected in " << meshFile << " file." << endl;
   tnlList< tnlString > parsedMeshType;
   if( ! parseObjectType( meshType, parsedMeshType ) )
   {
      cerr << "Unable to parse the mesh type " << meshType << "." << endl;
      return false;
   }
   if( ! resolveMeshType( parsedMeshType, parameters ) )
      return EXIT_FAILURE;
   return EXIT_SUCCESS;
}
