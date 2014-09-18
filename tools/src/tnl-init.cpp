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
#include <functions/tnlTestFunction.h>
#include <mesh/tnlDummyMesh.h>
#include <mesh/tnlGrid.h>


void setupConfig( tnlConfigDescription& config )
{
   config.addDelimiter                            ( "General settings:" );
   config.addEntry< tnlString >( "mesh", "Mesh file. If none is given, a regular rectangular mesh is assumed.", "mesh.tnl" );
   config.addRequiredEntry< tnlString >( "function", "Function name." );
      config.addEntryEnum< tnlString >( "exp-bump" );
      config.addEntryEnum< tnlString >( "sin-bumps" );
      config.addEntryEnum< tnlString >( "sin-wave" );
   config.addRequiredEntry< tnlString >( "output-file", "Output file name." );
   config.addEntry< bool >( "check-output-file", "If the output file already exists, do not recreate it.", false );
   config.addEntry< tnlString >( "help", "Write help." );   
   
   config.addDelimiter                            ( "Functions parameters:" );
   tnlTestFunction< 1 >::configSetup( config );
}

int main( int argc, char* argv[] )
{
   tnlParameterContainer parameters;
   tnlConfigDescription conf_desc;

   setupConfig( conf_desc );
   
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
