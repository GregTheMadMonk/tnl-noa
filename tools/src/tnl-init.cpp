/***************************************************************************
                          tnl-init.cpp  -  description
                             -------------------
    begin                : Nov 23, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include "tnl-init.h"

#include <TNL/core/tnlFile.h>
#include <TNL/debug/tnlDebug.h>
#include <TNL/config/tnlConfigDescription.h>
#include <TNL/config/tnlParameterContainer.h>
#include <TNL/functions/tnlTestFunction.h>
#include <TNL/mesh/tnlDummyMesh.h>
#include <TNL/mesh/tnlGrid.h>

using namespace TNL;

void setupConfig( tnlConfigDescription& config )
{
   config.addDelimiter                            ( "General settings:" );
   config.addEntry< tnlString >( "mesh", "Mesh file. If none is given, a regular rectangular mesh is assumed.", "mesh.tnl" );
   config.addEntry< tnlString >( "real-type", "Precision of the function evaluation.", "mesh-real-type" );
      config.addEntryEnum< tnlString >( "mesh-real-type" );
      config.addEntryEnum< tnlString >( "float" );
      config.addEntryEnum< tnlString >( "double" );
      config.addEntryEnum< tnlString >( "long-double" );
   config.addEntry< double >( "initial-time", "Initial time for a serie of snapshots of the time-dependent function.", 0.0 );
   config.addEntry< double >( "final-time", "Final time for a serie of snapshots of the time-dependent function.", 0.0 );
   config.addEntry< double >( "snapshot-period", "Period between snapshots in a serie of the time-dependent function.", 0.0 );
   config.addEntry< int >( "x-derivative", "Order of the partial derivative w.r.t x.", 0 );
   config.addEntry< int >( "y-derivative", "Order of the partial derivative w.r.t y.", 0 );
   config.addEntry< int >( "z-derivative", "Order of the partial derivative w.r.t <.", 0 );
   config.addEntry< bool >( "numerical-differentiation", "The partial derivatives will be computed numerically.", false );
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
 
   if( ! parseCommandLine( argc, argv, conf_desc, parameters ) )
      return EXIT_FAILURE;

   tnlString meshFile = parameters. getParameter< tnlString >( "mesh" );
   tnlString meshType;
   if( ! getObjectType( meshFile, meshType ) )
   {
      std::cerr << "I am not able to detect the mesh type from the file " << meshFile << "." << std::endl;
      return EXIT_FAILURE;
   }
   std::cout << meshType << " detected in " << meshFile << " file." << std::endl;
   tnlList< tnlString > parsedMeshType;
   if( ! parseObjectType( meshType, parsedMeshType ) )
   {
      std::cerr << "Unable to parse the mesh type " << meshType << "." << std::endl;
      return false;
   }
   if( ! resolveMeshType( parsedMeshType, parameters ) )
      return EXIT_FAILURE;
   return EXIT_SUCCESS;
}
