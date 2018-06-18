/***************************************************************************
                          tnl-lattice-init.cpp  -  description
                             -------------------
    begin                : Jun 13, 2018
    copyright            : (C) 2018 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include "tnl-lattice-init.h"

#include <TNL/Config/ConfigDescription.h>
#include <TNL/Config/ParameterContainer.h>


using namespace TNL;

void setupConfig( Config::ConfigDescription& config )
{
   config.addDelimiter                            ( "General settings:" );
   config.addEntry< String >( "mesh", "Mesh file of the 3D lattice.", "mesh.tnl" );
   config.addEntry< String >( "real-type", "Precision of the function evaluation.", "mesh-real-type" );
      config.addEntryEnum< String >( "mesh-real-type" );
      config.addEntryEnum< String >( "float" );
      config.addEntryEnum< String >( "double" );
      config.addEntryEnum< String >( "long-double" );
   config.addEntry< String >( "profile-mesh", "Mesh file of the 2D mesh function with geometry profile", "profile-mesh.tnl" );
   config.addEntry< String >( "profile-file", "The profile mesh function file.", "profile.tnl" );
   config.addEntry< String >( "output-file", "Output 3D mesh function file.", "init.tnl" );
   config.addEntry< String >( "operation", "Operation to be done with the profile.", "extrude" );
      config.addEntryEnum< String >( "extrude" );
   config.addEntry< String >( "profile-orientation", "Axis the profile is orthogonal to.", "z" );
      config.addEntryEnum< String >( "x" );
      config.addEntryEnum< String >( "y" );
      config.addEntryEnum< String >( "z" );
   config.addEntry< double >( "extrude-start", "Position where the extrude operation starts.", 0.0 );
   config.addEntry< double >( "extrude-stop", "Position where the extrude operation stops.", 1.0 );
}



int main( int argc, char* argv[] )
{

   Config::ParameterContainer parameters;
   Config::ConfigDescription configDescription;

   setupConfig( configDescription );
   
   if( ! Config::parseCommandLine( argc, argv, configDescription, parameters ) )
      return EXIT_FAILURE;
   
   if( ! resolveProfileMeshType( parameters ) )
      return EXIT_FAILURE;

   return EXIT_SUCCESS;   
}