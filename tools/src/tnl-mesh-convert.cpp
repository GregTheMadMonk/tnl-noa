/***************************************************************************
                          tnl-mesh-convert.cpp  -  description
                             -------------------
    begin                : Feb 19, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef HAVE_ICPC
#include "tnl-mesh-convert.h"
#endif
#include <TNL/config/tnlParameterContainer.h>

void configSetup( tnlConfigDescription& config )
{
   config.addDelimiter                            ( "General settings:" );
   config.addRequiredEntry< tnlString >( "input-file", "Input file with the mesh." );
   config.addEntry< tnlString >( "output-file", "Output mesh file in TNL or VTK format.", "mesh.tnl" );
   //config.addEntry< tnlString >( "output-format", "Output mesh file format.", "vtk" );
   config.addEntry< int >( "verbose", "Set the verbosity of the program.", 1 );
   config.addEntry< tnlString >( "mesh-name", "The mesh name.", "tnl-mesh" );
}

int main( int argc, char* argv[] )
{
   tnlParameterContainer parameters;
   tnlConfigDescription conf_desc;
 
   configSetup( conf_desc );

   if( ! parseCommandLine( argc, argv, conf_desc, parameters ) )
   {
      conf_desc.printUsage( argv[ 0 ] );
      return EXIT_FAILURE;
   }
#ifndef HAVE_ICPC
   if( ! convertMesh( parameters ) )
      return EXIT_FAILURE;
#endif
   return EXIT_SUCCESS;
}


