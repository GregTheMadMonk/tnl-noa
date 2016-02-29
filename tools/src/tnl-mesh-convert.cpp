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

#ifndef HAVE_ICPC
#include "tnl-mesh-convert.h"
#endif
#include "tnlConfig.h"
#include <config/tnlParameterContainer.h>

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


