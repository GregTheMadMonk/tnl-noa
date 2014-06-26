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
#include <config/tnlParameterContainer.h>

void configSetup( tnlConfigDescription& config )
{
   config.addDelimiter( "General parameters" );
   config.addEntry        < tnlString >( "output-file",   "Output file.", "mesh.tnl" );
   config.addEntry        < int >      ( "verbose",       "Set the verbosity of the program.", 1 );

   config.addDelimiter                 ( "Grid parameters" );
   config.addEntry        < tnlString >( "grid-name",     "The grid name.", "tnl-grid" );
   config.addRequiredEntry< int >      ( "dimensions",    "The grid dimensions." );
   config.addEntry        < tnlString >( "real-type",     "Precision of the real type describing the grid.", "double" );
      config.addEntryEnum < tnlString >( "float" );
      config.addEntryEnum < tnlString >( "double" );
      config.addEntryEnum < tnlString >( "long-double" );
   config.addEntry        < tnlString >( "index-type",    "Type for the indexing of the grid elements." ,"int" );
      config.addEntryEnum < tnlString >( "int" );
      config.addEntryEnum < tnlString >( "long-int" );
   config.addEntry        < double >   ( "origin-x",      "The x-coordinate of the origin.", 0.0 );
   config.addEntry        < double >   ( "origin-y",      "The y-coordinate of the origin.", 0.0 );
   config.addEntry        < double >   ( "origin-z",      "The z-coordinate of the origin.", 0.0 );
   config.addEntry        < double >   ( "proportions-x", "The proportions of the grid along the x axis.", 1.0 );
   config.addEntry        < double >   ( "proportions-y", "The proportions of the grid along the y axis.", 1.0 );
   config.addEntry        < double >   ( "proportions-z", "The proportions of the grid along the z axis.", 1.0 );
   config.addEntry        < int >      ( "size-x",        "Number of elements along the x axis." );
   config.addEntry        < int >      ( "size-y",        "Number of elements along the y axis." );
   config.addEntry        < int >      ( "size-z",        "Number of elements along the z axis." );
}

int main( int argc, char* argv[] )
{
   tnlParameterContainer parameters;
   tnlConfigDescription conf_desc;
   configSetup( conf_desc );
   if( ! ParseCommandLine( argc, argv, conf_desc, parameters ) )
   {
      conf_desc.printUsage( argv[ 0 ] );
      return EXIT_FAILURE;
   }
   if( ! resolveRealType( parameters ) )
      return EXIT_FAILURE;
   return EXIT_SUCCESS;
}


