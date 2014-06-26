/***************************************************************************
                          tnl-diff.cpp  -  description
                             -------------------
    begin                : Nov 17, 2013
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

#include "tnl-diff.h"
#include <mesh/tnlDummyMesh.h>
#include <mesh/tnlGrid.h>
#include <mesh/tnlLinearGridGeometry.h>

void setupConfig( tnlConfigDescription& config )
{
   config.addEntry< tnlString >( "mesh", "Input mesh file.", "mesh.tnl" );
   config.addRequiredEntry< tnlList< tnlString > >( "input-files", "The first set of the input files." );
   config.addEntry< tnlString >( "output-file", "File for the output data.", "tnl-diff.log" );
   config.addEntry< tnlString >( "mode", "Mode 'couples' compares two subsequent files. Mode 'sequence' compares the input files against the first one. 'halves' compares the files from the and the second half of the intput files.", "couples" );
      config.addEntryEnum< tnlString >( "couples" );
      config.addEntryEnum< tnlString >( "sequence" );
      config.addEntryEnum< tnlString >( "halves" );
   config.addEntry< bool >( "write-difference", "Write difference grid function.", false );
   config.addEntry< bool >( "write-exact-curve", "Write exact curve with given radius.", false );
   config.addEntry< int >( "edges-skip", "Width of the edges that will be skipped - not included into the error norms.", 0 );
   config.addEntry< bool >( "write-graph", "Draws a graph in the Gnuplot format of the dependence of the error norm on t.", true );
   config.addEntry< bool >( "write-log-graph", "Draws a logarithmic graph in the Gnuplot format of the dependence of the error norm on t.", true );
   config.addEntry< double >( "tau", "Time period between the subsequent solutions.", 0.0 );
   config.addEntry< int >( "verbose", "Sets verbosity.", 1 );
}

int main( int argc, char* argv[] )
{
   tnlParameterContainer parameters;
   tnlConfigDescription conf_desc;
   setupConfig( conf_desc );
   if( ! ParseCommandLine( argc, argv, conf_desc, parameters ) )
   {
      conf_desc.printUsage( argv[ 0 ] );
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
   if( parsedMeshType[ 0 ] == "tnlGrid" )
   {
      tnlList< tnlString > parsedGeometryType;
      if( ! parseObjectType( parsedMeshType[ 5 ], parsedGeometryType ) )
      {
         cerr << "Unable to parse the geometry type " << parsedMeshType[ 5 ] << "." << endl;
         return false;
      }
      if( parsedGeometryType[ 0 ] == "tnlIdenticalGridGeometry" )
      {
         int dimensions = atoi( parsedGeometryType[ 1 ].getString() );
         if( dimensions == 1 )
         {
            typedef tnlGrid< 1, double, tnlHost, int, tnlIdenticalGridGeometry > MeshType;
            if( ! processFiles< MeshType >( parameters ) )
               return EXIT_FAILURE;
         }
         if( dimensions == 2 )
         {
            typedef tnlGrid< 2, double, tnlHost, int, tnlIdenticalGridGeometry > MeshType;
            if( ! processFiles< MeshType >( parameters ) )
               return EXIT_FAILURE;
         }
         if( dimensions == 3 )
         {
            typedef tnlGrid< 3, double, tnlHost, int, tnlIdenticalGridGeometry > MeshType;
            if( ! processFiles< MeshType >( parameters ) )
               return EXIT_FAILURE;
         }
         return EXIT_SUCCESS;
      }
      if( parsedGeometryType[ 0 ] == "tnlLinearGridGeometry" )
      {
         int dimensions = atoi( parsedGeometryType[ 1 ].getString() );
         if( dimensions == 1 )
         {
            typedef tnlGrid< 1, double, tnlHost, int, tnlLinearGridGeometry > MeshType;
            if( ! processFiles< MeshType >( parameters ) )
               return EXIT_FAILURE;
         }
         if( dimensions == 2 )
         {
            typedef tnlGrid< 2, double, tnlHost, int, tnlLinearGridGeometry > MeshType;
            if( ! processFiles< MeshType >( parameters ) )
               return EXIT_FAILURE;
         }
         if( dimensions == 3 )
         {
            typedef tnlGrid< 3, double, tnlHost, int, tnlLinearGridGeometry > MeshType;
            if( ! processFiles< MeshType >( parameters ) )
               return EXIT_FAILURE;
         }
         return EXIT_SUCCESS;
      }

   }
   return EXIT_FAILURE;
}


