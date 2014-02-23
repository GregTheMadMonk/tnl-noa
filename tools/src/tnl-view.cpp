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
#include <mesh/tnlGrid.h>
#include <mesh/tnlIdenticalGridGeometry.h>
#include <mesh/tnlLinearGridGeometry.h>

// TODO: Remove
#include <mesh/tnlMesh.h>
#include <mesh/tnlMeshWriterNetgen.h>
#include <mesh/config/tnlMeshConfigBase.h>
#include <mesh/topologies/tnlMeshTriangleTag.h>

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
   if( parsedMeshType[ 0 ] == "tnlMesh" )
   {
      tnlString meshFile = parameters. GetParameter< tnlString >( "mesh" );
      struct MeshConfig : public tnlMeshConfigBase< 2 >
      {
         typedef tnlMeshTriangleTag CellTag;
      };
      tnlMesh< MeshConfig > mesh;
      if( ! mesh.load( meshFile ) )
         return EXIT_FAILURE;
      if( ! tnlMeshWriterNetgen::writeMesh( "tnl-mesh.ng", mesh, true ) )
         return EXIT_FAILURE;
   }
   return EXIT_FAILURE;
}
