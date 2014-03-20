/***************************************************************************
                          tnl-mesh-convert.h  -  description
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

#ifndef TNL_MESH_CONVERT_H_
#define TNL_MESH_CONVERT_H_

#include <config/tnlParameterContainer.h>
#include <mesh/tnlMeshReaderNetgen.h>
#include <mesh/config/tnlMeshConfigBase.h>
#include <mesh/topologies/tnlMeshTriangleTag.h>
#include <mesh/topologies/tnlMeshTetrahedronTag.h>
#include <mesh/tnlMesh.h>
#include <mesh/tnlMeshInitializer.h>
#include <mesh/tnlMeshIntegrityChecker.h>
#include <core/mfilename.h>

template< int Dimensions >
bool readMeshWithDimensions( const tnlParameterContainer& parameters )
{
   const tnlString& inputFileName = parameters.GetParameter< tnlString >( "input-mesh-file" );
   const tnlString fileExt = getFileExtension( inputFileName );

   if( Dimensions == 2 )
   {
      struct MeshConfig : public tnlMeshConfigBase< 2 >
      {
         typedef tnlMeshTriangleTag CellTag;
      };            
      typedef tnlMesh< MeshConfig >  MeshType;
      MeshType mesh;
      if( fileExt == "ng" &&
          ! tnlMeshReaderNetgen::readMesh<>( inputFileName, mesh, true ) )
         return false;
      tnlMeshInitializer< MeshConfig > meshInitializer;
      meshInitializer.setVerbose( true );
      if( ! meshInitializer.initMesh( mesh ) )
         return false;
      if( ! tnlMeshIntegrityChecker< MeshType >::checkMesh( mesh ) )
         return false;
      tnlString outputFile;
      if( parameters.GetParameter< tnlString >( "output-file", outputFile ) )
      {
         cout << "Writing the 2D mesh to the file " << outputFile << "." << endl;
         if( ! mesh.save( outputFile ) )
         {
            cerr << "I am not able to safe the mesh into the file " << outputFile << "." << endl;
            return false;
         }
      }
   }
   if( Dimensions == 3 )
   {
      struct MeshConfig : public tnlMeshConfigBase< 3 >
      {
         typedef tnlMeshTetrahedronTag CellTag;
      };
      typedef tnlMesh< MeshConfig > MeshType;
      MeshType mesh;
      if( fileExt == "ng" &&
          ! tnlMeshReaderNetgen::readMesh<>( inputFileName, mesh, true ) )
         return false;
      tnlMeshInitializer< MeshConfig > meshInitializer;
      meshInitializer.setVerbose( true );
      if( ! meshInitializer.initMesh( mesh ) )
         return false;
      if( ! tnlMeshIntegrityChecker< MeshType >::checkMesh( mesh ) )
         return false;
      tnlString outputFile;
      if( parameters.GetParameter< tnlString >( "output-file", outputFile ) )
      {
         cout << "Writing the 3D mesh to the file " << outputFile << "." << endl;
         if( ! mesh.save( outputFile ) )
         {
            cerr << "I am not able to safe the mesh into the file " << outputFile << "." << endl;
            return false;
         }
      }
   }

   return true;
}

bool convertMesh( const tnlParameterContainer& parameters )
{
   tnlString inputFileName;
   if( parameters.GetParameter( "input-mesh-file", inputFileName ) )
   {
      const tnlString fileExt = getFileExtension( inputFileName );
      if( fileExt == "ng" )
      {
         int dimensions;
         if( ! tnlMeshReaderNetgen::detectDimensions( inputFileName, dimensions ) )
            return false;
         if( dimensions == 2 &&
             ! readMeshWithDimensions< 2 >( parameters ) )
            return false;
         if( dimensions == 3 &&
             ! readMeshWithDimensions< 3 >( parameters ) )
            return false;
      }
   }
   return true;
}

#endif /* TNL_MESH_CONVERT_H_ */
