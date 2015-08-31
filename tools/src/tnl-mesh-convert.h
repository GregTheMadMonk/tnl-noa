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
#include <mesh/tnlMeshWriterVTKLegacy.h>
#include <mesh/config/tnlMeshConfigBase.h>
#include <mesh/topologies/tnlMeshTriangleTopology.h>
#include <mesh/topologies/tnlMeshTetrahedronTopology.h>
#include <mesh/tnlMesh.h>
#include <mesh/tnlMeshInitializer.h>
#include <mesh/tnlMeshIntegrityChecker.h>
#include <core/mfilename.h>

template< typename MeshReader,
          typename MeshType >
bool convertMesh( const tnlParameterContainer& parameters )
{
   const tnlString& inputFileName = parameters.getParameter< tnlString >( "input-file" );
   const tnlString& outputFileName = parameters.getParameter< tnlString >( "output-file" );
   const tnlString outputFileExt = getFileExtension( outputFileName );

   MeshType mesh;
   if( ! MeshReader::readMesh( inputFileName, mesh, true ) )
      return false;
   /*tnlMeshInitializer< typename MeshType::Config > meshInitializer;
   meshInitializer.setVerbose( true );
   if( ! meshInitializer.initMesh( mesh ) )
      return false;
   if( ! tnlMeshIntegrityChecker< MeshType >::checkMesh( mesh ) )
      return false;*/
   cout << "Writing the mesh to a file " << outputFileName << "." << endl;
   if( outputFileExt == "tnl" )
   {         
      if( ! mesh.save( outputFileName ) )
      {
         cerr << "I am not able to write the mesh into the file " << outputFileName << "." << endl;
         return false;
      }
   }
   if( outputFileExt == "vtk" )
   {
      if( ! tnlMeshWriterVTKLegacy::write( outputFileName, mesh, true ) )
      {
         cerr << "I am not able to write the mesh into the file " << outputFileName << "." << endl;
         return false;         
      }
      return true;
   }   
}

bool readNetgenMesh( const tnlParameterContainer& parameters )
{
   const tnlString& inputFileName = parameters.getParameter< tnlString >( "input-file" );
   
   tnlMeshReaderNetgen meshReader;
   if( ! meshReader.detectMesh( inputFileName ) )
      return false;

   cout << "Reading mesh with " << meshReader.getDimensions() << " dimensions..." << endl;
   
   if( meshReader.getDimensions() == 2 )
   {
      if( meshReader.getVerticesInCell() == 3 )
      {
         typedef tnlMesh< tnlMeshConfigBase< tnlMeshTriangleTopology > > MeshType;
         cout << "Mesh consisting of triangles was detected ... " << endl;
         return convertMesh< tnlMeshReaderNetgen, MeshType >( parameters );
      }
      if( meshReader.getVerticesInCell() == 4 )
      {
         typedef tnlMesh< tnlMeshConfigBase< tnlMeshQuadrilateralTopology > > MeshType;
         cout << "Mesh consisting of quadrilaterals was detected ... " << endl;
         return convertMesh< tnlMeshReaderNetgen, MeshType >( parameters );
      }            
   }
   if( meshReader.getDimensions() == 3 )
   {
      if( meshReader.getVerticesInCell() == 4 )
      {
         typedef tnlMesh< tnlMeshConfigBase< tnlMeshTetrahedronTopology > > MeshType;
         cout << "Mesh consisting of tetrahedrons was detected ... " << endl;
         return convertMesh< tnlMeshReaderNetgen, MeshType >( parameters );
      }
      if( meshReader.getVerticesInCell() == 8 )
      {
         typedef tnlMesh< tnlMeshConfigBase< tnlMeshHexahedronTopology > > MeshType;
         cout << "Mesh consisting of hexahedrons was detected ... " << endl;
         return convertMesh< tnlMeshReaderNetgen, MeshType >( parameters );
      }
   }
   cerr << "Wrong mesh dimensions were detected ( " << meshReader.getDimensions() << " )." << endl;
   return false;
}

bool convertMesh( const tnlParameterContainer& parameters )
{
   tnlString inputFileName = parameters.getParameter< tnlString >( "input-file" );

   const tnlString fileExt = getFileExtension( inputFileName );
   if( fileExt == "ng" )
      return readNetgenMesh( parameters );
}


#endif /* TNL_MESH_CONVERT_H_ */
