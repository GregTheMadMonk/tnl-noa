/***************************************************************************
                          tnl-mesh-convert.h  -  description
                             -------------------
    begin                : Feb 19, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNL_MESH_CONVERT_H_
#define TNL_MESH_CONVERT_H_

#include <TNL/Config/ParameterContainer.h>
#include <TNL/Meshes/MeshDetails/MeshReaderNetgen.h>
#include <TNL/Meshes/MeshDetails/MeshWriterVTKLegacy.h>
#include <TNL/Meshes/MeshConfigBase.h>
#include <TNL/Meshes/Topologies/MeshTriangleTopology.h>
#include <TNL/Meshes/Topologies/MeshTetrahedronTopology.h>
#include <TNL/Meshes/Mesh.h>
#include <TNL/Meshes/MeshDetails/initializer/MeshInitializer.h>
#include <TNL/Meshes/MeshDetails/MeshIntegrityChecker.h>
#include <TNL/FileName.h>

using namespace TNL;
using namespace TNL::Meshes;

template< typename MeshReader,
          typename MeshType >
bool convertMesh( const Config::ParameterContainer& parameters )
{
   const String& inputFileName = parameters.getParameter< String >( "input-file" );
   const String& outputFileName = parameters.getParameter< String >( "output-file" );
   const String outputFileExt = getFileExtension( outputFileName );

   MeshType mesh;
   if( ! MeshReader::readMesh( inputFileName, mesh, true ) )
      return false;
   /*MeshInitializer< typename MeshType::Config > meshInitializer;
   meshInitializer.setVerbose( true );
   if( ! meshInitializer.initMesh( mesh ) )
      return false;
   if( ! MeshIntegrityChecker< MeshType >::checkMesh( mesh ) )
      return false;*/
  std::cout << mesh << std::endl;
  std::cout << "Writing the mesh to a file " << outputFileName << "." << std::endl;
   if( outputFileExt == "tnl" )
   {
      if( ! mesh.save( outputFileName ) )
      {
         std::cerr << "I am not able to write the mesh into the file " << outputFileName << "." << std::endl;
         return false;
      }
   }
   if( outputFileExt == "vtk" )
   {
      if( ! MeshWriterVTKLegacy::write( outputFileName, mesh, true ) )
      {
         std::cerr << "I am not able to write the mesh into the file " << outputFileName << "." << std::endl;
         return false;
      }
      return true;
   }
}

bool readNetgenMesh( const Config::ParameterContainer& parameters )
{
   const String& inputFileName = parameters.getParameter< String >( "input-file" );
 
   MeshReaderNetgen meshReader;
   if( ! meshReader.detectMesh( inputFileName ) )
      return false;

  std::cout << "Reading mesh with " << meshReader.getDimensions() << " dimensions..." << std::endl;
 
   if( meshReader.getDimensions() == 2 )
   {
      if( meshReader.getVerticesInCell() == 3 )
      {
         typedef Mesh< MeshConfigBase< MeshTriangleTopology > > MeshType;
        std::cout << "Mesh consisting of triangles was detected ... " << std::endl;
         return convertMesh< MeshReaderNetgen, MeshType >( parameters );
      }
      if( meshReader.getVerticesInCell() == 4 )
      {
         typedef Mesh< MeshConfigBase< MeshQuadrilateralTopology > > MeshType;
        std::cout << "Mesh consisting of quadrilaterals was detected ... " << std::endl;
         return convertMesh< MeshReaderNetgen, MeshType >( parameters );
      }
   }
   if( meshReader.getDimensions() == 3 )
   {
      if( meshReader.getVerticesInCell() == 4 )
      {
         typedef Mesh< MeshConfigBase< MeshTetrahedronTopology > > MeshType;
        std::cout << "Mesh consisting of tetrahedrons was detected ... " << std::endl;
         return convertMesh< MeshReaderNetgen, MeshType >( parameters );
      }
      if( meshReader.getVerticesInCell() == 8 )
      {
         typedef Mesh< MeshConfigBase< MeshHexahedronTopology > > MeshType;
        std::cout << "Mesh consisting of hexahedrons was detected ... " << std::endl;
         return convertMesh< MeshReaderNetgen, MeshType >( parameters );
      }
   }
   std::cerr << "Wrong mesh dimensions were detected ( " << meshReader.getDimensions() << " )." << std::endl;
   return false;
}

bool convertMesh( const Config::ParameterContainer& parameters )
{
   String inputFileName = parameters.getParameter< String >( "input-file" );

   const String fileExt = getFileExtension( inputFileName );
   if( fileExt == "ng" )
      return readNetgenMesh( parameters );
}


#endif /* TNL_MESH_CONVERT_H_ */
