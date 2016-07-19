/***************************************************************************
                          tnlMeshWriterVTKLegacy.h  -  description
                             -------------------
    begin                : Mar 20, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <fstream>
#include <istream>
#include <sstream>
#include <iomanip>

#include <mesh/topologies/tnlMeshTriangleTopology.h>
#include <mesh/topologies/tnlMeshQuadrilateralTopology.h>
#include <mesh/topologies/tnlMeshTetrahedronTopology.h>
#include <mesh/topologies/tnlMeshHexahedronTopology.h>
#include <mesh/tnlMeshEntity.h>

namespace TNL {

enum tnlVTKMeshEntities { tnlVTKVertex = 1,
                          tnlVTKPolyVertex = 2,
                          tnlVTKLine = 3,
                          tnlVTKPolyLine = 4,
                          tnlVTKTriangle = 5,
                          tnlVTKTriangleStrip = 6,
                          tnlVTKPolygon = 7,
                          tnlVTKPixel = 8,
                          tnlVTKQuad = 9,
                          tnlVTKTetra = 10,
                          tnlVTKVoxel = 11,
                          tnlVTKHexahedron = 12,
                          tnlVTKWedge = 13,
                          tnlVTKPyramid = 14 };

template< typename MeshEntity >
struct tnlMeshEntityVTKType{};

template< typename MeshConfig > struct tnlMeshEntityVTKType< tnlMeshEntity< MeshConfig, tnlMeshTriangleTopology > >     { enum { VTKType = tnlVTKTriangle }; };
template< typename MeshConfig > struct tnlMeshEntityVTKType< tnlMeshEntity< MeshConfig, tnlMeshQuadrilateralTopology > >{ enum { VTKType = tnlVTKQuad }; };
template< typename MeshConfig > struct tnlMeshEntityVTKType< tnlMeshEntity< MeshConfig, tnlMeshTetrahedronTopology > >  { enum { VTKType = tnlVTKTetra }; };
template< typename MeshConfig > struct tnlMeshEntityVTKType< tnlMeshEntity< MeshConfig, tnlMeshHexahedronTopology > >   { enum { VTKType = tnlVTKHexahedron }; };

class tnlMeshWriterVTKLegacy
{
   public:

   template< typename MeshType >
   static bool write( const tnlString& fileName,
                      MeshType& mesh,
                      bool verbose )
   {
      if( MeshType::dimensions > 3 )
      {
         std::cerr << "You try to write mesh with " << MeshType::dimensions
              << "dimensions but VTK legacy format supports only 1D, 2D and 3D meshes." << std::endl;
         return false;
      }
      std::fstream outputFile;
      outputFile.open( fileName.getString(), std::ios::out );
      if( ! outputFile )
      {
         std::cerr << "I am not able to open the output file " << fileName << "." << std::endl;
         return false;
      }
      outputFile << std::setprecision( 6 );
      outputFile << std::fixed;

      if( ! writeMesh( outputFile, mesh, verbose ) )
         return false;
   }

   template< typename MeshType >
   static bool writeMesh( std::ostream& file,
                          MeshType& mesh,
                          bool verbose )
   {
      typedef typename MeshType::MeshTraits::CellType CellType;
      file << "# vtk DataFile Version 2.0" << std::endl;
      file << "TNL Mesh" << std::endl;
      file << "ASCII" << std::endl;
      file << "DATASET UNSTRUCTURED_GRID" << std::endl;
      file << std::endl;
      file << "POINTS " << mesh.template getNumberOfEntities< 0 >() << " double" << std::endl;
      for( int i = 0; i < mesh.template getNumberOfEntities< 0 >(); i++ )
      {
         mesh.template getEntity< 0 >( i ).getPoint().write( file );
         for( int j = MeshType::dimensions; j < 3; j++ )
            file << " 0.0";
         file << std::endl;
      }
      file << std::endl;
      file << "CELLS " << mesh.getNumberOfCells();
      long int listSize( 0 );
      for( int i = 0; i < mesh.getNumberOfCells(); i++ )
         listSize += mesh.getCell( i ).template getNumberOfSubentities< 0 >() + 1;
      file << " " << listSize << std::endl;
      for( int i = 0; i < mesh.getNumberOfCells(); i++ )
      {
         int numberOfVertices = mesh.getCell( i ).template getNumberOfSubentities< 0 >();
         file << numberOfVertices << " ";
         for( int j = 0; j < numberOfVertices - 1; j++ )
            file << mesh.getCell( i ).template getSubentityIndex< 0 >( j ) << " ";
         file << mesh.getCell( i ).template getSubentityIndex< 0 >( numberOfVertices - 1 ) << std::endl;
      }
      file << std::endl;
      file << "CELL_TYPES " <<  mesh.getNumberOfCells() << std::endl;
      for( int i = 0; i < mesh.getNumberOfCells(); i++ )
      {
         file << tnlMeshEntityVTKType< CellType >::VTKType << std::endl;
      }
      file << std::endl;
      return true;
   }


};

} // namespace TNL
