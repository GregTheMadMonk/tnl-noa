/***************************************************************************
                          MeshWriterVTKLegacy.h  -  description
                             -------------------
    begin                : Mar 20, 2014
    copyright            : (C) 2014 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

/***
 * Authors:
 * Oberhuber Tomas, tomas.oberhuber@fjfi.cvut.cz
 * Zabka Vitezslav, zabkav@gmail.com
 */

#pragma once

#include <fstream>
#include <istream>
#include <sstream>
#include <iomanip>

#include <TNL/Meshes/Topologies/MeshTriangleTopology.h>
#include <TNL/Meshes/Topologies/MeshQuadrilateralTopology.h>
#include <TNL/Meshes/Topologies/MeshTetrahedronTopology.h>
#include <TNL/Meshes/Topologies/MeshHexahedronTopology.h>
#include <TNL/Meshes/MeshEntity.h>

namespace TNL {
namespace Meshes {

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
struct MeshEntityVTKType{};

template< typename MeshConfig, typename Device > struct MeshEntityVTKType< MeshEntity< MeshConfig, Device, MeshTriangleTopology > >     { enum { VTKType = tnlVTKTriangle }; };
template< typename MeshConfig, typename Device > struct MeshEntityVTKType< MeshEntity< MeshConfig, Device, MeshQuadrilateralTopology > >{ enum { VTKType = tnlVTKQuad }; };
template< typename MeshConfig, typename Device > struct MeshEntityVTKType< MeshEntity< MeshConfig, Device, MeshTetrahedronTopology > >  { enum { VTKType = tnlVTKTetra }; };
template< typename MeshConfig, typename Device > struct MeshEntityVTKType< MeshEntity< MeshConfig, Device, MeshHexahedronTopology > >   { enum { VTKType = tnlVTKHexahedron }; };

class MeshWriterVTKLegacy
{
   public:

   template< typename MeshType >
   static bool write( const String& fileName,
                      MeshType& mesh,
                      bool verbose )
   {
      if( MeshType::dimension > 3 )
      {
         std::cerr << "You try to write mesh with " << MeshType::dimension
              << "dimension but VTK legacy format supports only 1D, 2D and 3D meshes." << std::endl;
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

      return writeMesh( outputFile, mesh, verbose );
   }

   template< typename MeshType >
   static bool writeMesh( std::ostream& file,
                          MeshType& mesh,
                          bool verbose )
   {
      typedef typename MeshType::MeshTraitsType::CellType CellType;
      file << "# vtk DataFile Version 2.0" << std::endl;
      file << "TNL Mesh" << std::endl;
      file << "ASCII" << std::endl;
      file << "DATASET UNSTRUCTURED_GRID" << std::endl;
      file << std::endl;
      file << "POINTS " << mesh.template getEntitiesCount< 0 >() << " double" << std::endl;
      for( int i = 0; i < mesh.template getEntitiesCount< 0 >(); i++ )
      {
         mesh.template getEntity< 0 >( i ).getPoint().write( file );
         for( int j = MeshType::dimension; j < 3; j++ )
            file << " 0.0";
         file << std::endl;
      }
      file << std::endl;
      file << "CELLS " << mesh.getCellsCount();
      long int listSize( 0 );
      for( int i = 0; i < mesh.getCellsCount(); i++ )
         listSize += mesh.getCell( i ).template getSubentitiesCount< 0 >() + 1;
      file << " " << listSize << std::endl;
      for( int i = 0; i < mesh.getCellsCount(); i++ )
      {
         int numberOfVertices = mesh.getCell( i ).template getSubentitiesCount< 0 >();
         file << numberOfVertices << " ";
         for( int j = 0; j < numberOfVertices - 1; j++ )
            file << mesh.getCell( i ).template getSubentityIndex< 0 >( j ) << " ";
         file << mesh.getCell( i ).template getSubentityIndex< 0 >( numberOfVertices - 1 ) << std::endl;
      }
      file << std::endl;
      file << "CELL_TYPES " <<  mesh.getCellsCount() << std::endl;
      for( int i = 0; i < mesh.getCellsCount(); i++ )
      {
         file << MeshEntityVTKType< CellType >::VTKType << std::endl;
      }
      file << std::endl;
      return true;
   }


};

} // namespace Meshes
} // namespace TNL
