/***************************************************************************
                          TNL.h  -  description
                             -------------------
    begin                : Nov 20, 2016
    copyright            : (C) 2016 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/String.h>
#include <TNL/Object.h>
#include <TNL/Meshes/VTKTraits.h>

namespace TNL {
namespace Meshes {
namespace Readers {

class TNLReader
{
public:
   TNLReader() = delete;

   TNLReader( const String& fileName )
   : fileName( fileName )
   {}

   bool
   detectMesh()
   {
      this->reset();

      const String objectType = getObjectType( fileName );
      const std::vector< String > parsedMeshType = parseObjectType( objectType );
      if( ! parsedMeshType.size() ) {
         std::cerr << "Unable to parse the mesh type " << meshType << "." << std::endl;
         return false;
      }

      meshType = parsedMeshType[ 0 ];
      if( meshType == "Meshes::Grid" ) {
         // save parts necessary to determine the mesh type
         meshDimension = worldDimension = std::atoi( parsedMeshType[ 1 ].getString() );
         realType = parsedMeshType[ 2 ];
         globalIndexType = localIndexType = parsedMeshType[ 4 ];
         // populate entity types (not necessary for GridTypeResolver, but while we're at it...)
         if( meshDimension == 1 )
            cellShape = VTK::EntityShape::Line;
         else if( meshDimension == 2 )
            cellShape = VTK::EntityShape::Quad;
         else if( meshDimension == 3 )
            cellShape = VTK::EntityShape::Hexahedron;
      }
      else if( meshType == "Meshes::Mesh" ) {
         const std::vector< String > parsedMeshConfig = parseObjectType( parsedMeshType[ 1 ] );
         if( ! parsedMeshConfig.size() ) {
            std::cerr << "Unable to parse the mesh config type " << parsedMeshType[ 1 ] << "." << std::endl;
            return false;
         }
         if( parsedMeshConfig.size() != 7 ) {
            std::cerr << "The parsed mesh config type has wrong size (expected 7 elements):" << std::endl;
            std::cerr << "[ ";
            for( std::size_t i = 0; i < parsedMeshConfig.size() - 1; i++ )
               std::cerr << parsedMeshConfig[ i ] << ", ";
            std::cerr << parsedMeshConfig.back() << " ]" << std::endl;
            return false;
         }

         // save parts necessary to determine the mesh type
         const String topology = parsedMeshConfig[ 1 ];
         worldDimension = std::atoi( parsedMeshConfig[ 2 ].getString() );
         realType = parsedMeshConfig[ 3 ];
         globalIndexType = parsedMeshConfig[ 4 ];
         localIndexType = parsedMeshConfig[ 5 ];

         if( topology == "MeshEdgeTopology" )
            cellShape = VTK::EntityShape::Line;
         else if( topology == "MeshTriangleTopology" )
            cellShape = VTK::EntityShape::Triangle;
         else if( topology == "MeshQuadrilateralTopology" )
            cellShape = VTK::EntityShape::Quad;
         else if( topology == "MeshTetrahedronTopology" )
            cellShape = VTK::EntityShape::Tetra;
         else if( topology == "MeshHexahedronTopology" )
            cellShape = VTK::EntityShape::Hexahedron;
         else {
            std::cerr << "Detected topology '" << topology << "' is not supported." << std::endl;
            return false;
         }
      }
      else {
         std::cerr << "The mesh type " << meshType << " is not supported (yet)." << std::endl;
         return false;
      }

      return true;
   }

   template< typename MeshType >
   bool
   readMesh( MeshType& mesh )
   {
      mesh.load( fileName );
      return true;
   }

   String
   getMeshType() const
   {
      return meshType;
   }

   int
   getMeshDimension() const
   {
      return meshDimension;
   }

   int
   getWorldDimension() const
   {
      return worldDimension;
   }

   VTK::EntityShape
   getCellShape() const
   {
      return cellShape;
   }
 
   String
   getRealType() const
   {
      return realType;
   }

   String
   getGlobalIndexType() const
   {
      return globalIndexType;
   }
 
   String
   getLocalIndexType() const
   {
      return localIndexType;
   }
 
protected:
   String fileName;
   String meshType;
   int meshDimension = 0;
   int worldDimension = 0;
   VTK::EntityShape cellShape = VTK::EntityShape::Vertex;
   String realType;
   String globalIndexType;
   String localIndexType;

   void reset()
   {
      meshType = "";
      meshDimension = worldDimension = 0;
      cellShape = VTK::EntityShape::Vertex;
      realType = localIndexType = globalIndexType = "";
   }
};

} // namespace Readers
} // namespace Meshes
} // namespace TNL
