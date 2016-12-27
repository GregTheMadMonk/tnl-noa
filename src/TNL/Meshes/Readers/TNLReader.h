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
#include <TNL/Containers/List.h>
#include <TNL/Meshes/Readers/VTKEntityType.h>

namespace TNL {
namespace Meshes {
namespace Readers {

class TNLReader
{
public:
   bool
   detectMesh( const String& fileName )
   {
      this->reset();
      this->fileName = fileName;

      String objectType;
      if( ! getObjectType( fileName, objectType ) )
      {
         std::cerr << "I am not able to detect the mesh type from the file " << fileName << "." << std::endl;
         return EXIT_FAILURE;
      }

      Containers::List< String > parsedMeshType;
      if( ! parseObjectType( objectType, parsedMeshType ) )
      {
         std::cerr << "Unable to parse the mesh type " << meshType << "." << std::endl;
         return false;
      }

      meshType = parsedMeshType[ 0 ];
      if( meshType == "Meshes::Grid" ) {
         // save parts necessary to determine the mesh type
         meshDimension = worldDimension = std::atoi( parsedMeshType[ 1 ].getString() );
         realType = parsedMeshType[ 2 ];
         globalIndexType = localIndexType = idType = parsedMeshType[ 4 ];
         // populate entity types (not necessary for GridTypeResolver, but while we're at it...)
         if( meshDimension == 1 )
            cellVTKType = VTKEntityType::Line;
         else if( meshDimension == 2 )
            cellVTKType = VTKEntityType::Quad;
         else if( meshDimension == 3 )
            cellVTKType = VTKEntityType::Hexahedron;
      }
      // TODO: loading of unstructured meshes
      else {
         std::cerr << "The mesh type " << meshType << " is not supported (yet)." << std::endl;
         return false;
      }

      return true;
   }

   template< typename MeshType >
   static bool
   readMesh( const String& fileName, MeshType& mesh )
   {
      return mesh.load( fileName );
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

   VTKEntityType
   getCellVTKType() const
   {
      return cellVTKType;
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
 
   String
   getIdType() const
   {
      return idType;
   }
 
protected:
   String fileName;
   String meshType;
   int meshDimension = 0;
   int worldDimension = 0;
   VTKEntityType cellVTKType = VTKEntityType::Vertex;
   String realType;
   String globalIndexType;
   String localIndexType;
   String idType;

   void reset()
   {
      fileName = "";
      meshType = "";
      meshDimension = worldDimension = 0;
      cellVTKType = VTKEntityType::Vertex;
      realType = localIndexType = globalIndexType = idType = "";
   }
};

} // namespace Readers
} // namespace Meshes
} // namespace TNL
