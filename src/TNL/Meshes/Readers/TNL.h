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

namespace TNL {
namespace Meshes {
namespace Readers {

class TNL
{
public:
   bool
   readFile( const String& fileName )
   {
      this->reset();

      String meshType;
      if( ! getObjectType( fileName, meshType ) )
      {
         std::cerr << "I am not able to detect the mesh type from the file " << fileName << "." << std::endl;
         return EXIT_FAILURE;
      }

      Containers::List< String > parsedMeshType;
      if( ! parseObjectType( meshType, parsedMeshType ) )
      {
         std::cerr << "Unable to parse the mesh type " << meshType << "." << std::endl;
         return false;
      }

      meshType = parsedMeshType[ 0 ];
      if( meshType == "Meshes::Grid" ) {
         // save parts necessary to determine the mesh type
         meshDimension = std::atoi( parsedMeshType[ 1 ].getString() );
         realType = parsedMeshType[ 2 ];
         indexType = parsedMeshType[ 4 ];
      }
      // TODO: loading of unstructured meshes
      else {
         std::cerr << "The mesh type " << meshType << " is not supported (yet)." << std::endl;
         return false;
      }

      return true;
   }

   template< typename MeshType >
   bool
   initializeMesh( MeshType& mesh )
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

   String
   getRealType() const
   {
      return realType;
   }

   String
   getIndexType() const
   {
      return indexType;
   }
 
protected:
   String fileName;
   String meshType;
   int meshDimension = 0;
   String realType;
   String indexType;

   void reset()
   {
      fileName = "";
      meshType = "";
      meshDimension = 0;
      realType = "";
      indexType = "";
   }
};

} // namespace Readers
} // namespace Meshes
} // namespace TNL
