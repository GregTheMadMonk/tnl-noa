/***************************************************************************
                          resolveMeshType.hpp  -  description
                             -------------------
    begin                : Nov 22, 2016
    copyright            : (C) 2016 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <experimental/filesystem>

#include <TNL/Meshes/TypeResolver/resolveMeshType.h>
#include <TNL/Meshes/TypeResolver/GridTypeResolver.h>
#include <TNL/Meshes/TypeResolver/MeshTypeResolver.h>
#include <TNL/Meshes/Readers/getMeshReader.h>

namespace TNL {
namespace Meshes {

template< typename ConfigTag,
          typename Device,
          typename Functor >
bool
resolveMeshType( Functor&& functor,
                 const std::string& fileName,
                 const std::string& fileFormat )
{
   std::cout << "Detecting mesh from file " << fileName << " ..." << std::endl;

   std::shared_ptr< Readers::MeshReader > reader = Readers::getMeshReader( fileName, fileFormat );
   if( reader == nullptr )
      return false;

   reader->detectMesh();

   if( reader->getMeshType() == "Meshes::Grid" || reader->getMeshType() == "Meshes::DistributedGrid" )
      return GridTypeResolver< ConfigTag, Device >::run( *reader, functor );
   else if( reader->getMeshType() == "Meshes::Mesh" || reader->getMeshType() == "Meshes::DistributedMesh" )
      return MeshTypeResolver< ConfigTag, Device >::run( *reader, functor );
   else {
      std::cerr << "The mesh type " << reader->getMeshType() << " is not supported." << std::endl;
      return false;
   }
}

template< typename ConfigTag,
          typename Device,
          typename Functor >
bool
resolveAndLoadMesh( Functor&& functor,
                    const std::string& fileName,
                    const std::string& fileFormat )
{
   auto wrapper = [&]( auto& reader, auto&& mesh ) -> bool
   {
      using MeshType = std::decay_t< decltype(mesh) >;
      std::cout << "Loading a mesh from the file " << fileName << " ..." << std::endl;
      try {
         reader.loadMesh( mesh );
      }
      catch( const Meshes::Readers::MeshReaderError& e ) {
         std::cerr << "Failed to load the mesh from the file " << fileName << ". The error is:\n" << e.what() << std::endl;
         return false;
      }
      return functor( reader, std::forward<MeshType>(mesh) );
   };
   return resolveMeshType< ConfigTag, Device >( wrapper, fileName, fileFormat );
}

template< typename Mesh >
bool
loadMesh( Mesh& mesh,
          const std::string& fileName,
          const std::string& fileFormat )
{
   std::cout << "Loading a mesh from the file " << fileName << " ..." << std::endl;

   std::shared_ptr< Readers::MeshReader > reader = Readers::getMeshReader( fileName, fileFormat );
   if( reader == nullptr )
      return false;

   try {
      reader->loadMesh( mesh );
   }
   catch( const Meshes::Readers::MeshReaderError& e ) {
      std::cerr << "Failed to load the mesh from the file " << fileName << ". The error is:\n" << e.what() << std::endl;
      return false;
   }

   return true;
}

template< typename MeshConfig >
bool
loadMesh( Mesh< MeshConfig, Devices::Cuda >& mesh,
          const std::string& fileName,
          const std::string& fileFormat )
{
   Mesh< MeshConfig, Devices::Host > hostMesh;
   if( ! loadMesh( hostMesh, fileName, fileFormat ) )
      return false;
   mesh = hostMesh;
   return true;
}

} // namespace Meshes
} // namespace TNL
