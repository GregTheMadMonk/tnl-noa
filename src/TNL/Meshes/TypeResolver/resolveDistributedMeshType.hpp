/***************************************************************************
                          resolveDistributedMesh.hpp  -  description
                             -------------------
    begin                : Apr 9, 2020
    copyright            : (C) 2020 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <experimental/filesystem>

#include <TNL/Meshes/TypeResolver/resolveDistributedMeshType.h>
#include <TNL/Meshes/TypeResolver/MeshTypeResolver.h>
#include <TNL/Meshes/Readers/PVTUReader.h>

namespace TNL {
namespace Meshes {

template< typename ConfigTag,
          typename Device,
          typename Functor >
bool
resolveDistributedMeshType( Functor&& functor,
                            const std::string& fileName,
                            const std::string& fileFormat )
{
   std::cout << "Detecting distributed mesh from file " << fileName << " ..." << std::endl;

   auto wrapper = [&functor] ( Readers::MeshReader& reader, auto&& localMesh )
   {
      using LocalMesh = std::decay_t< decltype(localMesh) >;
      using DistributedMesh = DistributedMeshes::DistributedMesh< LocalMesh >;
      return std::forward<Functor>(functor)( reader, DistributedMesh{ std::move(localMesh) } );
   };

   namespace fs = std::experimental::filesystem;
   std::string format = fileFormat;
   if( format == "auto" ) {
      format = fs::path(fileName).extension();
      if( format.length() > 0 )
         // remove dot from the extension
         format = format.substr(1);
   }

   if( format == "pvtu" ) {
      // FIXME: The XML VTK files don't store the local index type.
      // The reader has some defaults, but they might be disabled by the BuildConfigTags - in
      // this case we should use the first enabled type.
      Readers::PVTUReader reader( fileName );
      reader.detectMesh();
      if( reader.getMeshType() == "Meshes::DistributedMesh" ) {
         return MeshTypeResolver< ConfigTag, Device >::run( static_cast<Readers::MeshReader&>(reader), wrapper );
      }
      else {
         std::cerr << "The mesh type " << reader.getMeshType() << " is not supported in the VTK reader." << std::endl;
         return false;
      }
   }
   else {
      if( fileFormat == "auto" )
         std::cerr << "File '" << fileName << "' has unsupported format (based on the file extension): " << format << ".";
      else
         std::cerr << "Unsupported fileFormat parameter: " << fileFormat << ".";
      std::cerr << " Supported formats are 'pvtu'." << std::endl;
      return false;
   }
}

template< typename ConfigTag,
          typename Device,
          typename Functor >
bool
resolveAndLoadDistributedMesh( Functor&& functor,
                               const std::string& fileName,
                               const std::string& fileFormat )
{
   auto wrapper = [&]( Readers::MeshReader& reader, auto&& mesh ) -> bool
   {
      using MeshType = std::decay_t< decltype(mesh) >;
      std::cout << "Loading a mesh from the file " << fileName << " ..." << std::endl;
      try {
         dynamic_cast<Readers::PVTUReader&>(reader).loadMesh( mesh );
      }
      catch( const Meshes::Readers::MeshReaderError& e ) {
         std::cerr << "Failed to load the mesh from the file " << fileName << ". The error is:\n" << e.what() << std::endl;
         return false;
      }
      return functor( reader, std::forward<MeshType>(mesh) );
   };
   return resolveDistributedMeshType< ConfigTag, Device >( wrapper, fileName, fileFormat );
}

template< typename MeshConfig,
          typename Device >
bool
loadDistributedMesh( Mesh< MeshConfig, Device >& mesh,
                     DistributedMeshes::DistributedMesh< Mesh< MeshConfig, Device > >& distributedMesh,
                     const std::string& fileName,
                     const std::string& fileFormat )
{
   // TODO: simplify interface, pass only the distributed mesh
   TNL_ASSERT_EQ( &mesh, &distributedMesh.getLocalMesh(), "mesh is not local mesh of the distributed mesh" );

   namespace fs = std::experimental::filesystem;
   std::string format = fileFormat;
   if( format == "auto" ) {
      format = fs::path(fileName).extension();
      if( format.length() > 0 )
         // remove dot from the extension
         format = format.substr(1);
   }

   if( format == "pvtu" ) {
      Readers::PVTUReader reader( fileName );
      reader.loadMesh( distributedMesh );
      return true;
   }
   else {
      if( fileFormat == "auto" )
         std::cerr << "File '" << fileName << "' has unsupported format (based on the file extension): " << format << ".";
      else
         std::cerr << "Unsupported fileFormat parameter: " << fileFormat << ".";
      std::cerr << " Supported formats are 'pvtu'." << std::endl;
      return false;
   }
}

// overloads for grids
template< int Dimension,
          typename Real,
          typename Device,
          typename Index >
bool
loadDistributedMesh( Grid< Dimension, Real, Device, Index >& mesh,
                     DistributedMeshes::DistributedMesh< Grid< Dimension, Real, Device, Index > > &distributedMesh,
                     const std::string& fileName,
                     const std::string& fileFormat )
{
   // TODO: implement a PVTI reader
   std::cerr << "Loading a distributed mesh from a " << fileFormat << " file is not implemented yet." << std::endl;
   return false;
}

} // namespace Meshes
} // namespace TNL
