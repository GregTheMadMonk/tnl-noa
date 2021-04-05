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
#include <TNL/Meshes/Readers/TNLReader.h>
#include <TNL/Meshes/Readers/NetgenReader.h>
#include <TNL/Meshes/Readers/VTKReader.h>
#include <TNL/Meshes/Readers/VTUReader.h>

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

   namespace fs = std::experimental::filesystem;
   std::string format = fileFormat;
   if( format == "auto" ) {
      format = fs::path(fileName).extension();
      if( format.length() > 0 )
         // remove dot from the extension
         format = format.substr(1);
   }

   // TODO: when TNLReader is gone, use the MeshReader type instead of a template parameter in the mesh type resolver (and remove static_casts in this function)
   if( format == "tnl" ) {
      Readers::TNLReader reader( fileName );
      if( ! reader.detectMesh() )
         return false;
      if( reader.getMeshType() == "Meshes::Grid" )
         return GridTypeResolver< ConfigTag, Device >::run( reader, functor );
      else if( reader.getMeshType() == "Meshes::Mesh" )
         return MeshTypeResolver< ConfigTag, Device >::run( reader, functor );
      else {
         std::cerr << "The mesh type " << reader.getMeshType() << " is not supported in the TNL reader." << std::endl;
         return false;
      }
   }
   else if( format == "ng" ) {
      // FIXME: The Netgen files don't store the real, global index and local index types.
      // The reader has some defaults, but they might be disabled by the BuildConfigTags - in
      // this case we should use the first enabled type.
      Readers::NetgenReader reader( fileName );
      reader.detectMesh();
      if( reader.getMeshType() == "Meshes::Mesh" )
         return MeshTypeResolver< ConfigTag, Device >::run( static_cast<Readers::MeshReader&>(reader), functor );
      else {
         std::cerr << "The mesh type " << reader.getMeshType() << " is not supported in the Netgen reader." << std::endl;
         return false;
      }
   }
   else if( format == "vtk" ) {
      // FIXME: The VTK files don't store the global and local index types.
      // The reader has some defaults, but they might be disabled by the BuildConfigTags - in
      // this case we should use the first enabled type.
      Readers::VTKReader reader( fileName );
      reader.detectMesh();
      if( reader.getMeshType() == "Meshes::Mesh" )
         return MeshTypeResolver< ConfigTag, Device >::run( static_cast<Readers::MeshReader&>(reader), functor );
      else {
         std::cerr << "The mesh type " << reader.getMeshType() << " is not supported in the VTK reader." << std::endl;
         return false;
      }
   }
   else if( format == "vtu" ) {
      // FIXME: The XML VTK files don't store the local index type.
      // The reader has some defaults, but they might be disabled by the BuildConfigTags - in
      // this case we should use the first enabled type.
      Readers::VTUReader reader( fileName );
      reader.detectMesh();
      if( reader.getMeshType() == "Meshes::Mesh" )
         return MeshTypeResolver< ConfigTag, Device >::run( static_cast<Readers::MeshReader&>(reader), functor );
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
      std::cerr << " Supported formats are 'tnl', 'vtk', 'vtu' and 'ng'." << std::endl;
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

template< typename MeshConfig,
          typename Device >
bool
loadMesh( Mesh< MeshConfig, Device >& mesh,
          const std::string& fileName,
          const std::string& fileFormat )
{
   std::cout << "Loading a mesh from the file " << fileName << " ..." << std::endl;

   namespace fs = std::experimental::filesystem;
   std::string format = fileFormat;
   if( format == "auto" ) {
      format = fs::path(fileName).extension();
      if( format.length() > 0 )
         // remove dot from the extension
         format = format.substr(1);
   }

   try {
      if( format == "tnl" )
         mesh.load( fileName );
      else if( format == "ng" ) {
         Readers::NetgenReader reader( fileName );
         reader.loadMesh( mesh );
      }
      else if( format == "vtk" ) {
         Readers::VTKReader reader( fileName );
         reader.loadMesh( mesh );
      }
      else if( format == "vtu" ) {
         Readers::VTUReader reader( fileName );
         reader.loadMesh( mesh );
      }
      else {
         if( fileFormat == "auto" )
            std::cerr << "File '" << fileName << "' has unsupported format (based on the file extension): " << format << ".";
         else
            std::cerr << "Unsupported fileFormat parameter: " << fileFormat << ".";
         std::cerr << " Supported formats are 'tnl', 'vtk', 'vtu' and 'ng'." << std::endl;
         return false;
      }
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

// overload for grids
template< int Dimension,
          typename Real,
          typename Device,
          typename Index >
bool
loadMesh( Grid< Dimension, Real, Device, Index >& grid,
          const std::string& fileName,
          const std::string& fileFormat )
{
   std::cout << "Loading a grid from the file " << fileName << "..." << std::endl;
   try {
      grid.load( fileName );
      return true;
   }
   catch(...) {
      std::cerr << "I am not able to load the grid from the file " << fileName << "." << std::endl;
      return false;
   }
}

} // namespace Meshes
} // namespace TNL
