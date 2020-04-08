/***************************************************************************
                          MeshResolver_impl.h  -  description
                             -------------------
    begin                : Nov 22, 2016
    copyright            : (C) 2016 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Meshes/TypeResolver/TypeResolver.h>
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
bool resolveMeshType( const String& fileName, Functor&& functor )
{
   std::cout << "Detecting mesh from file " << fileName << " ..." << std::endl;
   if( fileName.endsWith( ".tnl" ) ) {
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
   else if( fileName.endsWith( ".ng" ) ) {
      // FIXME: The Netgen files don't store the real, global index and local index types.
      // The reader has some defaults, but they might be disabled by the BuildConfigTags - in
      // this case we should use the first enabled type.
      Readers::NetgenReader reader( fileName );
      if( ! reader.detectMesh() )
         return false;
      if( reader.getMeshType() == "Meshes::Mesh" )
         return MeshTypeResolver< ConfigTag, Device >::run( reader, functor );
      else {
         std::cerr << "The mesh type " << reader.getMeshType() << " is not supported in the Netgen reader." << std::endl;
         return false;
      }
   }
   else if( fileName.endsWith( ".vtk" ) ) {
      // FIXME: The VTK files don't store the global and local index types.
      // The reader has some defaults, but they might be disabled by the BuildConfigTags - in
      // this case we should use the first enabled type.
      Readers::VTKReader reader( fileName );
      if( ! reader.detectMesh() )
         return false;
      if( reader.getMeshType() == "Meshes::Mesh" )
         return MeshTypeResolver< ConfigTag, Device >::run( reader, functor );
      else {
         std::cerr << "The mesh type " << reader.getMeshType() << " is not supported in the VTK reader." << std::endl;
         return false;
      }
   }
   else if( fileName.endsWith( ".vtu" ) ) {
      // FIXME: The XML VTK files don't store the local index type.
      // The reader has some defaults, but they might be disabled by the BuildConfigTags - in
      // this case we should use the first enabled type.
      Readers::VTUReader reader( fileName );
      if( ! reader.detectMesh() )
         return false;
      if( reader.getMeshType() == "Meshes::Mesh" )
         return MeshTypeResolver< ConfigTag, Device >::run( reader, functor );
      else {
         std::cerr << "The mesh type " << reader.getMeshType() << " is not supported in the VTK reader." << std::endl;
         return false;
      }
   }
   else {
      std::cerr << "File '" << fileName << "' has unknown extension. Supported extensions are '.tnl', '.vtk', '.vtu' and '.ng'." << std::endl;
      return false;
   }
}

template< typename ConfigTag,
          typename Device,
          typename Functor >
bool resolveAndLoadMesh( const String& fileName, Functor&& functor )
{
   auto wrapper = [&]( auto& reader, auto&& mesh ) -> bool
   {
      using MeshType = std::decay_t< decltype(mesh) >;
      std::cout << "Loading a mesh from the file " << fileName << " ..." << std::endl;
      if( ! reader.readMesh( mesh ) ) {
         std::cerr << "Failed to load the mesh from the file " << fileName << ". The mesh type is "
                   << getType< MeshType >() << std::endl;
         return false;
      }
      return functor( reader, std::forward<MeshType>(mesh) );
   };
   return resolveMeshType< ConfigTag, Device >( fileName, wrapper );
}

template< typename MeshConfig,
          typename Device >
bool
loadMesh( const String& fileName,
          Mesh< MeshConfig, Device >& mesh )
{
   std::cout << "Loading a mesh from the file " << fileName << " ..." << std::endl;
   bool status = true;

   if( fileName.endsWith( ".tnl" ) )
      mesh.load( fileName );
   else if( fileName.endsWith( ".ng" ) ) {
      Readers::NetgenReader reader( fileName );
      status = reader.detectMesh();
      if( status )
         status = reader.readMesh( mesh );
   }
   else if( fileName.endsWith( ".vtk" ) ) {
      Readers::VTKReader reader( fileName );
      status = reader.detectMesh();
      if( status )
         status = reader.readMesh( mesh );
   }
   else if( fileName.endsWith( ".vtu" ) ) {
      Readers::VTUReader reader( fileName );
      status = reader.detectMesh();
      if( status )
         status = reader.readMesh( mesh );
   }
   else {
      std::cerr << "File '" << fileName << "' has unknown extension. Supported extensions are '.tnl', '.vtk', '.vtu' and '.ng'." << std::endl;
      return false;
   }

   if( ! status )
   {
      std::cerr << "Failed to load the mesh from the file " << fileName << ". The mesh type is "
                << getType< decltype(mesh) >() << std::endl;
      return false;
   }
   return true;
}

template< typename MeshConfig >
bool
loadMesh( const String& fileName,
          Mesh< MeshConfig, Devices::Cuda >& mesh )
{
   Mesh< MeshConfig, Devices::Host > hostMesh;
   if( ! loadMesh( fileName, hostMesh ) )
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
loadMesh( const String& fileName,
          Grid< Dimension, Real, Device, Index >& grid )
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
