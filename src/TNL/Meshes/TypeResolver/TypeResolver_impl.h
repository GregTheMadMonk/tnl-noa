/***************************************************************************
                          MeshResolver_impl.h  -  description
                             -------------------
    begin                : Nov 22, 2016
    copyright            : (C) 2016 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <string>
#include <utility>

#include <TNL/Meshes/TypeResolver/TypeResolver.h>
#include <TNL/Meshes/Readers/TNLReader.h>
#include <TNL/Meshes/Readers/NetgenReader.h>
#include <TNL/Meshes/Readers/VTKReader.h>
#include <TNL/Meshes/TypeResolver/GridTypeResolver.h>
#include <TNL/Meshes/TypeResolver/MeshTypeResolver.h>

// TODO: implement this in TNL::String
inline bool ends_with( const std::string& value, const std::string& ending )
{
   if (ending.size() > value.size())
      return false;
   return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

namespace TNL {
namespace Meshes {

/*
 * TODO:
 * The variadic template parameter pack ProblemSetterArgs will not be necessary
 * in C++14 as it will be possible to use generic lambda functions to pass
 * parameters to the ProblemSetter:
 *
 *    // wrapper for MeshTypeResolver
 *    template< typename MeshType >
 *    using ProblemSetterWrapper = ProblemSetter< Real, Device, Index, MeshType, ConfigTag, SolverStarter< ConfigTag > >;
 *
 *    bool run( const Config::ParameterContainer& parameters )
 *    {
 *       const String& meshFileName = parameters.getParameter< String >( "mesh" );
 *       auto wrapper = []( auto&& mesh ) {
 *           return ProblemSetterWrapper< decltype(mesh) >::run( parameters );
 *       };
 *       return MeshTypeResolver< ConfigTag, Device, wrapper >::run( meshFileName );
 *    }
 */
template< typename ConfigTag,
          typename Device,
          template< typename MeshType > class ProblemSetter,
          typename... ProblemSetterArgs >
bool resolveMeshType( const String& fileName_,
                      ProblemSetterArgs&&... problemSetterArgs )
{
   std::cout << "Detecting mesh from file " << fileName_ << " ..." << std::endl;
   std::string fileName( fileName_.getString() );
   if( ends_with( fileName, ".tnl" ) ) {
      Readers::TNLReader reader;
      if( ! reader.detectMesh( fileName_ ) )
         return false;
      if( reader.getMeshType() == "Meshes::Grid" )
         return GridTypeResolver< decltype(reader), ConfigTag, Device, ProblemSetter, ProblemSetterArgs... >::
            run( reader, std::forward<ProblemSetterArgs>(problemSetterArgs)... );
      else if( reader.getMeshType() == "Meshes::Mesh" )
         return MeshTypeResolver< decltype(reader), ConfigTag, Device, ProblemSetter, ProblemSetterArgs... >::
            run( reader, std::forward<ProblemSetterArgs>(problemSetterArgs)... );
      else {
         std::cerr << "The mesh type " << reader.getMeshType() << " is not supported in the TNL reader." << std::endl;
         return false;
      }
   }
   else if( ends_with( fileName, ".ng" ) ) {
      // FIXME: The Netgen files don't store the real, global index, local index and id types.
      // The reader has some defaults, but they might be disabled by the BuildConfigTags - in
      // this case we should use the first enabled type.
      Readers::NetgenReader reader;
      if( ! reader.detectMesh( fileName_ ) )
         return false;
      if( reader.getMeshType() == "Meshes::Mesh" )
         return MeshTypeResolver< decltype(reader), ConfigTag, Device, ProblemSetter, ProblemSetterArgs... >::
            run( reader, std::forward<ProblemSetterArgs>(problemSetterArgs)... );
      else {
         std::cerr << "The mesh type " << reader.getMeshType() << " is not supported in the Netgen reader." << std::endl;
         return false;
      }
   }
   else if( ends_with( fileName, ".vtk" ) ) {
      // FIXME: The VTK files don't store the global index, local index and id types.
      // The reader has some defaults, but they might be disabled by the BuildConfigTags - in
      // this case we should use the first enabled type.
      Readers::VTKReader reader;
      if( ! reader.detectMesh( fileName_ ) )
         return false;
      if( reader.getMeshType() == "Meshes::Mesh" )
         return MeshTypeResolver< decltype(reader), ConfigTag, Device, ProblemSetter, ProblemSetterArgs... >::
            run( reader, std::forward<ProblemSetterArgs>(problemSetterArgs)... );
      else {
         std::cerr << "The mesh type " << reader.getMeshType() << " is not supported in the VTK reader." << std::endl;
         return false;
      }
   }
   else {
      std::cerr << "File '" << fileName << "' has unknown extension. Supported extensions are '.tnl', '.vtk' and '.ng'." << std::endl;
      return false;
   }
}

// TODO: reorganize
template< typename MeshConfig, typename Device >
bool
loadMesh( const String& fileName_, Mesh< MeshConfig, Device >& mesh )
{
   std::cout << "Loading mesh from file " << fileName_ << " ..." << std::endl;
   std::string fileName( fileName_.getString() );
   bool status = true;

   if( ends_with( fileName, ".tnl" ) )
      status = mesh.load( fileName_ );
   else if( ends_with( fileName, ".ng" ) ) {
      Readers::NetgenReader reader;
      status = reader.readMesh( fileName_, mesh );
   }
   else if( ends_with( fileName, ".vtk" ) ) {
      Readers::VTKReader reader;
      status = reader.readMesh( fileName_, mesh );
   }
   else {
      std::cerr << "File '" << fileName << "' has unknown extension. Supported extensions are '.tnl', '.vtk' and '.ng'." << std::endl;
      return false;
   }

   if( ! status )
   {
      std::cerr << "I am not able to load the mesh from the file " << fileName_ << ". "
                   "Perhaps the mesh stored in the file is not supported by the mesh "
                   "passed to the loadMesh function? The mesh type is "
                << mesh.getType() << std::endl;
      return false;
   }
   return true;
}

template< typename MeshConfig >
bool
loadMesh( const String& fileName, Mesh< MeshConfig, Devices::Cuda >& mesh )
{
   Mesh< MeshConfig, Devices::Host > hostMesh;
   if( ! loadMesh( fileName, hostMesh ) )
      return false;
   mesh = hostMesh;
   return true;
}

template< int Dimension, typename Real, typename Device, typename Index >
bool
loadMesh( const String& fileName, Grid< Dimension, Real, Device, Index >& mesh )
{
   std::cout << "Loading mesh from file " << fileName << " ..." << std::endl;
   if( ! mesh.load( fileName ) )
   {
      std::cerr << "I am not able to load the grid from the file " << fileName << ". "
                   "You may create it with tools like tnl-grid-setup."
                << std::endl;
      return false;
   }
   return true;
}

} // namespace Meshes
} // namespace TNL
