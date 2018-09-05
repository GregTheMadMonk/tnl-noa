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
#include <TNL/Communicators/NoDistrCommunicator.h>

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
template< typename CommunicatorType,
          typename MeshConfig,
          typename Device >
bool
loadMesh( const String& fileName,
          Mesh< MeshConfig, Device >& mesh,
          DistributedMeshes::DistributedMesh< Mesh< MeshConfig, Device > >& distributedMesh )
{
   if( CommunicatorType::isDistributed() )
   {
      std::cerr << "Distributed Mesh is not supported yet, only Distributed Grid is supported.";
      return false;
   }

   std::cout << "Loading mesh from file " << fileName << " ..." << std::endl;
   std::string fileName_( fileName.getString() );
   bool status = true;

   if( ends_with( fileName_, ".tnl" ) )
      mesh.load( fileName );
   else if( ends_with( fileName_, ".ng" ) ) {
      Readers::NetgenReader reader;
      status = reader.readMesh( fileName, mesh );
   }
   else if( ends_with( fileName_, ".vtk" ) ) {
      Readers::VTKReader reader;
      status = reader.readMesh( fileName, mesh );
   }
   else {
      std::cerr << "File '" << fileName << "' has unknown extension. Supported extensions are '.tnl', '.vtk' and '.ng'." << std::endl;
      return false;
   }

   if( ! status )
   {
      std::cerr << "I am not able to load the mesh from the file " << fileName << ". "
                   "Perhaps the mesh stored in the file is not supported by the mesh "
                   "passed to the loadMesh function? The mesh type is "
                << getType< decltype(mesh) >() << std::endl;
      return false;
   }
   return true;
}

template< typename Problem,
          typename MeshConfig,
          typename Device >
bool
decomposeMesh( const Config::ParameterContainer& parameters,
               const String& prefix,
               Mesh< MeshConfig, Device >& mesh,
               DistributedMeshes::DistributedMesh< Mesh< MeshConfig, Device > >& distributedMesh,
               Problem& problem )
{
   using CommunicatorType = typename Problem::CommunicatorType;
   if( CommunicatorType::isDistributed() )
   {
       std::cerr << "Distributed Mesh is not supported yet, only Distributed Grid is supported.";
       return false;
   }
   return true;
}

template< typename CommunicatorType,
          typename MeshConfig >
bool
loadMesh( const String& fileName,
          Mesh< MeshConfig, Devices::Cuda >& mesh,
          DistributedMeshes::DistributedMesh< Mesh< MeshConfig, Devices::Cuda > >& distributedMesh )
{
   if(CommunicatorType::isDistributed())
   {
       std::cerr << "Distributed Mesh is not supported yet, only Distributed Grid is supported.";
       return false;
   }

   Mesh< MeshConfig, Devices::Host > hostMesh;
   DistributedMeshes::DistributedMesh< Mesh< MeshConfig, Devices::Host > > hostDistributedMesh;
   if( ! loadMesh< CommunicatorType >( fileName, hostMesh, hostDistributedMesh ) )
      return false;
   mesh = hostMesh;
   // TODO
//   distributedMesh = hostDistributedMesh;
   return true;
}

// Specializations for grids
template< typename CommunicatorType,
          int Dimension,
          typename Real,
          typename Device,
          typename Index >
bool
loadMesh( const String& fileName,
          Grid< Dimension, Real, Device, Index >& mesh,
          DistributedMeshes::DistributedMesh< Grid< Dimension, Real, Device, Index > > &distributedMesh )
{

   if( CommunicatorType::isDistributed() )
   {
      std::cout << "Loading a global mesh from the file " << fileName << "...";
      Grid< Dimension, Real, Device, Index > globalGrid;
      try
      {
         globalGrid.load( fileName );
      }
      catch(...)
      {
         std::cerr << std::endl;
         std::cerr << "I am not able to load the global mesh from the file " << fileName << "." << std::endl;
         return false;
      }
      std::cout << " [ OK ] " << std::endl;
  
      typename Meshes::DistributedMeshes::DistributedMesh<Grid< Dimension, Real, Device, Index >>::SubdomainOverlapsType overlap;
      distributedMesh.template setGlobalGrid< CommunicatorType >( globalGrid );
      distributedMesh.setupGrid(mesh);
      return true;
   }
   else
   {
      std::cout << "Loading a mesh from the file " << fileName << "...";
      try
      {
         mesh.load( fileName );
      }
      catch(...)
      {
         std::cerr << std::endl;
         std::cerr << "I am not able to load the mesh from the file " << fileName << "." << std::endl;
         std::cerr << " You may create it with tools like tnl-grid-setup or tnl-mesh-convert." << std::endl;
         return false;
      }
      std::cout << " [ OK ] " << std::endl;
      return true;
    }
}

template< typename Problem,
          int Dimension,
          typename Real,
          typename Device,
          typename Index >
bool
decomposeMesh( const Config::ParameterContainer& parameters,
               const String& prefix,
               Grid< Dimension, Real, Device, Index >& mesh,
               DistributedMeshes::DistributedMesh< Grid< Dimension, Real, Device, Index > > &distributedMesh,
               Problem& problem )
{
   using GridType = Grid< Dimension, Real, Device, Index >;
   using DistributedGridType = DistributedMeshes::DistributedMesh< GridType >;
   using SubdomainOverlapsType = typename DistributedGridType::SubdomainOverlapsType;
   using CommunicatorType = typename Problem::CommunicatorType;

   if( CommunicatorType::isDistributed() )
   {
      SubdomainOverlapsType lower( 0 ), upper( 0 );
      distributedMesh.setOverlaps( lower, upper );
      distributedMesh.setupGrid( mesh );
      
      problem.getSubdomainOverlaps( parameters, prefix, mesh, lower, upper  );
      distributedMesh.setOverlaps( lower, upper );
      distributedMesh.setupGrid( mesh );
      return true;
   }
   else
      return true;
}

// convenient overload for non-distributed meshes
template< typename Mesh >
bool
loadMesh( const String& fileName, Mesh& mesh )
{
   using Communicator = TNL::Communicators::NoDistrCommunicator;
   using DistributedMesh = DistributedMeshes::DistributedMesh< Mesh>;
   DistributedMesh distributedMesh;
   return loadMesh< Communicator >( fileName, mesh, distributedMesh );
}

} // namespace Meshes
} // namespace TNL
