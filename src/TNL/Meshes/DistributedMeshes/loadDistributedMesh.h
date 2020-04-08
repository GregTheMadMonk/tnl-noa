/***************************************************************************
                          loadDistributedMesh.h  -  description
                             -------------------
    begin                : Apr 9, 2020
    copyright            : (C) 2020 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Meshes/Mesh.h>
#include <TNL/Meshes/DistributedMeshes/DistributedMesh.h>

namespace TNL {
namespace Meshes {

template< typename CommunicatorType,
          typename MeshConfig,
          typename Device >
bool
loadDistributedMesh( const String& fileName,
                     Mesh< MeshConfig, Device >& mesh,
                     DistributedMeshes::DistributedMesh< Mesh< MeshConfig, Device > >& distributedMesh )
{
   std::cerr << "Distributed Mesh is not supported yet, only Distributed Grid is supported.";
   return false;
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
   std::cerr << "Distributed Mesh is not supported yet, only Distributed Grid is supported.";
   return false;
}

// overloads for grids
template< typename CommunicatorType,
          int Dimension,
          typename Real,
          typename Device,
          typename Index >
bool
loadDistributedMesh( const String& fileName,
                     Grid< Dimension, Real, Device, Index >& mesh,
                     DistributedMeshes::DistributedMesh< Grid< Dimension, Real, Device, Index > > &distributedMesh )
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

   SubdomainOverlapsType lower( 0 ), upper( 0 );
   distributedMesh.setOverlaps( lower, upper );
   distributedMesh.setupGrid( mesh );

   problem.getSubdomainOverlaps( parameters, prefix, mesh, lower, upper  );
   distributedMesh.setOverlaps( lower, upper );
   distributedMesh.setupGrid( mesh );

   return true;
}


} // namespace Meshes
} // namespace TNL
