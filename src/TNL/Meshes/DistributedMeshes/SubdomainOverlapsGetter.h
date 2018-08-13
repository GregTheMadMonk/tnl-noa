/***************************************************************************
                          SubdomainOverlapsGetter.h  -  description
                             -------------------
    begin                : Aug 13, 2018
    copyright            : (C) 2018 by oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Meshes/Grid.h>
#include <TNL/Meshes/Mesh.h>
#include <TNL/Meshes/DistributedMeshes/DistributedMesh.h>

namespace TNL {
   namespace Meshes {
      namespace DistributedMeshes {
      
template< typename Mesh,
          typename Communicator >
class SubdomainOverlapsGetter
{};

template< typename MeshConfig,
          typename Device,
          typename Communicator >
class SubdomainOverlapsGetter< Mesh< MeshConfig, Device >, Communicator >
{
   public:
      
      using MeshType = Mesh< MeshConfig, Device >;
      using DeviceType = Device;
      using IndexType = typename MeshType::IndexType;
      using DistributedMeshType = DistributedMesh< MeshType >;
      using SubdomainOverlapsType = typename DistributedMeshType::SubdomainOverlapsType;
      using CommunicatorType = Communicator;
      
      // Computes subdomain overlaps
      // TODO: Its gonna be very likely different for Mesh than for Grid
      static void getOverlaps( const MeshType& mesh,
                               SubdomainOverlapsType& lower,
                               SubdomainOverlapsType& upper,
                               IndexType subdomainOverlapSize );
};

template< int Dimension,
          typename Real,
          typename Device,
          typename Index,
          typename Communicator >
class SubdomainOverlapsGetter< Grid< Dimension, Real, Device, Index >, Communicator >
{
   public:
      
      using MeshType = Grid< Dimension, Real, Device, Index >;
      using DeviceType = Device;
      using IndexType = Index;
      using DistributedMeshType = DistributedMesh< MeshType >;
      using SubdomainOverlapsType = typename DistributedMeshType::SubdomainOverlapsType;
      using CoordinatesType = typename DistributedMeshType::CoordinatesType;
      using CommunicatorType = Communicator;
      
      // Computes subdomain overlaps
      /* SubdomainOverlapsType is a touple of the same size as the mesh dimensions. 
       * lower.x() is overlap of the subdomain at boundary where x = 0,
       * lower.y() is overlap of the subdomain at boundary where y = 0 etc.
       * upper.x() is overlap of the subdomain at boundary where x = grid.getDimensions().x() - 1,
       * upper.y() is overlap of the subdomain at boundary where y = grid.getDimensions().y() - 1 etc.
       */
      static void getOverlaps( const MeshType& mesh,
                               SubdomainOverlapsType& lower,
                               SubdomainOverlapsType& upper,
                               IndexType subdomainOverlapSize );
   
};

      } // namespace DistributedMeshes
   } // namespace Meshes
} // namespace TNL

#include <TNL/Meshes/DistributedMeshes/SubdomainOverlapsGetter.hpp>
