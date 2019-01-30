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
      static void getOverlaps( const DistributedMeshType* distributedMesh,
                               SubdomainOverlapsType& lower,
                               SubdomainOverlapsType& upper,
                               IndexType subdomainOverlapSize,
                               const SubdomainOverlapsType& periodicBoundariesOverlapSize = 0 );
};

// TODO: Specializations by the grid dimension can be avoided when the MPI directions are 
// rewritten in a dimension independent way

template< typename Real,
          typename Device,
          typename Index,
          typename Communicator >
class SubdomainOverlapsGetter< Grid< 1, Real, Device, Index >, Communicator >
{
   public:
      
      static const int Dimension = 1;
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
       * upper.x() is overlap of the subdomain at boundary where x = grid.getDimensions().x() - 1,
       */
      static void getOverlaps( const DistributedMeshType* distributedMesh,
                               SubdomainOverlapsType& lower,
                               SubdomainOverlapsType& upper,
                               IndexType subdomainOverlapSize,
                               const SubdomainOverlapsType& periodicBoundariesOverlapSize = 0 );
   
};


template< typename Real,
          typename Device,
          typename Index,
          typename Communicator >
class SubdomainOverlapsGetter< Grid< 2, Real, Device, Index >, Communicator >
{
   public:
      
      static const int Dimension = 2;
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
       * lower.y() is overlap of the subdomain at boundary where y = 0,
       * upper.x() is overlap of the subdomain at boundary where x = grid.getDimensions().x() - 1,
       * upper.y() is overlap of the subdomain at boundary where y = grid.getDimensions().y() - 1.
       */
      static void getOverlaps( const DistributedMeshType* distributedMesh,
                               SubdomainOverlapsType& lower,
                               SubdomainOverlapsType& upper,
                               IndexType subdomainOverlapSize,
                               const SubdomainOverlapsType& periodicBoundariesOverlapSize = 0 );
   
};

template< typename Real,
          typename Device,
          typename Index,
          typename Communicator >
class SubdomainOverlapsGetter< Grid< 3, Real, Device, Index >, Communicator >
{
   public:
      
      static const int Dimension = 3;
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
       * lower.y() is overlap of the subdomain at boundary where y = 0,
       * lower.z() is overlap of the subdomain at boundary where z = 0,
       * upper.x() is overlap of the subdomain at boundary where x = grid.getDimensions().x() - 1,
       * upper.y() is overlap of the subdomain at boundary where y = grid.getDimensions().y() - 1,
       * upper.z() is overlap of the subdomain at boundary where z = grid.getDimensions().z() - 1,
       */
      static void getOverlaps( const DistributedMeshType* distributedMesh,
                               SubdomainOverlapsType& lower,
                               SubdomainOverlapsType& upper,
                               IndexType subdomainOverlapSize,
                               const SubdomainOverlapsType& periodicBoundariesOverlapSize = 0 );
   
};


      } // namespace DistributedMeshes
   } // namespace Meshes
} // namespace TNL

#include <TNL/Meshes/DistributedMeshes/SubdomainOverlapsGetter.hpp>
