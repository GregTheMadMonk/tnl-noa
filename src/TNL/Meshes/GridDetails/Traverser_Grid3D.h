/***************************************************************************
                          Traverser_Grid3D.h  -  description
                             -------------------
    begin                : Jul 29, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Meshes/Traverser.h>
#include <TNL/SharedPointer.h>

namespace TNL {
namespace Meshes {

template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
class Traverser< Meshes::Grid< 3, Real, Device, Index >, GridEntity, 3 >
{
   public:
      using GridType = Meshes::Grid< 3, Real, Device, Index >;
      using GridPointer = SharedPointer< GridType >;
      using CoordinatesType = typename GridType::CoordinatesType;
      using DistributedGridType = Meshes::DistributedMeshes::DistributedMesh< GridType >;
      using SubdomainOverlapsType = typename DistributedGridType::SubdomainOverlapsType;

      template< typename UserData,
                typename EntitiesProcessor >
      void processBoundaryEntities( const GridPointer& gridPointer,
                                    SharedPointer< UserData, Device >& userDataPointer ) const;

      template< typename UserData,
                typename EntitiesProcessor >
      void processInteriorEntities( const GridPointer& gridPointer,
                                    SharedPointer< UserData, Device >& userDataPointer ) const;
      template< typename UserData,
                typename EntitiesProcessor >
      void processAllEntities( const GridPointer& gridPointer,
                               SharedPointer< UserData, Device >& userDataPointer ) const;
 
};

template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
class Traverser< Meshes::Grid< 3, Real, Device, Index >, GridEntity, 2 >
{
   public:
      using GridType = Meshes::Grid< 3, Real, Device, Index >;
      using GridPointer = SharedPointer< GridType >;
      using CoordinatesType = typename GridType::CoordinatesType;
      using DistributedGridType = Meshes::DistributedMeshes::DistributedMesh< GridType >;
      using SubdomainOverlapsType = typename DistributedGridType::SubdomainOverlapsType;

      template< typename UserData,
                typename EntitiesProcessor >
      void processBoundaryEntities( const GridPointer& gridPointer,
                                    SharedPointer< UserData, Device >& userDataPointer ) const;

      template< typename UserData,
                typename EntitiesProcessor >
      void processInteriorEntities( const GridPointer& gridPointer,
                                    SharedPointer< UserData, Device >& userDataPointer ) const;
      template< typename UserData,
                typename EntitiesProcessor >
      void processAllEntities( const GridPointer& gridPointer,
                               SharedPointer< UserData, Device >& userDataPointer ) const;
 
};

template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
class Traverser< Meshes::Grid< 3, Real, Device, Index >, GridEntity, 1 >
{
   public:
      using GridType = Meshes::Grid< 3, Real, Device, Index >;
      using GridPointer = SharedPointer< GridType >;
      using CoordinatesType = typename GridType::CoordinatesType;
      using DistributedGridType = Meshes::DistributedMeshes::DistributedMesh< GridType >;
      using SubdomainOverlapsType = typename DistributedGridType::SubdomainOverlapsType;

      template< typename UserData,
                typename EntitiesProcessor >
      void processBoundaryEntities( const GridPointer& gridPointer,
                                    SharedPointer< UserData, Device >& userDataPointer ) const;

      template< typename UserData,
                typename EntitiesProcessor >
      void processInteriorEntities( const GridPointer& gridPointer,
                                    SharedPointer< UserData, Device >& userDataPointer ) const;

      template< typename UserData,
                typename EntitiesProcessor >
      void processAllEntities( const GridPointer& gridPointer,
                               SharedPointer< UserData, Device >& userDataPointer ) const;
 
};

template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
class Traverser< Meshes::Grid< 3, Real, Device, Index >, GridEntity, 0 >
{
   public:
      using GridType = Meshes::Grid< 3, Real, Device, Index >;
      using GridPointer = SharedPointer< GridType >;
      using CoordinatesType = typename GridType::CoordinatesType;
      using DistributedGridType = Meshes::DistributedMeshes::DistributedMesh< GridType >;
      using SubdomainOverlapsType = typename DistributedGridType::SubdomainOverlapsType;

      template< typename UserData,
                typename EntitiesProcessor >
      void processBoundaryEntities( const GridPointer& gridPointer,
                                    SharedPointer< UserData, Device >& userDataPointer ) const;

      template< typename UserData,
                typename EntitiesProcessor >
      void processInteriorEntities( const GridPointer& gridPointer,
                                    SharedPointer< UserData, Device >& userDataPointer ) const;
 
      template< typename UserData,
                typename EntitiesProcessor >
      void processAllEntities( const GridPointer& gridPointer,
                               SharedPointer< UserData, Device >& userDataPointer ) const;
};

} // namespace Meshes
} // namespace TNL

#include <TNL/Meshes/GridDetails/Traverser_Grid3D_impl.h>
