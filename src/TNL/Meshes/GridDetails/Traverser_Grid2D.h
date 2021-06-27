/***************************************************************************
                          Traverser_Grid2D.h  -  description
                             -------------------
    begin                : Jul 29, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Meshes/Traverser.h>
#include <TNL/Pointers/SharedPointer.h>

namespace TNL {
namespace Meshes {

template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
class Traverser< Meshes::Grid< 2, Real, Device, Index >, GridEntity, 2 >
{
   public:
      using GridType = Meshes::Grid< 2, Real, Device, Index >;
      using GridPointer = Pointers::SharedPointer< GridType >;
      using CoordinatesType = typename GridType::CoordinatesType;

      template< typename EntitiesProcessor,
                typename UserData >
      void processBoundaryEntities( const GridPointer& gridPointer,
                                    UserData& userData ) const;

      template< typename EntitiesProcessor,
                typename UserData >
      void processInteriorEntities( const GridPointer& gridPointer,
                                    UserData& userData ) const;
      template< typename EntitiesProcessor,
                typename UserData >
      void processAllEntities( const GridPointer& gridPointer,
                               UserData& userData ) const;
};

template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
class Traverser< Meshes::Grid< 2, Real, Device, Index >, GridEntity, 1 >
{
   public:
      using GridType = Meshes::Grid< 2, Real, Device, Index >;
      using GridPointer = Pointers::SharedPointer< GridType >;
      using CoordinatesType = typename GridType::CoordinatesType;

      template< typename EntitiesProcessor,
                typename UserData >
      void processBoundaryEntities( const GridPointer& gridPointer,
                                    UserData& userData ) const;

      template< typename EntitiesProcessor,
                typename UserData >
      void processInteriorEntities( const GridPointer& gridPointer,
                                    UserData& userData ) const;

      template< typename EntitiesProcessor,
                typename UserData >
      void processAllEntities( const GridPointer& gridPointer,
                               UserData& userData ) const;
};

template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
class Traverser< Meshes::Grid< 2, Real, Device, Index >, GridEntity, 0 >
{
   public:
      using GridType = Meshes::Grid< 2, Real, Device, Index >;
      using GridPointer = Pointers::SharedPointer< GridType >;
      using CoordinatesType = typename GridType::CoordinatesType;

      template< typename EntitiesProcessor,
                typename UserData >
      void processBoundaryEntities( const GridPointer& gridPointer,
                                    UserData& userData ) const;

      template< typename EntitiesProcessor,
                typename UserData >
      void processInteriorEntities( const GridPointer& gridPointer,
                                    UserData& userData ) const;

      template< typename EntitiesProcessor,
                typename UserData >
      void processAllEntities( const GridPointer& gridPointer,
                               UserData& userData ) const;
};

} // namespace Meshes
} // namespace TNL

#include <TNL/Meshes/GridDetails/Traverser_Grid2D_impl.h>
