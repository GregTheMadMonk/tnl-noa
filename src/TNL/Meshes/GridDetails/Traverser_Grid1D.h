// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Traverser.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Pointers/SharedPointer.h>

namespace noa::TNL {
namespace Meshes {

template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
class Traverser< Meshes::Grid< 1, Real, Device, Index >, GridEntity, 1 >
{
   public:
      using GridType = Meshes::Grid< 1, Real, Device, Index >;
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
class Traverser< Meshes::Grid< 1, Real, Device, Index >, GridEntity, 0 >
{
   public:
      using GridType = Meshes::Grid< 1, Real, Device, Index >;
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
} // namespace noa::TNL

#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/GridDetails/Traverser_Grid1D_impl.h>
