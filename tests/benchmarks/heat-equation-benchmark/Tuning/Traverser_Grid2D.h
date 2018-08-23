/***************************************************************************
                          Traverser_Grid2D.h  -  description
                             -------------------
    begin                : Jul 29, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

//#include <TNL/Meshes/Traverser.h>
#include <TNL/Pointers/SharedPointer.h>

namespace TNL {
   
template< typename Mesh,
          typename MeshEntity,
          int EntitiesDimension = MeshEntity::getEntityDimension() >
class Traverser
{
   public:
      using MeshType = Mesh;
      using MeshPointer = Pointers::SharedPointer<  MeshType >;
      using DeviceType = typename MeshType::DeviceType;

      template< typename UserData,
                typename EntitiesProcessor >
      void processBoundaryEntities( const MeshPointer& meshPointer,
                                    Pointers::SharedPointer<  UserData, DeviceType >& userDataPointer ) const;

      template< typename UserData,
                typename EntitiesProcessor >
      void processInteriorEntities( const MeshPointer& meshPointer,
                                    Pointers::SharedPointer<  UserData, DeviceType >& userDataPointer ) const;

      template< typename UserData,
                typename EntitiesProcessor >
      void processAllEntities( const MeshPointer& meshPointer,
                               Pointers::SharedPointer<  UserData, DeviceType >& userDataPointer ) const;
}; 

template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
class Traverser< Meshes::Grid< 2, Real, Device, Index >, GridEntity, 2 >
{
   public:
      typedef Meshes::Grid< 2, Real, Device, Index > GridType;
      typedef Pointers::SharedPointer<  GridType > GridPointer;
      typedef typename GridType::CoordinatesType CoordinatesType;

      template< typename UserData,
                typename EntitiesProcessor >
      void processBoundaryEntities( const GridPointer& gridPointer,
                                    Pointers::SharedPointer<  UserData, Device >& userDataPointer ) const;

      template< typename UserData,
                typename EntitiesProcessor >
      void processInteriorEntities( const GridPointer& gridPointer,
                                    Pointers::SharedPointer<  UserData, Device >& userDataPointer ) const;
      template< typename UserData,
                typename EntitiesProcessor >
      void processAllEntities( const GridPointer& gridPointer,
                               Pointers::SharedPointer<  UserData, Device >& userDataPointer ) const;
 
};


} // namespace TNL

#include "Traverser_Grid2D_impl.h"
