/***************************************************************************
                          Traverser_Grid3D.h  -  description
                             -------------------
    begin                : Jul 29, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {
namespace Meshes {

#include <TNL/SharedPointer.h>

template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
class Traverser< Meshes::Grid< 3, Real, Device, Index >, GridEntity, 3 >
{
   public:
      typedef Meshes::Grid< 3, Real, Device, Index > GridType;
      typedef SharedPointer< GridType > GridPointer;
      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;

      template< typename UserData,
                typename EntitiesProcessor >
      void processBoundaryEntities( const GridPointer& gridPointer,
                                    SharedPointer< UserData, DeviceType >& userDataPointer ) const;

      template< typename UserData,
                typename EntitiesProcessor >
      void processInteriorEntities( const GridPointer& gridPointer,
                                    SharedPointer< UserData, DeviceType >& userDataPointer ) const;
      template< typename UserData,
                typename EntitiesProcessor >
      void processAllEntities( const GridPointer& gridPointer,
                               SharedPointer< UserData, DeviceType >& userDataPointer ) const;
 
};

template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
class Traverser< Meshes::Grid< 3, Real, Device, Index >, GridEntity, 2 >
{
   public:
      typedef Meshes::Grid< 3, Real, Device, Index > GridType;
      typedef SharedPointer< GridType > GridPointer;
      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;

      template< typename UserData,
                typename EntitiesProcessor >
      void processBoundaryEntities( const GridPointer& gridPointer,
                                    SharedPointer< UserData, DeviceType >& userDataPointer ) const;

      template< typename UserData,
                typename EntitiesProcessor >
      void processInteriorEntities( const GridPointer& gridPointer,
                                    SharedPointer< UserData, DeviceType >& userDataPointer ) const;
      template< typename UserData,
                typename EntitiesProcessor >
      void processAllEntities( const GridPointer& gridPointer,
                               SharedPointer< UserData, DeviceType >& userDataPointer ) const;
 
};

template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
class Traverser< Meshes::Grid< 3, Real, Device, Index >, GridEntity, 1 >
{
   public:
      typedef Meshes::Grid< 3, Real, Device, Index > GridType;
      typedef SharedPointer< GridType > GridPointer;
      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;

      template< typename UserData,
                typename EntitiesProcessor >
      void processBoundaryEntities( const GridPointer& gridPointer,
                                    SharedPointer< UserData, DeviceType >& userDataPointer ) const;

      template< typename UserData,
                typename EntitiesProcessor >
      void processInteriorEntities( const GridPointer& gridPointer,
                                    SharedPointer< UserData, DeviceType >& userDataPointer ) const;

      template< typename UserData,
                typename EntitiesProcessor >
      void processAllEntities( const GridPointer& gridPointer,
                               SharedPointer< UserData, DeviceType >& userDataPointer ) const;
 
};

template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
class Traverser< Meshes::Grid< 3, Real, Device, Index >, GridEntity, 0 >
{
   public:
      typedef Meshes::Grid< 3, Real, Device, Index > GridType;
      typedef SharedPointer< GridType > GridPointer;
      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;

      template< typename UserData,
                typename EntitiesProcessor >
      void processBoundaryEntities( const GridPointer& gridPointer,
                                    SharedPointer< UserData, DeviceType >& userDataPointer ) const;

      template< typename UserData,
                typename EntitiesProcessor >
      void processInteriorEntities( const GridPointer& gridPointer,
                                    SharedPointer< UserData, DeviceType >& userDataPointer ) const;
 
      template< typename UserData,
                typename EntitiesProcessor >
      void processAllEntities( const GridPointer& gridPointer,
                               SharedPointer< UserData, DeviceType >& userDataPointer ) const;
};

} // namespace Meshes
} // namespace TNL

#include <TNL/Meshes/GridDetails/Traverser_Grid3D_impl.h>
