/***************************************************************************
                          tnlTraverser_Grid2D.h  -  description
                             -------------------
    begin                : Jul 29, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {

template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
class tnlTraverser< tnlGrid< 2, Real, Device, Index >, GridEntity, 2 >
{
   public:
      typedef tnlGrid< 2, Real, Device, Index > GridType;
      typedef tnlSharedPointer< GridType > GridPointer;
      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;

      template< typename UserData,
                typename EntitiesProcessor >
      void processBoundaryEntities( const GridPointer& gridPointer,
                                    UserData& userData ) const;

      template< typename UserData,
                typename EntitiesProcessor >
      void processInteriorEntities( const GridPointer& gridPointer,
                                    UserData& userData ) const;
      template< typename UserData,
                typename EntitiesProcessor >
      void processAllEntities( const GridPointer& gridPointer,
                               UserData& userData ) const;
 
};

template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
class tnlTraverser< tnlGrid< 2, Real, Device, Index >, GridEntity, 1 >
{
   public:
      typedef tnlGrid< 2, Real, Device, Index > GridType;
      typedef tnlSharedPointer< GridType > GridPointer;
      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;

      template< typename UserData,
                typename EntitiesProcessor >
      void processBoundaryEntities( const GridPointer& gridPointer,
                                    UserData& userData ) const;

      template< typename UserData,
                typename EntitiesProcessor >
      void processInteriorEntities( const GridPointer& gridPointer,
                                    UserData& userData ) const;

      template< typename UserData,
                typename EntitiesProcessor >
      void processAllEntities( const GridPointer& gridPointer,
                               UserData& userData ) const;
 
};

template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
class tnlTraverser< tnlGrid< 2, Real, Device, Index >, GridEntity, 0 >
{
   public:
      typedef tnlGrid< 2, Real, Device, Index > GridType;
      typedef tnlSharedPointer< GridType > GridPointer;
      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;

      template< typename UserData,
                typename EntitiesProcessor >
      void processBoundaryEntities( const GridPointer& gridPointer,
                                    UserData& userData ) const;

      template< typename UserData,
                typename EntitiesProcessor >
      void processInteriorEntities( const GridPointer& gridPointer,
                                    UserData& userData ) const;
      
      template< typename UserData,
                typename EntitiesProcessor >
      void processAllEntities( const GridPointer& gridPointer,
                               UserData& userData ) const;
};

} //namespace TNL

#include <TNL/mesh/grids/tnlTraverser_Grid2D_impl.h>
