/***************************************************************************
                          Traverser_Grid1D.h  -  description
                             -------------------
    begin                : Jul 28, 2014
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
class Traverser< Meshes::Grid< 1, Real, Device, Index >, GridEntity, 1 >
{
   public:
      typedef Meshes::Grid< 1, Real, Device, Index > GridType;
      typedef SharedPointer< GridType > GridPointer;
      typedef typename GridType::CoordinatesType CoordinatesType;

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
class Traverser< Meshes::Grid< 1, Real, Device, Index >, GridEntity, 0 >
{
   public:
      typedef Meshes::Grid< 1, Real, Device, Index > GridType;
      typedef SharedPointer< GridType > GridPointer;
      typedef typename GridType::CoordinatesType CoordinatesType;

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

#include <TNL/Meshes/GridDetails/Traverser_Grid1D_impl.h>
