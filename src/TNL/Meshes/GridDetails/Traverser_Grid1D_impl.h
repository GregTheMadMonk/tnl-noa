/***************************************************************************
                          Traverser_Grid1D_impl.h  -  description
                             -------------------
    begin                : Jul 29, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Meshes/GridDetails/GridTraverser.h>

namespace TNL {
namespace Meshes {

/****
 * Grid 1D, cells
 */
template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
   template< typename EntitiesProcessor,
             typename UserData >
void
Traverser< Meshes::Grid< 1, Real, Device, Index >, GridEntity, 1 >::
processBoundaryEntities( const GridPointer& gridPointer,
                         UserData& userData ) const
{
   /****
    * Boundary cells
    */
   static_assert( GridEntity::getEntityDimension() == 1, "The entity has wrong dimensions." );

   if( gridPointer->getLocalBegin() < gridPointer->getInteriorBegin() && gridPointer->getInteriorEnd() < gridPointer->getLocalEnd() )
   {
      // 2 boundaries (left and right)
      GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, true >(
         gridPointer,
         gridPointer->getInteriorBegin() - 1,
         gridPointer->getInteriorEnd() + 1,
         userData,
         asynchronousMode );
   }
   else if( gridPointer->getLocalBegin() < gridPointer->getInteriorBegin() )
   {
      // 1 boundary (left)
      GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
         gridPointer,
         gridPointer->getInteriorBegin() - 1,
         gridPointer->getInteriorBegin(),
         userData,
         asynchronousMode );
   }
   else if( gridPointer->getInteriorEnd() < gridPointer->getLocalEnd() )
   {
      // 1 boundary (right)
      GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
         gridPointer,
         gridPointer->getInteriorEnd(),
         gridPointer->getInteriorEnd() + 1,
         userData,
         asynchronousMode );
   }
}

template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
   template< typename EntitiesProcessor,
             typename UserData >
void
Traverser< Meshes::Grid< 1, Real, Device, Index >, GridEntity, 1 >::
processInteriorEntities( const GridPointer& gridPointer,
                         UserData& userData ) const
{
   /****
    * Interior cells
    */
   static_assert( GridEntity::getEntityDimension() == 1, "The entity has wrong dimension." );

   GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
      gridPointer,
      gridPointer->getInteriorBegin(),
      gridPointer->getInteriorEnd(),
      userData,
      asynchronousMode );
}

template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
   template< typename EntitiesProcessor,
             typename UserData >
void
Traverser< Meshes::Grid< 1, Real, Device, Index >, GridEntity, 1 >::
processAllEntities(
   const GridPointer& gridPointer,
   UserData& userData ) const
{
   /****
    * All cells
    */
   static_assert( GridEntity::getEntityDimension() == 1, "The entity has wrong dimensions." );

   GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
      gridPointer,
      gridPointer->getLocalBegin(),
      gridPointer->getLocalEnd(),
      userData,
      asynchronousMode );
}

/****
 * Grid 1D, vertices
 */
template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
   template< typename EntitiesProcessor,
             typename UserData >
void
Traverser< Meshes::Grid< 1, Real, Device, Index >, GridEntity, 0 >::
processBoundaryEntities( const GridPointer& gridPointer,
                         UserData& userData ) const
{
   /****
    * Boundary vertices
    */
   static_assert( GridEntity::getEntityDimension() == 0, "The entity has wrong dimension." );

   GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, true >(
      gridPointer,
      CoordinatesType( 0 ),
      gridPointer->getDimensions() + 1,
      userData,
      asynchronousMode );
}

template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
   template< typename EntitiesProcessor,
             typename UserData >
void
Traverser< Meshes::Grid< 1, Real, Device, Index >, GridEntity, 0 >::
processInteriorEntities( const GridPointer& gridPointer,
                         UserData& userData ) const
{
   /****
    * Interior vertices
    */
   static_assert( GridEntity::getEntityDimension() == 0, "The entity has wrong dimension." );

   GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
      gridPointer,
      CoordinatesType( 1 ),
      gridPointer->getDimensions(),
      userData,
      asynchronousMode );
}

template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
   template< typename EntitiesProcessor,
             typename UserData >
void
Traverser< Meshes::Grid< 1, Real, Device, Index >, GridEntity, 0 >::
processAllEntities(
   const GridPointer& gridPointer,
   UserData& userData ) const
{
   /****
    * All vertices
    */
   static_assert( GridEntity::getEntityDimension() == 0, "The entity has wrong dimension." );

   GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
      gridPointer,
      CoordinatesType( 0 ),
      gridPointer->getDimensions() + 1,
      userData,
      asynchronousMode );
}

} // namespace Meshes
} // namespace TNL
