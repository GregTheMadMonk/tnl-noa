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
   template< typename UserData,
             typename EntitiesProcessor >
void
Traverser< Meshes::Grid< 1, Real, Device, Index >, GridEntity, 1 >::
processBoundaryEntities( const GridPointer& gridPointer,
                         UserData& userData ) const
{
   /****
    * Boundary cells
    */
   static_assert( GridEntity::entityDimensions == 1, "The entity has wrong dimensions." );

   GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, true >(
      gridPointer,
      CoordinatesType( 0 ),
      gridPointer->getDimensions() - CoordinatesType( 1 ),
      CoordinatesType(),
      CoordinatesType(),
      userData );
}

template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
   template< typename UserData,
             typename EntitiesProcessor >
void
Traverser< Meshes::Grid< 1, Real, Device, Index >, GridEntity, 1 >::
processInteriorEntities( const GridPointer& gridPointer,
                         UserData& userData ) const
{
   /****
    * Interior cells
    */
   static_assert( GridEntity::entityDimensions == 1, "The entity has wrong dimensions." );

   GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
      gridPointer,
      CoordinatesType( 1 ),
      gridPointer->getDimensions() - CoordinatesType( 2 ),
      CoordinatesType(),
      CoordinatesType(),
      userData );
}

template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
   template< typename UserData,
             typename EntitiesProcessor >
void
Traverser< Meshes::Grid< 1, Real, Device, Index >, GridEntity, 1 >::
processAllEntities(
   const GridPointer& gridPointer,
   UserData& userData ) const
{
   /****
    * All cells
    */
   static_assert( GridEntity::entityDimensions == 1, "The entity has wrong dimensions." );

   GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
      gridPointer,
      CoordinatesType( 0 ),
      gridPointer->getDimensions() - CoordinatesType( 1 ),
      CoordinatesType(),
      CoordinatesType(),
      userData );
}

/****
 * Grid 1D, vertices
 */
template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
   template< typename UserData,
             typename EntitiesProcessor >
void
Traverser< Meshes::Grid< 1, Real, Device, Index >, GridEntity, 0 >::
processBoundaryEntities( const GridPointer& gridPointer,
                         UserData& userData ) const
{
   /****
    * Boundary vertices
    */
   static_assert( GridEntity::entityDimensions == 0, "The entity has wrong dimensions." );

   GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, true >(
      gridPointer,
      CoordinatesType( 0 ),
      gridPointer->getDimensions(),
      CoordinatesType(),
      CoordinatesType(),
      userData );
}

template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
   template< typename UserData,
             typename EntitiesProcessor >
void
Traverser< Meshes::Grid< 1, Real, Device, Index >, GridEntity, 0 >::
processInteriorEntities( const GridPointer& gridPointer,
                         UserData& userData ) const
{
   /****
    * Interior vertices
    */
   static_assert( GridEntity::entityDimensions == 0, "The entity has wrong dimensions." );

   GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
      gridPointer,
      CoordinatesType( 1 ),
      gridPointer->getDimensions() - CoordinatesType( 1 ),
      CoordinatesType(),
      CoordinatesType(),
      userData );
}

template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
   template< typename UserData,
             typename EntitiesProcessor >
void
Traverser< Meshes::Grid< 1, Real, Device, Index >, GridEntity, 0 >::
processAllEntities(
   const GridPointer& gridPointer,
   UserData& userData ) const
{
   /****
    * All vertices
    */
   static_assert( GridEntity::entityDimensions == 0, "The entity has wrong dimensions." );

   GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
      gridPointer,
      CoordinatesType( 0 ),
      gridPointer->getDimensions(),
      CoordinatesType(),
      CoordinatesType(),
      userData );
}

} // namespace Meshes
} // namespace TNL
