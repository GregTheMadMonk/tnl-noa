/***************************************************************************
                          Traverser_Grid2D_impl.h  -  description
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
 * Grid 2D, cells
 */
template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
   template< typename UserData,
             typename EntitiesProcessor >
void
Traverser< Meshes::Grid< 2, Real, Device, Index >, GridEntity, 2 >::
processBoundaryEntities( const GridPointer& gridPointer,
                         UserData& userData ) const
{
   /****
    * Boundary cells
    */
   static_assert( GridEntity::entityDimensions == 2, "The entity has wrong dimensions." );

   GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, true, 1, 1 >(
      gridPointer,
      CoordinatesType( 0, 0 ),
      gridPointer->getDimensions() - CoordinatesType( 1, 1 ),
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
Traverser< Meshes::Grid< 2, Real, Device, Index >, GridEntity, 2 >::
processInteriorEntities( const GridPointer& gridPointer,
                         UserData& userData ) const
{
   /****
    * Interior cells
    */
   static_assert( GridEntity::entityDimensions == 2, "The entity has wrong dimensions." );
 
   GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
      gridPointer,
      CoordinatesType( 1, 1 ),
      gridPointer->getDimensions() - CoordinatesType( 2, 2 ),
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
Traverser< Meshes::Grid< 2, Real, Device, Index >, GridEntity, 2 >::
processAllEntities( const GridPointer& gridPointer,
                    UserData& userData ) const
{
   /****
    * All cells
    */
   static_assert( GridEntity::entityDimensions == 2, "The entity has wrong dimensions." );
 
   GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
      gridPointer,
      CoordinatesType( 0, 0 ),
      gridPointer->getDimensions() - CoordinatesType( 1, 1 ),
      CoordinatesType(),
      CoordinatesType(),
      userData );
}

/****
 * Grid 2D, faces
 */
template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
   template< typename UserData,
             typename EntitiesProcessor >
void
Traverser< Meshes::Grid< 2, Real, Device, Index >, GridEntity, 1 >::
processBoundaryEntities( const GridPointer& gridPointer,
                         UserData& userData ) const
{
   /****
    * Boundary faces
    */
   static_assert( GridEntity::entityDimensions == 1, "The entity has wrong dimensions." );
 
   GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, true, 1, 0 >(
      gridPointer,
      CoordinatesType( 0, 0 ),
      gridPointer->getDimensions() - CoordinatesType( 0, 1 ),
      CoordinatesType( 1, 0 ),
      CoordinatesType( 0, 1 ),
      userData );

   GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, true, 0, 1 >(
      gridPointer,
      CoordinatesType( 0, 0 ),
      gridPointer->getDimensions() - CoordinatesType( 1, 0 ),
      CoordinatesType( 0, 1 ),
      CoordinatesType( 1, 0 ),
      userData );
}

template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
   template< typename UserData,
             typename EntitiesProcessor >
void
Traverser< Meshes::Grid< 2, Real, Device, Index >, GridEntity, 1 >::
processInteriorEntities( const GridPointer& gridPointer,
                         UserData& userData ) const
{
   /****
    * Interior faces
    */
   static_assert( GridEntity::entityDimensions == 1, "The entity has wrong dimensions." );
 
   GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
      gridPointer,
      CoordinatesType( 1, 0 ),
      gridPointer->getDimensions() - CoordinatesType( 1, 1 ),
      CoordinatesType( 1, 0 ),
      CoordinatesType( 0, 1 ),
      userData );

   GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
      gridPointer,
      CoordinatesType( 0, 1 ),
      gridPointer->getDimensions() - CoordinatesType( 1, 1 ),
      CoordinatesType( 0, 1 ),
      CoordinatesType( 1, 0 ),
      userData );
}

template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
   template< typename UserData,
             typename EntitiesProcessor >
void
Traverser< Meshes::Grid< 2, Real, Device, Index >, GridEntity, 1 >::
processAllEntities( const GridPointer& gridPointer,
                    UserData& userData ) const
{
   /****
    * All faces
    */
   static_assert( GridEntity::entityDimensions == 1, "The entity has wrong dimensions." );
 
   GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
      gridPointer,
      CoordinatesType( 0, 0 ),
      gridPointer->getDimensions() - CoordinatesType( 0, 1 ),
      CoordinatesType( 1, 0 ),
      CoordinatesType( 0, 1 ),
      userData );

   GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
      gridPointer,
      CoordinatesType( 0, 0 ),
      gridPointer->getDimensions() - CoordinatesType( 1, 0 ),
      CoordinatesType( 0, 1 ),
      CoordinatesType( 1, 0 ),
      userData );
}

template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
   template< typename UserData,
             typename EntitiesProcessor >
void
Traverser< Meshes::Grid< 2, Real, Device, Index >, GridEntity, 0 >::
processBoundaryEntities( const GridPointer& gridPointer,
                         UserData& userData ) const
{
   /****
    * Boundary vertices
    */
   static_assert( GridEntity::entityDimensions == 0, "The entity has wrong dimensions." );
 
   GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, true, 1, 1 >(
      gridPointer,
      CoordinatesType( 0, 0 ),
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
Traverser< Meshes::Grid< 2, Real, Device, Index >, GridEntity, 0 >::
processInteriorEntities( const GridPointer& gridPointer,
                         UserData& userData ) const
{
   /****
    * Interior vertices
    */
   static_assert( GridEntity::entityDimensions == 0, "The entity has wrong dimensions." );
 
   GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
      gridPointer,
      CoordinatesType( 1, 1 ),
      gridPointer->getDimensions() - CoordinatesType( 1, 1 ),
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
Traverser< Meshes::Grid< 2, Real, Device, Index >, GridEntity, 0 >::
processAllEntities( const GridPointer& gridPointer,
                         UserData& userData ) const
{
   /****
    * All vertices
    */
   static_assert( GridEntity::entityDimensions == 0, "The entity has wrong dimensions." );
 
   GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
      gridPointer,
      CoordinatesType( 0, 0 ),
      gridPointer->getDimensions(),
      CoordinatesType(),
      CoordinatesType(),
      userData );
}

} // namespace Meshes
} // namespace TNL
