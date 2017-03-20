/***************************************************************************
                          Traverser_Grid3D_impl.h  -  description
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
 * Grid 3D, cells
 */
template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
   template< typename UserData,
             typename EntitiesProcessor >
void
Traverser< Meshes::Grid< 3, Real, Device, Index >, GridEntity, 3 >::
processBoundaryEntities( const GridPointer& gridPointer,
                         SharedPointer< UserData, DeviceType >& userDataPointer ) const
{
   /****
    * Boundary cells
    */
   static_assert( GridEntity::entityDimensions == 3, "The entity has wrong dimensions." );

   GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, true, 1, 1, 1 >(
      gridPointer,
      CoordinatesType( 0, 0, 0 ),
      gridPointer->getDimensions() - CoordinatesType( 1, 1, 1 ),
      userDataPointer,
      0 );
}

template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
   template< typename UserData,
             typename EntitiesProcessor >
void
Traverser< Meshes::Grid< 3, Real, Device, Index >, GridEntity, 3 >::
processInteriorEntities( const GridPointer& gridPointer,
                         SharedPointer< UserData, DeviceType >& userDataPointer ) const
{
   /****
    * Interior cells
    */
   static_assert( GridEntity::entityDimensions == 3, "The entity has wrong dimensions." );
 
   GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
      gridPointer,
      CoordinatesType( 1, 1, 1 ),
      gridPointer->getDimensions() - CoordinatesType( 2, 2, 2 ),
      userDataPointer,
      0 );
}

template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
   template< typename UserData,
             typename EntitiesProcessor >
void
Traverser< Meshes::Grid< 3, Real, Device, Index >, GridEntity, 3 >::
processAllEntities( const GridPointer& gridPointer,
                    SharedPointer< UserData, DeviceType >& userDataPointer ) const
{
   /****
    * All cells
    */
   static_assert( GridEntity::entityDimensions == 3, "The entity has wrong dimensions." );
 
   GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
      gridPointer,
      CoordinatesType( 0, 0, 0 ),
      gridPointer->getDimensions() - CoordinatesType( 1, 1, 1 ),
      userDataPointer,
      0 );
}

/****
 * Grid 3D, faces
 */
template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
   template< typename UserData,
             typename EntitiesProcessor >
void
Traverser< Meshes::Grid< 3, Real, Device, Index >, GridEntity, 2 >::
processBoundaryEntities( const GridPointer& gridPointer,
                         SharedPointer< UserData, DeviceType >& userDataPointer ) const
{
   /****
    * Boundary faces
    */
   static_assert( GridEntity::entityDimensions == 2, "The entity has wrong dimensions." );
 
   GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, true, 1, 0, 0, CoordinatesType, CoordinatesType >(
      gridPointer,
      CoordinatesType( 0, 0, 0 ),
      gridPointer->getDimensions() - CoordinatesType( 0, 1, 1 ),
      userDataPointer,
      2,
      CoordinatesType( 1, 0, 0 ),
      CoordinatesType( 0, 1, 1 ) );

   GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, true, 0, 1, 0, CoordinatesType, CoordinatesType >(
      gridPointer,
      CoordinatesType( 0, 0, 0 ),
      gridPointer->getDimensions() - CoordinatesType( 1, 0, 1 ),
      userDataPointer,
      1,
      CoordinatesType( 0, 1, 0 ),
      CoordinatesType( 1, 0, 1 ) );
 
   GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, true, 0, 0, 1, CoordinatesType, CoordinatesType >(
      gridPointer,
      CoordinatesType( 0, 0, 0 ),
      gridPointer->getDimensions() - CoordinatesType( 1, 1, 0 ),
      userDataPointer,
      0,
      CoordinatesType( 0, 0, 1 ),
      CoordinatesType( 1, 1, 0 ) );
}

template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
   template< typename UserData,
             typename EntitiesProcessor >
void
Traverser< Meshes::Grid< 3, Real, Device, Index >, GridEntity, 2 >::
processInteriorEntities( const GridPointer& gridPointer,
                         SharedPointer< UserData, DeviceType >& userDataPointer ) const
{
   /****
    * Interior faces
    */
   static_assert( GridEntity::entityDimensions == 2, "The entity has wrong dimensions." );
 
   GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false, 1, 1, 1, CoordinatesType, CoordinatesType >(
      gridPointer,
      CoordinatesType( 1, 0, 0 ),
      gridPointer->getDimensions() - CoordinatesType( 1, 1, 1 ),
      userDataPointer,
      2,
      CoordinatesType( 1, 0, 0 ),
      CoordinatesType( 0, 1, 1 ) );

   GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false, 1, 1, 1, CoordinatesType, CoordinatesType >(
      gridPointer,
      CoordinatesType( 0, 1, 0 ),
      gridPointer->getDimensions() - CoordinatesType( 1, 1, 1 ),
      userDataPointer,
      1,
      CoordinatesType( 0, 1, 0 ),
      CoordinatesType( 1, 0, 1 ) );

   GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false, 1, 1, 1, CoordinatesType, CoordinatesType >(
      gridPointer,
      CoordinatesType( 0, 0, 1 ),
      gridPointer->getDimensions() - CoordinatesType( 1, 1, 1 ),
      userDataPointer,
      0,
      CoordinatesType( 0, 0, 1 ),
      CoordinatesType( 1, 1, 0 ) );
}

template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
   template< typename UserData,
             typename EntitiesProcessor >
void
Traverser< Meshes::Grid< 3, Real, Device, Index >, GridEntity, 2 >::
processAllEntities( const GridPointer& gridPointer,
                    SharedPointer< UserData, DeviceType >& userDataPointer ) const
{
   /****
    * All faces
    */
   static_assert( GridEntity::entityDimensions == 2, "The entity has wrong dimensions." );
   GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false, 1, 1, 1, CoordinatesType, CoordinatesType >(
      gridPointer,
      CoordinatesType( 0, 0, 0 ),
      gridPointer->getDimensions() - CoordinatesType( 0, 1, 1 ),
      userDataPointer,
      2,
      CoordinatesType( 1, 0, 0 ),
      CoordinatesType( 0, 1, 1 ) );

   GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false, 1, 1, 1, CoordinatesType, CoordinatesType >(
      gridPointer,
      CoordinatesType( 0, 0, 0 ),
      gridPointer->getDimensions() - CoordinatesType( 1, 0, 1 ),
      userDataPointer,
      1,
      CoordinatesType( 0, 1, 0 ),
      CoordinatesType( 1, 0, 1 ) );
 
   GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false, 1, 1, 1, CoordinatesType, CoordinatesType >(
      gridPointer,
      CoordinatesType( 0, 0, 0 ),
      gridPointer->getDimensions() - CoordinatesType( 1, 1, 0 ),
      userDataPointer,
      0,
      CoordinatesType( 0, 0, 1 ),
      CoordinatesType( 1, 1, 0 ) );
}

/****
 * Grid 3D, edges
 */
template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
   template< typename UserData,
             typename EntitiesProcessor >
void
Traverser< Meshes::Grid< 3, Real, Device, Index >, GridEntity, 1 >::
processBoundaryEntities( const GridPointer& gridPointer,
                         SharedPointer< UserData, DeviceType >& userDataPointer ) const
{
   /****
    * Boundary edges
    */
   static_assert( GridEntity::entityDimensions == 1, "The entity has wrong dimensions." );
 
   GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, true, 0, 1, 1, CoordinatesType, CoordinatesType >(
      gridPointer,
      CoordinatesType( 0, 0, 0 ),
      gridPointer->getDimensions() - CoordinatesType( 1, 0, 0 ),
      userDataPointer,
      2,
      CoordinatesType( 0, 1, 1 ),
      CoordinatesType( 1, 0, 0 ) );

   GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, true, 1, 0, 1, CoordinatesType, CoordinatesType >(
      gridPointer,
      CoordinatesType( 0, 0, 0 ),
      gridPointer->getDimensions() - CoordinatesType( 0, 1, 0 ),
      userDataPointer,
      1,
      CoordinatesType( 1, 0, 1 ),
      CoordinatesType( 0, 1, 0 ) );
 
   GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, true, 1, 1, 0, CoordinatesType, CoordinatesType >(
      gridPointer,
      CoordinatesType( 0, 0, 0 ),
      gridPointer->getDimensions() - CoordinatesType( 0, 0, 1 ),
      userDataPointer,
      0,
      CoordinatesType( 1, 1, 0 ),
      CoordinatesType( 0, 0, 1 ) );
}

template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
   template< typename UserData,
             typename EntitiesProcessor >
void
Traverser< Meshes::Grid< 3, Real, Device, Index >, GridEntity, 1 >::
processInteriorEntities( const GridPointer& gridPointer,
                         SharedPointer< UserData, DeviceType >& userDataPointer ) const
{
   /****
    * Interior edges
    */
   static_assert( GridEntity::entityDimensions == 1, "The entity has wrong dimensions." );
 
   GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false, 1, 1, 1, CoordinatesType, CoordinatesType >(
      gridPointer,
      CoordinatesType( 0, 1, 1 ),
      gridPointer->getDimensions() - CoordinatesType( 0, 1, 1 ),
      userDataPointer,
      2,
      CoordinatesType( 0, 1, 1 ),
      CoordinatesType( 1, 0, 0 ) );

   GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false, 1, 1, 1, CoordinatesType, CoordinatesType >(
      gridPointer,
      CoordinatesType( 1, 0, 1 ),
      gridPointer->getDimensions() - CoordinatesType( 1, 0, 1 ),
      userDataPointer,
      1,
      CoordinatesType( 1, 0, 1 ),
      CoordinatesType( 0, 1, 0 ) );

   GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false, 1, 1, 1, CoordinatesType, CoordinatesType >(
      gridPointer,
      CoordinatesType( 1, 1, 0 ),
      gridPointer->getDimensions() - CoordinatesType( 1, 1, 0 ),
      userDataPointer,
      0,
      CoordinatesType( 1, 1, 0 ),
      CoordinatesType( 0, 0, 1 ) );
}

template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
   template< typename UserData,
             typename EntitiesProcessor >
void
Traverser< Meshes::Grid< 3, Real, Device, Index >, GridEntity, 1 >::
processAllEntities( const GridPointer& gridPointer,
                    SharedPointer< UserData, DeviceType >& userDataPointer ) const
{
   /****
    * All edges
    */
   static_assert( GridEntity::entityDimensions == 1, "The entity has wrong dimensions." );
   GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false, 1, 1, 1, CoordinatesType, CoordinatesType >(
      gridPointer,
      CoordinatesType( 0, 0, 0 ),
      gridPointer->getDimensions() - CoordinatesType( 1, 0, 0 ),
      userDataPointer,
      2,
      CoordinatesType( 0, 1, 1 ),      
      CoordinatesType( 1, 0, 0 ) );

   GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false, 1, 1, 1, CoordinatesType, CoordinatesType >(
      gridPointer,
      CoordinatesType( 0, 0, 0 ),
      gridPointer->getDimensions() - CoordinatesType( 0, 1, 0 ),
      userDataPointer,
      1,
      CoordinatesType( 1, 0, 1 ),      
      CoordinatesType( 0, 1, 0 ) );
 
   GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false, 1, 1, 1, CoordinatesType, CoordinatesType >(
      gridPointer,
      CoordinatesType( 0, 0, 0 ),
      gridPointer->getDimensions() - CoordinatesType( 0, 0, 1 ),
      userDataPointer,
      0,
      CoordinatesType( 1, 1, 0 ),      
      CoordinatesType( 0, 0, 1 ) );
}

/****
 * Grid 3D, vertices
 */
template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
   template< typename UserData,
             typename EntitiesProcessor >
void
Traverser< Meshes::Grid< 3, Real, Device, Index >, GridEntity, 0 >::
processBoundaryEntities( const GridPointer& gridPointer,
                         SharedPointer< UserData, DeviceType >& userDataPointer ) const
{
   /****
    * Boundary vertices
    */
   static_assert( GridEntity::entityDimensions == 0, "The entity has wrong dimensions." );
 
   GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, true, 1, 1, 1 >(
      gridPointer,
      CoordinatesType( 0, 0, 0 ),
      gridPointer->getDimensions(),
      userDataPointer,
      0 );
}

template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
   template< typename UserData,
             typename EntitiesProcessor >
void
Traverser< Meshes::Grid< 3, Real, Device, Index >, GridEntity, 0 >::
processInteriorEntities( const GridPointer& gridPointer,
                         SharedPointer< UserData, DeviceType >& userDataPointer ) const
{
   /****
    * Interior vertices
    */
   static_assert( GridEntity::entityDimensions == 0, "The entity has wrong dimensions." );
 
   GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
      gridPointer,
      CoordinatesType( 1, 1, 1 ),
      gridPointer->getDimensions() - CoordinatesType( 1, 1, 1 ),
      userDataPointer,
      0 );
}
 
template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
   template< typename UserData,
             typename EntitiesProcessor >
void
Traverser< Meshes::Grid< 3, Real, Device, Index >, GridEntity, 0 >::
processAllEntities( const GridPointer& gridPointer,
                    SharedPointer< UserData, DeviceType >& userDataPointer ) const
{
   /****
    * All vertices
    */
   static_assert( GridEntity::entityDimensions == 0, "The entity has wrong dimensions." );
 
   GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
      gridPointer,
      CoordinatesType( 0, 0, 0 ),
      gridPointer->getDimensions(),
      userDataPointer,
      0 );
}

} // namespace Meshes
} // namespace TNL
