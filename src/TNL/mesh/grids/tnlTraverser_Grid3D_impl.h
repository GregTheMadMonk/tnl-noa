/***************************************************************************
                          tnlTraverser_Grid3D_impl.h  -  description
                             -------------------
    begin                : Jul 29, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/mesh/grids/tnlGridTraverser.h>

namespace TNL {

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
tnlTraverser< tnlGrid< 3, Real, Device, Index >, GridEntity, 3 >::
processBoundaryEntities( const GridType& grid,
                         UserData& userData ) const
{
   /****
    * Boundary cells
    */
   static_assert( GridEntity::entityDimensions == 3, "The entity has wrong dimensions." );

   tnlGridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, true, 1, 1, 1 >(
      grid,
      CoordinatesType( 0, 0, 0 ),
      grid.getDimensions() - CoordinatesType( 1, 1, 1 ),
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
tnlTraverser< tnlGrid< 3, Real, Device, Index >, GridEntity, 3 >::
processInteriorEntities( const GridType& grid,
                         UserData& userData ) const
{
   /****
    * Interior cells
    */
   static_assert( GridEntity::entityDimensions == 3, "The entity has wrong dimensions." );
 
   tnlGridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
      grid,
      CoordinatesType( 1, 1, 1 ),
      grid.getDimensions() - CoordinatesType( 2, 2, 2 ),
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
tnlTraverser< tnlGrid< 3, Real, Device, Index >, GridEntity, 3 >::
processAllEntities( const GridType& grid,
                    UserData& userData ) const
{
   /****
    * All cells
    */
   static_assert( GridEntity::entityDimensions == 3, "The entity has wrong dimensions." );
 
   tnlGridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
      grid,
      CoordinatesType( 0, 0, 0 ),
      grid.getDimensions() - CoordinatesType( 1, 1, 1 ),
      CoordinatesType(),
      CoordinatesType(),
      userData );
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
tnlTraverser< tnlGrid< 3, Real, Device, Index >, GridEntity, 2 >::
processBoundaryEntities( const GridType& grid,
                         UserData& userData ) const
{
   /****
    * Boundary faces
    */
   static_assert( GridEntity::entityDimensions == 2, "The entity has wrong dimensions." );
 
   tnlGridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, true, 1, 0, 0 >(
      grid,
      CoordinatesType( 0, 0, 0 ),
      grid.getDimensions() - CoordinatesType( 0, 1, 1 ),
      CoordinatesType( 1, 0, 0 ),
      CoordinatesType( 0, 1, 1 ),
      userData );

   tnlGridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, true, 0, 1, 0 >(
      grid,
      CoordinatesType( 0, 0, 0 ),
      grid.getDimensions() - CoordinatesType( 1, 0, 1 ),
      CoordinatesType( 0, 1, 0 ),
      CoordinatesType( 1, 0, 1 ),
      userData );
 
   tnlGridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, true, 0, 0, 1 >(
      grid,
      CoordinatesType( 0, 0, 0 ),
      grid.getDimensions() - CoordinatesType( 1, 1, 0 ),
      CoordinatesType( 0, 1, 0 ),
      CoordinatesType( 1, 1, 0 ),
      userData );
}

template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
   template< typename UserData,
             typename EntitiesProcessor >
void
tnlTraverser< tnlGrid< 3, Real, Device, Index >, GridEntity, 2 >::
processInteriorEntities( const GridType& grid,
                         UserData& userData ) const
{
   /****
    * Interior faces
    */
   static_assert( GridEntity::entityDimensions == 2, "The entity has wrong dimensions." );
 
   tnlGridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
      grid,
      CoordinatesType( 1, 0, 0 ),
      grid.getDimensions() - CoordinatesType( 1, 1, 1 ),
      CoordinatesType( 1, 0, 0 ),
      CoordinatesType( 0, 1, 1 ),
      userData );

   tnlGridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
      grid,
      CoordinatesType( 0, 1, 0 ),
      grid.getDimensions() - CoordinatesType( 1, 1, 1 ),
      CoordinatesType( 0, 1, 0 ),
      CoordinatesType( 1, 0, 1 ),
      userData );

   tnlGridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
      grid,
      CoordinatesType( 0, 0, 1 ),
      grid.getDimensions() - CoordinatesType( 1, 1, 1 ),
      CoordinatesType( 0, 0, 1 ),
      CoordinatesType( 1, 1, 0 ),
      userData );
}

template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
   template< typename UserData,
             typename EntitiesProcessor >
void
tnlTraverser< tnlGrid< 3, Real, Device, Index >, GridEntity, 2 >::
processAllEntities( const GridType& grid,
                         UserData& userData ) const
{
   /****
    * All faces
    */
   static_assert( GridEntity::entityDimensions == 2, "The entity has wrong dimensions." );
   tnlGridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
      grid,
      CoordinatesType( 0, 0, 0 ),
      grid.getDimensions() - CoordinatesType( 0, 1, 1 ),
      CoordinatesType( 1, 0, 0 ),
      CoordinatesType( 0, 1, 1 ),
      userData );

   tnlGridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
      grid,
      CoordinatesType( 0, 0, 0 ),
      grid.getDimensions() - CoordinatesType( 1, 0, 1 ),
      CoordinatesType( 0, 1, 0 ),
      CoordinatesType( 1, 0, 1 ),
      userData );
 
   tnlGridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
      grid,
      CoordinatesType( 0, 0, 0 ),
      grid.getDimensions() - CoordinatesType( 1, 1, 0 ),
      CoordinatesType( 0, 0, 1 ),
      CoordinatesType( 1, 1, 0 ),
      userData );
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
tnlTraverser< tnlGrid< 3, Real, Device, Index >, GridEntity, 1 >::
processBoundaryEntities( const GridType& grid,
                         UserData& userData ) const
{
   /****
    * Boundary edges
    */
   static_assert( GridEntity::entityDimensions == 1, "The entity has wrong dimensions." );
 
   tnlGridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, true, 0, 1, 1 >(
      grid,
      CoordinatesType( 0, 0, 0 ),
      grid.getDimensions() - CoordinatesType( 1, 0, 0 ),
      CoordinatesType( 0, 1, 1 ),
      CoordinatesType( 1, 0, 0 ),
      userData );

   tnlGridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, true, 1, 0, 1 >(
      grid,
      CoordinatesType( 0, 0, 0 ),
      grid.getDimensions() - CoordinatesType( 0, 1, 0 ),
      CoordinatesType( 1, 0, 1 ),
      CoordinatesType( 0, 1, 0 ),
      userData );
 
   tnlGridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, true, 1, 1, 0 >(
      grid,
      CoordinatesType( 0, 0, 0 ),
      grid.getDimensions() - CoordinatesType( 0, 0, 1 ),
      CoordinatesType( 1, 1, 0 ),
      CoordinatesType( 0, 0, 1 ),
      userData );
}

template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
   template< typename UserData,
             typename EntitiesProcessor >
void
tnlTraverser< tnlGrid< 3, Real, Device, Index >, GridEntity, 1 >::
processInteriorEntities( const GridType& grid,
                         UserData& userData ) const
{
   /****
    * Interior edges
    */
   static_assert( GridEntity::entityDimensions == 1, "The entity has wrong dimensions." );
 
   tnlGridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
      grid,
      CoordinatesType( 0, 1, 1 ),
      grid.getDimensions() - CoordinatesType( 0, 1, 1 ),
      CoordinatesType( 0, 1, 1 ),
      CoordinatesType( 1, 0, 0 ),
      userData );

   tnlGridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
      grid,
      CoordinatesType( 1, 0, 1 ),
      grid.getDimensions() - CoordinatesType( 1, 0, 1 ),
      CoordinatesType( 1, 0, 1 ),
      CoordinatesType( 0, 1, 0 ),
      userData );

   tnlGridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
      grid,
      CoordinatesType( 1, 1, 0 ),
      grid.getDimensions() - CoordinatesType( 1, 1, 0 ),
      CoordinatesType( 1, 1, 0 ),
      CoordinatesType( 0, 0, 1 ),
      userData );
}

template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
   template< typename UserData,
             typename EntitiesProcessor >
void
tnlTraverser< tnlGrid< 3, Real, Device, Index >, GridEntity, 1 >::
processAllEntities( const GridType& grid,
                    UserData& userData ) const
{
   /****
    * All edges
    */
   static_assert( GridEntity::entityDimensions == 1, "The entity has wrong dimensions." );
   tnlGridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
      grid,
      CoordinatesType( 0, 0, 0 ),
      grid.getDimensions() - CoordinatesType( 1, 0, 0 ),
      CoordinatesType( 0, 1, 1 ),
      CoordinatesType( 1, 0, 0 ),
      userData );

   tnlGridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
      grid,
      CoordinatesType( 0, 0, 0 ),
      grid.getDimensions() - CoordinatesType( 0, 1, 0 ),
      CoordinatesType( 1, 0, 1 ),
      CoordinatesType( 0, 1, 0 ),
      userData );
 
   tnlGridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
      grid,
      CoordinatesType( 0, 0, 0 ),
      grid.getDimensions() - CoordinatesType( 0, 0, 1 ),
      CoordinatesType( 1, 1, 0 ),
      CoordinatesType( 0, 0, 1 ),
      userData );
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
tnlTraverser< tnlGrid< 3, Real, Device, Index >, GridEntity, 0 >::
processBoundaryEntities( const GridType& grid,
                         UserData& userData ) const
{
   /****
    * Boundary vertices
    */
   static_assert( GridEntity::entityDimensions == 0, "The entity has wrong dimensions." );
 
   tnlGridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, true, 1, 1, 1 >(
      grid,
      CoordinatesType( 0, 0, 0 ),
      grid.getDimensions(),
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
tnlTraverser< tnlGrid< 3, Real, Device, Index >, GridEntity, 0 >::
processInteriorEntities( const GridType& grid,
                         UserData& userData ) const
{
   /****
    * Interior vertices
    */
   static_assert( GridEntity::entityDimensions == 0, "The entity has wrong dimensions." );
 
   tnlGridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
      grid,
      CoordinatesType( 1, 1, 1 ),
      grid.getDimensions() - CoordinatesType( 1, 1, 1 ),
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
tnlTraverser< tnlGrid< 3, Real, Device, Index >, GridEntity, 0 >::
processAllEntities( const GridType& grid,
                         UserData& userData ) const
{
   /****
    * All vertices
    */
   static_assert( GridEntity::entityDimensions == 0, "The entity has wrong dimensions." );
 
   tnlGridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
      grid,
      CoordinatesType( 0, 0, 0 ),
      grid.getDimensions(),
      CoordinatesType(),
      CoordinatesType(),
      userData );
}

} // namespace TNL
