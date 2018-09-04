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

#include "Traverser_Grid3D.h"

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
                         SharedPointer< UserData, Device >& userDataPointer ) const
{
   /****
    * Boundary cells
    */
   static_assert( GridEntity::getEntityDimension() == 3, "The entity has wrong dimension." );

   DistributedGridType* distributedGrid = gridPointer->getDistributedMesh();
   if( distributedGrid == nullptr || !distributedGrid->isDistributed() )
   {
      GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, true, 1, 1, 1 >(
          gridPointer,
          CoordinatesType( 0, 0, 0 ),
          gridPointer->getDimensions() - CoordinatesType( 1, 1, 1 ),
          userDataPointer,
          0 );
   }
   else // distributed
   {
      const CoordinatesType begin = distributedGrid->getLowerOverlap();
      const CoordinatesType end = gridPointer->getDimensions() - distributedGrid->getUpperOverlap() -
                                  CoordinatesType( 1, 1, 1 );
      const int* neighbors = distributedGrid->getNeighbors();
      if( neighbors[ Meshes::DistributedMeshes::West ] == -1 )
      {
         GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
            gridPointer,
            begin,
            CoordinatesType( begin.x(), end.x(), end.y() ),
            userDataPointer,
            0 );
      }
       
      if( neighbors[ Meshes::DistributedMeshes::East ] == -1 )
      {
          GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
            gridPointer,
            CoordinatesType( end.x() , begin.y(), begin.z() ),
            end,
            userDataPointer,
            0 );
       }
       
      if( neighbors[ Meshes::DistributedMeshes::North ] == -1 )
      {
         GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
            gridPointer,
            begin,
            CoordinatesType( end.x(), begin.y(), end.z() ),
            userDataPointer,
            0 );
      }
       
      if( neighbors[ Meshes::DistributedMeshes::South ] == -1 )
      {
         GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
            gridPointer,
            CoordinatesType( begin.x(), end.y(), begin.z() ),
            end,
            userDataPointer,
            0 );
       }
       
      if( neighbors[ Meshes::DistributedMeshes::Bottom ] == -1 )
      {
         GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
            gridPointer,
            begin,
            CoordinatesType( end.x(), end.y(), begin.z() ),
            userDataPointer,
            0 );
      }
      
      if( neighbors[ Meshes::DistributedMeshes::Top ] == -1 )
      {
         GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
            gridPointer,
            CoordinatesType( begin.x(), begin.y(), end.z() ),
            end,
            userDataPointer,
            0 );
      } 
   }
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
                         SharedPointer< UserData, Device >& userDataPointer ) const
{
   /****
    * Interior cells
    */
   static_assert( GridEntity::getEntityDimension() == 3, "The entity has wrong dimension." );
   
   DistributedGridType* distributedGrid = gridPointer->getDistributedMesh();
   if( distributedGrid == nullptr || !distributedGrid->isDistributed() )
   { 
      GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
         gridPointer,
         CoordinatesType( 1, 1, 1 ),
         gridPointer->getDimensions() - CoordinatesType( 2, 2, 2 ),
         userDataPointer,
         0 );
   }
   else
   {
      const int* neighbors = distributedGrid->getNeighbors(); 
      CoordinatesType begin( distributedGrid->getLowerOverlap());
      CoordinatesType end( gridPointer->getDimensions() - distributedGrid->getUpperOverlap()- CoordinatesType(1,1,1) );
      
      if( neighbors[ Meshes::DistributedMeshes::West ] == -1 )
         begin.x() += 1 ;
       
      if( neighbors[ Meshes::DistributedMeshes::East ] == -1 )
         end.x() -= 1;
       
      if( neighbors[ Meshes::DistributedMeshes::North ] == -1 )
         begin.y() += 1;
       
      if( neighbors[ Meshes::DistributedMeshes::South ] == -1 )
         end.y() -= 1;
       
      if( neighbors[ Meshes::DistributedMeshes::Bottom ] == -1 )
         begin.z() += 1 ;
       
      if( neighbors[ Meshes::DistributedMeshes::Top ] == -1 )
         end.z() -= 1;

      GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
         gridPointer,
         begin,
         end,
         userDataPointer,
         0);      
   }
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
                    SharedPointer< UserData, Device >& userDataPointer ) const
{
   /****
    * All cells
    */
   static_assert( GridEntity::getEntityDimension() == 3, "The entity has wrong dimension." );
 
   DistributedGridType* distributedGrid=gridPointer->getDistributedMesh();
   if( distributedGrid == nullptr || !distributedGrid->isDistributed() )
   { 
      GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
         gridPointer,
         CoordinatesType( 0, 0, 0 ),
         gridPointer->getDimensions() - CoordinatesType( 1, 1, 1 ),
         userDataPointer,
         0 );
   }
   else
   {
      CoordinatesType begin( distributedGrid->getLowerOverlap());
      CoordinatesType end( gridPointer->getDimensions() - distributedGrid->getUpperOverlap()- CoordinatesType(1,1,1) );

      GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
         gridPointer,
         begin,
         end,
         userDataPointer,
         0 ); 
   }
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
                         SharedPointer< UserData, Device >& userDataPointer ) const
{
   /****
    * Boundary faces
    */
   static_assert( GridEntity::getEntityDimension() == 2, "The entity has wrong dimension." );
 
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
                         SharedPointer< UserData, Device >& userDataPointer ) const
{
   /****
    * Interior faces
    */
   static_assert( GridEntity::getEntityDimension() == 2, "The entity has wrong dimension." );
 
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
                    SharedPointer< UserData, Device >& userDataPointer ) const
{
   /****
    * All faces
    */
   static_assert( GridEntity::getEntityDimension() == 2, "The entity has wrong dimension." );
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
                         SharedPointer< UserData, Device >& userDataPointer ) const
{
   /****
    * Boundary edges
    */
   static_assert( GridEntity::getEntityDimension() == 1, "The entity has wrong dimension." );
 
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
                         SharedPointer< UserData, Device >& userDataPointer ) const
{
   /****
    * Interior edges
    */
   static_assert( GridEntity::getEntityDimension() == 1, "The entity has wrong dimension." );
 
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
                    SharedPointer< UserData, Device >& userDataPointer ) const
{
   /****
    * All edges
    */
   static_assert( GridEntity::getEntityDimension() == 1, "The entity has wrong dimension." );
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
                         SharedPointer< UserData, Device >& userDataPointer ) const
{
   /****
    * Boundary vertices
    */
   static_assert( GridEntity::getEntityDimension() == 0, "The entity has wrong dimension." );
 
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
                         SharedPointer< UserData, Device >& userDataPointer ) const
{
   /****
    * Interior vertices
    */
   static_assert( GridEntity::getEntityDimension() == 0, "The entity has wrong dimension." );
 
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
                    SharedPointer< UserData, Device >& userDataPointer ) const
{
   /****
    * All vertices
    */
   static_assert( GridEntity::getEntityDimension() == 0, "The entity has wrong dimension." );
 
   GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
      gridPointer,
      CoordinatesType( 0, 0, 0 ),
      gridPointer->getDimensions(),
      userDataPointer,
      0 );
}

} // namespace Meshes
} // namespace TNL
