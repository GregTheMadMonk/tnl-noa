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
   template< typename EntitiesProcessor,
             typename UserData >
void
Traverser< Meshes::Grid< 2, Real, Device, Index >, GridEntity, 2 >::
processBoundaryEntities( const GridPointer& gridPointer,
                         UserData& userData ) const
{
   /****
    * Boundary cells
    */
   static_assert( GridEntity::getEntityDimension() == 2, "The entity has wrong dimension." );

   DistributedGridType* distributedGrid = gridPointer->getDistributedMesh();
   if( distributedGrid == nullptr || !distributedGrid->isDistributed() )
   {
    GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, true, 1, 1 >(
       gridPointer,
       CoordinatesType( 0, 0 ),
       gridPointer->getDimensions() - CoordinatesType( 1, 1 ),
       userData,
       asynchronousMode,
       0 );
   }
   else //Distributed
   {
      const CoordinatesType begin = distributedGrid->getLowerOverlap();
      const CoordinatesType end = gridPointer->getDimensions() - distributedGrid->getUpperOverlap() -
                                  CoordinatesType( 1, 1 );
      const int* neighbors=distributedGrid->getNeighbors(); 
      if( neighbors[ Meshes::DistributedMeshes::ZzYzXm ] == -1 )
      {
         GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
            gridPointer,
            begin,
            CoordinatesType( begin.x(), end.y() ),
            userData,
            asynchronousMode,
            0 );
      }
       
      if(neighbors[Meshes::DistributedMeshes::ZzYzXp]==-1)
      {
         GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
            gridPointer,
            CoordinatesType( end.x(), begin.y() ),
            end,
            userData,
            asynchronousMode,
            0 );
      }
       
      if( neighbors[ Meshes::DistributedMeshes::ZzYmXz ] == -1 )
      {
         GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
            gridPointer,
            begin,
            CoordinatesType( end.x(), begin.y() ),
            userData,
            asynchronousMode,
            0 );
      }
       
      if( neighbors[ Meshes::DistributedMeshes::ZzYpXz ] == -1 )
      {
         GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
            gridPointer,
            CoordinatesType( begin.x(), end.y() ),
            end,
            userData,
            asynchronousMode,
            0 );
      }
   }
}

template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
   template< typename EntitiesProcessor,
             typename UserData >
void
Traverser< Meshes::Grid< 2, Real, Device, Index >, GridEntity, 2 >::
processInteriorEntities( const GridPointer& gridPointer,
                         UserData& userData ) const
{
   /****
    * Interior cells
    */
   static_assert( GridEntity::getEntityDimension() == 2, "The entity has wrong dimensions." );
   
   DistributedGridType* distributedGrid = gridPointer->getDistributedMesh();
   if( distributedGrid == nullptr || !distributedGrid->isDistributed() )
   {
      GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
         gridPointer,
         CoordinatesType( 1, 1 ),
         gridPointer->getDimensions() - CoordinatesType( 2, 2 ),
         userData,
         asynchronousMode,
         0 );
   }
   else // distributed
   {
      const int* neighbors = distributedGrid->getNeighbors(); 
      CoordinatesType begin( distributedGrid->getLowerOverlap());
      CoordinatesType end( gridPointer->getDimensions() - distributedGrid->getUpperOverlap()- CoordinatesType(1,1) );
      if( neighbors[ Meshes::DistributedMeshes::ZzYzXm ] == -1 )
         begin.x() += 1 ;
       
      if( neighbors[ Meshes::DistributedMeshes::ZzYzXp ] == -1 )
         end.x() -= 1;
       
      if( neighbors[ Meshes::DistributedMeshes::ZzYmXz ] == -1 )
         begin.y() += 1 ;
       
      if( neighbors[ Meshes::DistributedMeshes::ZzYpXz ] == -1 )
         end.y() -= 1;
      
       
      GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
         gridPointer,
         begin,
         end,
         userData,
         asynchronousMode,
         0);
   }
}

template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
   template< typename EntitiesProcessor,
             typename UserData >
void
Traverser< Meshes::Grid< 2, Real, Device, Index >, GridEntity, 2 >::
processAllEntities( const GridPointer& gridPointer,
                    UserData& userData ) const
{
   /****
    * All cells
    */
   static_assert( GridEntity::getEntityDimension() == 2, "The entity has wrong dimension." );
 
   DistributedGridType* distributedGrid = gridPointer->getDistributedMesh();
   if( distributedGrid == nullptr || !distributedGrid->isDistributed() )
   {
      GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
         gridPointer,
         CoordinatesType( 0, 0 ),
         gridPointer->getDimensions() - CoordinatesType( 1, 1 ),
         userData,
         asynchronousMode,
         0 );
   }
   else
   {
       //const int* neighbors=distributedGrid->getNeighbors(); 
       CoordinatesType begin( distributedGrid->getLowerOverlap() );
       CoordinatesType end( gridPointer->getDimensions() - distributedGrid->getUpperOverlap()- CoordinatesType(1,1) );
       
       GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
          gridPointer,
          begin,
          end,
          userData,
          asynchronousMode,
          0);   
   }
}

/****
 * Grid 2D, faces
 */
template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
   template< typename EntitiesProcessor,
             typename UserData >
void
Traverser< Meshes::Grid< 2, Real, Device, Index >, GridEntity, 1 >::
processBoundaryEntities( const GridPointer& gridPointer,
                         UserData& userData ) const
{
   /****
    * Boundary faces
    */
   static_assert( GridEntity::getEntityDimension() == 1, "The entity has wrong dimension." );
 
   GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, true, 1, 0, CoordinatesType, CoordinatesType >(
      gridPointer,
      CoordinatesType( 0, 0 ),
      gridPointer->getDimensions() - CoordinatesType( 0, 1 ),
      userData,
      asynchronousMode,
      1,
      CoordinatesType( 1, 0 ),
      CoordinatesType( 0, 1 ) );

   GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, true, 0, 1, CoordinatesType, CoordinatesType >(
      gridPointer,
      CoordinatesType( 0, 0 ),
      gridPointer->getDimensions() - CoordinatesType( 1, 0 ),
      userData,
      asynchronousMode,
      0,
      CoordinatesType( 0, 1 ),
      CoordinatesType( 1, 0 ) );
}

template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
   template< typename EntitiesProcessor,
             typename UserData >
void
Traverser< Meshes::Grid< 2, Real, Device, Index >, GridEntity, 1 >::
processInteriorEntities( const GridPointer& gridPointer,
                         UserData& userData ) const
{
   /****
    * Interior faces
    */
   static_assert( GridEntity::getEntityDimension() == 1, "The entity has wrong dimension." );
 
   GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false, 1, 1, CoordinatesType, CoordinatesType >(
      gridPointer,
      CoordinatesType( 1, 0 ),
      gridPointer->getDimensions() - CoordinatesType( 1, 1 ),
      userData,
      asynchronousMode,
      1,
      CoordinatesType( 1, 0 ),
      CoordinatesType( 0, 1 ) );

   GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false, 1, 1, CoordinatesType, CoordinatesType >(
      gridPointer,
      CoordinatesType( 0, 1 ),
      gridPointer->getDimensions() - CoordinatesType( 1, 1 ),
      userData,
      asynchronousMode,
      0,
      CoordinatesType( 0, 1 ),
      CoordinatesType( 1, 0 ) );
}

template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
   template< typename EntitiesProcessor,
             typename UserData >
void
Traverser< Meshes::Grid< 2, Real, Device, Index >, GridEntity, 1 >::
processAllEntities( const GridPointer& gridPointer,
                    UserData& userData ) const
{
   /****
    * All faces
    */
   static_assert( GridEntity::getEntityDimension() == 1, "The entity has wrong dimension." );
 
   GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false, 1, 1, CoordinatesType, CoordinatesType >(
      gridPointer,
      CoordinatesType( 0, 0 ),
      gridPointer->getDimensions() - CoordinatesType( 0, 1 ),
      userData,
      asynchronousMode,
      1,
      CoordinatesType( 1, 0 ),
      CoordinatesType( 0, 1 ) );

   GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false, 1, 1, CoordinatesType, CoordinatesType >(
      gridPointer,
      CoordinatesType( 0, 0 ),
      gridPointer->getDimensions() - CoordinatesType( 1, 0 ),
      userData,
      asynchronousMode,
      0,
      CoordinatesType( 0, 1 ),
      CoordinatesType( 1, 0 ) );
}

template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
   template< typename EntitiesProcessor,
             typename UserData >
void
Traverser< Meshes::Grid< 2, Real, Device, Index >, GridEntity, 0 >::
processBoundaryEntities( const GridPointer& gridPointer,
                         UserData& userData ) const
{
   /****
    * Boundary vertices
    */
   static_assert( GridEntity::getEntityDimension() == 0, "The entity has wrong dimension." );
 
   GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, true, 1, 1 >(
      gridPointer,
      CoordinatesType( 0, 0 ),
      gridPointer->getDimensions(),
      userData,
      asynchronousMode,
      0 );
}

template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
   template< typename EntitiesProcessor,
             typename UserData >
void
Traverser< Meshes::Grid< 2, Real, Device, Index >, GridEntity, 0 >::
processInteriorEntities( const GridPointer& gridPointer,
                         UserData& userData ) const
{
   /****
    * Interior vertices
    */
   static_assert( GridEntity::getEntityDimension() == 0, "The entity has wrong dimension." );
 
   GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
      gridPointer,
      CoordinatesType( 1, 1 ),
      gridPointer->getDimensions() - CoordinatesType( 1, 1 ),
      userData,
      asynchronousMode,
      0 );
}
 
template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
   template< typename EntitiesProcessor,
             typename UserData >
void
Traverser< Meshes::Grid< 2, Real, Device, Index >, GridEntity, 0 >::
processAllEntities( const GridPointer& gridPointer,
                    UserData& userData ) const
{
   /****
    * All vertices
    */
   static_assert( GridEntity::getEntityDimension() == 0, "The entity has wrong dimension." );
 
   GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
      gridPointer,
      CoordinatesType( 0, 0 ),
      gridPointer->getDimensions(),
      userData,
      asynchronousMode,
      0 );
}

} // namespace Meshes
} // namespace TNL
