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
#include <TNL/Meshes/DistributedMeshes/DistributedMesh.h>


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
   static_assert( GridEntity::getEntityDimension() == 1, "The entity has wrong dimensions." );
   
   DistributedGridType* distributedGrid = gridPointer->getDistributedMesh();
   if( distributedGrid == nullptr || ! distributedGrid->isDistributed() )
   {
        GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, true >(
           gridPointer,
           CoordinatesType( 0 ),
           gridPointer->getDimensions() - CoordinatesType( 1 ),
           userData,
           asynchronousMode );
   }
   else //Distributed
   {
       const int* neighbors=distributedGrid->getNeighbors(); 
       if( neighbors[ Meshes::DistributedMeshes::ZzYzXm ] == -1 )
       {
          GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
              gridPointer,
              CoordinatesType( 0 ) + distributedGrid->getLowerOverlap(),
              CoordinatesType( 0 ) + distributedGrid->getLowerOverlap(),
              userData,
              asynchronousMode );
       }
       
       if( neighbors[ Meshes::DistributedMeshes::ZzYzXp ] == -1 )
       {
          GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
              gridPointer,
              gridPointer->getDimensions() - CoordinatesType( 1 ) - distributedGrid->getUpperOverlap(),
              gridPointer->getDimensions() - CoordinatesType( 1 ) - distributedGrid->getUpperOverlap(),
              userData,
              asynchronousMode );
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
Traverser< Meshes::Grid< 1, Real, Device, Index >, GridEntity, 1 >::
processInteriorEntities( const GridPointer& gridPointer,
                         UserData& userData ) const
{
   /****
    * Interior cells
    */
   static_assert( GridEntity::getEntityDimension() == 1, "The entity has wrong dimension." );

   DistributedGridType* distributedGrid = gridPointer->getDistributedMesh();
   if( distributedGrid == nullptr || !distributedGrid->isDistributed() )
   {
        GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
           gridPointer,
           CoordinatesType( 1 ),
           gridPointer->getDimensions() - CoordinatesType( 2 ),
           userData,
           asynchronousMode );
   }
   else //Distributed
   {
      CoordinatesType begin( distributedGrid->getLowerOverlap() );
      CoordinatesType end( gridPointer->getDimensions() - distributedGrid->getUpperOverlap() - CoordinatesType( 1 ) );
      
      const int* neighbors = distributedGrid->getNeighbors(); 
      if( neighbors[ Meshes::DistributedMeshes::ZzYzXm ] == -1 )
         begin += CoordinatesType( 1 );
       
      if( neighbors[ Meshes::DistributedMeshes::ZzYzXp ] == -1 )
         end -= CoordinatesType( 1 );
      
      /*
         TNL_MPI_PRINT( " lowerOverlap = " << distributedGrid->getLowerOverlap() << 
               " upperOverlap = " << distributedGrid->getUpperOverlap() <<
               " gridPointer->getDimensions() = " << gridPointer->getDimensions() <<
               "begin = " << begin << " end = " << end);
       */

       GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
          gridPointer,
          begin,
          end,
          userData,
          asynchronousMode );
   }
   
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
   static_assert( GridEntity::getEntityDimension() == 1, "The entity has wrong dimensions." );
   
   DistributedGridType* distributedGrid = gridPointer->getDistributedMesh();
   if( distributedGrid == nullptr || !distributedGrid->isDistributed() )
   {
        GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
           gridPointer,
           CoordinatesType( 0 ),
           gridPointer->getDimensions() - CoordinatesType( 1 ),
           userData,
           asynchronousMode );
   }
   else //Distributed
   {
       CoordinatesType begin( distributedGrid->getLowerOverlap() );
       CoordinatesType end( gridPointer->getDimensions() - distributedGrid->getUpperOverlap() - 1 );
       
       GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
          gridPointer,
          begin,
          end,
          userData,
          asynchronousMode );
   }

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
   static_assert( GridEntity::getEntityDimension() == 0, "The entity has wrong dimension." );

   GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, true >(
      gridPointer,
      CoordinatesType( 0 ),
      gridPointer->getDimensions(),
      userData,
      asynchronousMode );
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
   static_assert( GridEntity::getEntityDimension() == 0, "The entity has wrong dimension." );

   GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
      gridPointer,
      CoordinatesType( 1 ),
      gridPointer->getDimensions() - CoordinatesType( 1 ),
      userData,
      asynchronousMode );
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
   static_assert( GridEntity::getEntityDimension() == 0, "The entity has wrong dimension." );

   GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
      gridPointer,
      CoordinatesType( 0 ),
      gridPointer->getDimensions(),
      userData,
      asynchronousMode );
}

} // namespace Meshes
} // namespace TNL
