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
#include <TNL/Meshes/DistributedGrid.h>


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
                         SharedPointer< UserData, DeviceType >& userDataPointer ) const
{
   /****
    * Boundary cells
    */
   static_assert( GridEntity::entityDimension == 1, "The entity has wrong dimensions." );
   
   auto distributedgrid=gridPointer->GetDistGrid();
   if(distributedgrid==nullptr||!distributedgrid->isMPIUsed())
   {
        GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, true >(
           gridPointer,
           CoordinatesType( 0 ),
           gridPointer->getDimensions() - CoordinatesType( 1 ),
           userDataPointer );
   }
   else
   {
       //MPI
#ifdef HAVE_MPI
       if(distributedgrid->getLeft()==-1)
       {
          GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
              gridPointer,
              CoordinatesType( 0 ),
              CoordinatesType( 0 ),
              userDataPointer );
       }
       
       if(distributedgrid->getRight()==-1)
       {
          GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
              gridPointer,
              gridPointer->getDimensions() - CoordinatesType( 1 ),
              gridPointer->getDimensions() - CoordinatesType( 1 ),
              userDataPointer );
       }
#endif
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
                         SharedPointer< UserData, DeviceType >& userDataPointer ) const
{
   /****
    * Interior cells
    */
   static_assert( GridEntity::entityDimension == 1, "The entity has wrong dimensions." );

   auto distributedgrid=gridPointer->GetDistGrid();
   if(distributedgrid==nullptr||!distributedgrid->isMPIUsed())
   {
        GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
           gridPointer,
           CoordinatesType( 1 ),
           gridPointer->getDimensions() - CoordinatesType( 2 ),
           userDataPointer );   
   }
   else
   {
       //MPI
#ifdef HAVE_MPI
       CoordinatesType begin( distributedgrid->getOverlap().x() );
       CoordinatesType end( gridPointer->getDimensions() - distributedgrid->getOverlap().x()-1 );
       if(distributedgrid->getLeft()==-1)
       {
           begin=CoordinatesType( 1 );
       }
       
       if(distributedgrid->getRight()==-1)
       {
           begin=gridPointer->getDimensions() - CoordinatesType( 2 );
       }
       
       GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
          gridPointer,
          begin,
          end,
          userDataPointer );
#endif
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
   SharedPointer< UserData, DeviceType >& userDataPointer ) const
{
   /****
    * All cells
    */

   static_assert( GridEntity::entityDimension == 1, "The entity has wrong dimensions." );
   
   auto distributedgrid=gridPointer->GetDistGrid();
   if(distributedgrid==nullptr||!distributedgrid->isMPIUsed())
   {
        GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
           gridPointer,
           CoordinatesType( 0 ),
           gridPointer->getDimensions() - CoordinatesType( 1 ),
           userDataPointer );  
   }
   else
   {
       //MPI
       #ifdef HAVE_MPI
       CoordinatesType begin( distributedgrid->getOverlap().x() );
       CoordinatesType end( gridPointer->getDimensions() - distributedgrid->getOverlap().x()-1 );
       if(distributedgrid->getLeft()==-1)
       {
           begin=CoordinatesType( 0 );
       }
       
       if(distributedgrid->getRight()==-1)
       {
           end=gridPointer->getDimensions() - CoordinatesType( 1 );
       }
       
       GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
          gridPointer,
          begin,
          end,
          userDataPointer );
        #endif
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
                         SharedPointer< UserData, DeviceType >& userDataPointer ) const
{
   /****
    * Boundary vertices
    */
   static_assert( GridEntity::entityDimension == 0, "The entity has wrong dimensions." );

   GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, true >(
      gridPointer,
      CoordinatesType( 0 ),
      gridPointer->getDimensions(),
      userDataPointer );
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
                         SharedPointer< UserData, DeviceType >& userDataPointer ) const
{
   /****
    * Interior vertices
    */
   static_assert( GridEntity::entityDimension == 0, "The entity has wrong dimensions." );

   GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
      gridPointer,
      CoordinatesType( 1 ),
      gridPointer->getDimensions() - CoordinatesType( 1 ),
      userDataPointer );
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
   SharedPointer< UserData, DeviceType >& userDataPointer ) const
{
   /****
    * All vertices
    */
   static_assert( GridEntity::entityDimension == 0, "The entity has wrong dimensions." );

   GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
      gridPointer,
      CoordinatesType( 0 ),
      gridPointer->getDimensions(),
      userDataPointer );
}

} // namespace Meshes
} // namespace TNL
