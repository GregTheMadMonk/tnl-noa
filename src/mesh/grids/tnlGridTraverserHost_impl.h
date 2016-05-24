/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   tnlGridTraverserHost_impl.h
 * Author: hanouvit
 *
 * Created on 16. května 2016, 15:10
 */

#ifndef TNLGRIDTRAVERSERHOST_IMPL_H
#define TNLGRIDTRAVERSERHOST_IMPL_H

/****
 * 1D traverser, host
 */
template< typename Real,           
          typename Index >      
   template<
      typename GridEntity,
      typename EntitiesProcessor,
      typename UserData,
      bool processOnlyBoundaryEntities >
void
tnlGridTraverser< tnlGrid< 1, Real, tnlHost, Index > >::
processEntities(
   const GridType& grid,
   const CoordinatesType& begin,
   const CoordinatesType& end,
   const CoordinatesType& entityOrientation,
   const CoordinatesType& entityBasis,   
   UserData& userData )
{

   
   GridEntity entity( grid );
   entity.setOrientation( entityOrientation );
   entity.setBasis( entityBasis );
   if( processOnlyBoundaryEntities )
   {
      entity.getCoordinates() = begin;
      entity.refresh();
      EntitiesProcessor::processEntity( entity.getMesh(), userData, entity );
      entity.getCoordinates() = end;
      entity.refresh();
      EntitiesProcessor::processEntity( entity.getMesh(), userData, entity );
   }
   else
   {
      for( entity.getCoordinates().x() = begin.x();
           entity.getCoordinates().x() <= end.x();
           entity.getCoordinates().x() ++ )
      {
         entity.refresh();
         EntitiesProcessor::processEntity( entity.getMesh(), userData, entity );
      }
   }
}

/****
 * 2D traverser, host
 */
template< typename Real,           
          typename Index >      
   template<
      typename GridEntity,
      typename EntitiesProcessor,
      typename UserData,
      bool processOnlyBoundaryEntities,
      int XOrthogonalBoundary,
      int YOrthogonalBoundary >
void
tnlGridTraverser< tnlGrid< 2, Real, tnlHost, Index > >::
processEntities(
   const GridType& grid,
   const CoordinatesType begin,
   const CoordinatesType end,
   const CoordinatesType& entityOrientation,
   const CoordinatesType& entityBasis,      
   UserData& userData )
{
   GridEntity entity( grid );
   entity.setOrientation( entityOrientation );
   entity.setBasis( entityBasis );

   if( processOnlyBoundaryEntities )
   {
      if( YOrthogonalBoundary )
         for( entity.getCoordinates().x() = begin.x();
              entity.getCoordinates().x() <= end.x();
              entity.getCoordinates().x() ++ )
         {
            entity.getCoordinates().y() = begin.y();
            entity.refresh();
            EntitiesProcessor::processEntity( entity.getMesh(), userData, entity );
            entity.getCoordinates().y() = end.y();
            entity.refresh();
            EntitiesProcessor::processEntity( entity.getMesh(), userData, entity );
         }
      if( XOrthogonalBoundary )
         for( entity.getCoordinates().y() = begin.y();
              entity.getCoordinates().y() <= end.y();
              entity.getCoordinates().y() ++ )
         {
            entity.getCoordinates().x() = begin.x();
            entity.refresh();
            EntitiesProcessor::processEntity( entity.getMesh(), userData, entity );
            entity.getCoordinates().x() = end.x();
            entity.refresh();
            EntitiesProcessor::processEntity( entity.getMesh(), userData, entity );
         }
   }
   else
   {
//#pragma omp parallel for firstprivate( entity, begin, end ) if( tnlHost::isOMPEnabled() )      
      for( entity.getCoordinates().y() = begin.y();
           entity.getCoordinates().y() <= end.y();
           entity.getCoordinates().y() ++ )
         for( entity.getCoordinates().x() = begin.x();
              entity.getCoordinates().x() <= end.x();
              entity.getCoordinates().x() ++ )
         {
            entity.refresh();
            EntitiesProcessor::processEntity( entity.getMesh(), userData, entity );
         }
   }
}

/****
 * 3D traverser, host
 */
template< typename Real,           
          typename Index >      
   template<
      typename GridEntity,
      typename EntitiesProcessor,
      typename UserData,
      bool processOnlyBoundaryEntities,
      int XOrthogonalBoundary,
      int YOrthogonalBoundary,
      int ZOrthogonalBoundary >
void
tnlGridTraverser< tnlGrid< 3, Real, tnlHost, Index > >::
processEntities(
   const GridType& grid,
   const CoordinatesType& begin,
   const CoordinatesType& end,
   const CoordinatesType& entityOrientation,
   const CoordinatesType& entityBasis,      
   UserData& userData )
{
   GridEntity entity( grid );
   entity.setOrientation( entityOrientation );
   entity.setBasis( entityBasis );

   if( processOnlyBoundaryEntities )
   {
      if( ZOrthogonalBoundary )
         for( entity.getCoordinates().y() = begin.y();
              entity.getCoordinates().y() <= end.y();
              entity.getCoordinates().y() ++ )
            for( entity.getCoordinates().x() = begin.x();
                 entity.getCoordinates().x() <= end.x();
                 entity.getCoordinates().x() ++ )
            {
               entity.getCoordinates().z() = begin.z();
               entity.refresh();
               EntitiesProcessor::processEntity( entity.getMesh(), userData, entity );
               entity.getCoordinates().z() = end.z();
               entity.refresh();
               EntitiesProcessor::processEntity( entity.getMesh(), userData, entity );
            }
      if( YOrthogonalBoundary )
         for( entity.getCoordinates().z() = begin.z();
                 entity.getCoordinates().z() <= end.z();
                 entity.getCoordinates().z() ++ )
            for( entity.getCoordinates().x() = begin.x();
                 entity.getCoordinates().x() <= end.x();
                 entity.getCoordinates().x() ++ )
            {
               entity.getCoordinates().y() = begin.y();
               entity.refresh();
               EntitiesProcessor::processEntity( entity.getMesh(), userData, entity );
               entity.getCoordinates().y() = end.y();
               entity.refresh();
               EntitiesProcessor::processEntity( entity.getMesh(), userData, entity );
            }
      if( XOrthogonalBoundary )
         for( entity.getCoordinates().z() = begin.z();
              entity.getCoordinates().z() <= end.z();
              entity.getCoordinates().z() ++ )
            for( entity.getCoordinates().y() = begin.y();
                 entity.getCoordinates().y() <= end.y();
                 entity.getCoordinates().y() ++ )
            {
               entity.getCoordinates().x() = begin.x();
               entity.refresh();
               EntitiesProcessor::processEntity( entity.getMesh(), userData, entity );
               entity.getCoordinates().x() = end.x();
               entity.refresh();
               EntitiesProcessor::processEntity( entity.getMesh(), userData, entity );
            }
   }
   else
   {
      for( entity.getCoordinates().z() = begin.z();
           entity.getCoordinates().z() <= end.z();
           entity.getCoordinates().z() ++ )
         for( entity.getCoordinates().y() = begin.y();
              entity.getCoordinates().y() <= end.y();
              entity.getCoordinates().y() ++ )
            for( entity.getCoordinates().x() = begin.x();
                 entity.getCoordinates().x() <= end.x();
                 entity.getCoordinates().x() ++ )
            {
               entity.refresh();
               EntitiesProcessor::processEntity( entity.getMesh(), userData, entity );
            }
   }
}

#endif /* TNLGRIDTRAVERSERHOST_IMPL_H */

