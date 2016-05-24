/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   tnlGridTraverserMIC_impl.h
 * Author: hanouvit
 *
 * Created on 16. května 2016, 15:16
 */

#ifndef TNLGRIDTRAVERSERMIC_IMPL_H
#define TNLGRIDTRAVERSERMIC_IMPL_H

/****
 * 1D traverser, MIC
 */
template< typename Real,           
          typename Index >      
   template<
      typename GridEntity,
      typename EntitiesProcessor,
      typename UserData,
      bool processOnlyBoundaryEntities >
void
tnlGridTraverser< tnlGrid< 1, Real, tnlMIC, Index > >::
processEntities(
   const GridType& grid,
   const CoordinatesType& begin,
   const CoordinatesType& end,
   const CoordinatesType& entityOrientation,
   const CoordinatesType& entityBasis,   
   UserData& userData )
{
/*
   
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
 */
    cout << "Not Implemented YET - 1D Traversar MIC" <<endl;
}

/****
 * 2D traverser, MIC
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
tnlGridTraverser< tnlGrid< 2, Real, tnlMIC, Index > >::
processEntities(
   const GridType& grid,
   const CoordinatesType begin,
   const CoordinatesType end,
   const CoordinatesType& entityOrientation,
   const CoordinatesType& entityBasis,      
   UserData& userData )
{
   //like CUDA
   satanHider< const CoordinatesType >  kernelBegin;
   kernelBegin.pointer = tnlMIC::passToDevice( begin );
   satanHider< const CoordinatesType >  kernelEnd;
   kernelEnd.pointer = tnlMIC::passToDevice( end );
   satanHider< const CoordinatesType > kernelEntityOrientation;
   kernelEntityOrientation.pointer = tnlMIC::passToDevice( entityOrientation );
   satanHider< const CoordinatesType > kernelEntityBasis;
   kernelEntityBasis.pointer = tnlMIC::passToDevice( entityBasis );
   satanHider< const typename GridEntity::MeshType > kernelGrid;
   kernelGrid.pointer  = tnlMIC::passToDevice( grid );
   satanHider< UserData > kernelUserData;
   kernelUserData.pointer = tnlMIC::passToDevice( userData );
   
#pragma offload target(mic) in(kernelBegin,kernelEnd,kernelEntityOrientation, kernelEntityBasis,kernelGrid,kernelUserData)
{
 
       
   //Like Host
   GridEntity entity( *(kernelGrid.pointer) );
   entity.setOrientation( *kernelEntityOrientation.pointer );
   entity.setBasis( *kernelEntityBasis.pointer );

   if( processOnlyBoundaryEntities )
   {
       
      if( YOrthogonalBoundary )
         for( entity.getCoordinates().x() = kernelBegin.pointer->x();
              entity.getCoordinates().x() <= kernelEnd.pointer->x();
              entity.getCoordinates().x() ++ )
         {            
            entity.getCoordinates().y() = kernelBegin.pointer->y();
            entity.refresh();
            EntitiesProcessor::processEntity( entity.getMesh(), *(kernelUserData.pointer), entity );
            entity.getCoordinates().y() = kernelEnd.pointer->y();
            entity.refresh();
            EntitiesProcessor::processEntity( entity.getMesh(), *(kernelUserData.pointer), entity );
         }
     
     if( XOrthogonalBoundary )
         for( entity.getCoordinates().y() = kernelBegin.pointer->y();
              entity.getCoordinates().y() <= kernelEnd.pointer->y();
              entity.getCoordinates().y() ++ )
         {
            entity.getCoordinates().x() = kernelBegin.pointer->x();
            entity.refresh();
            EntitiesProcessor::processEntity( entity.getMesh(), *(kernelUserData.pointer), entity );
            entity.getCoordinates().x() = kernelEnd.pointer->x();
            entity.refresh();
            EntitiesProcessor::processEntity( entity.getMesh(), *(kernelUserData.pointer), entity );
         }
   }
   else
   {
//#pragma omp parallel for firstprivate( entity, begin, end ) if( tnlHost::isOMPEnabled() )      
      for( entity.getCoordinates().y() = kernelBegin.pointer->y();
           entity.getCoordinates().y() <= kernelEnd.pointer->y();
           entity.getCoordinates().y() ++ )
         for( entity.getCoordinates().x() = kernelBegin.pointer->x();
              entity.getCoordinates().x() <= kernelEnd.pointer->x();
              entity.getCoordinates().x() ++ )
         {
            entity.refresh();
            EntitiesProcessor::processEntity( entity.getMesh(), *(kernelUserData.pointer), entity );
         }
   }
   
   
   
    }   
   
   tnlMIC::freeFromDevice( kernelGrid.pointer );
   tnlMIC::freeFromDevice( kernelBegin.pointer );
   tnlMIC::freeFromDevice( kernelEnd.pointer );
   tnlMIC::freeFromDevice( kernelEntityOrientation.pointer );
   tnlMIC::freeFromDevice( kernelEntityBasis.pointer );
   tnlMIC::freeFromDevice( kernelUserData.pointer );
   
        

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
tnlGridTraverser< tnlGrid< 3, Real, tnlMIC, Index > >::
processEntities(
   const GridType& grid,
   const CoordinatesType& begin,
   const CoordinatesType& end,
   const CoordinatesType& entityOrientation,
   const CoordinatesType& entityBasis,      
   UserData& userData )
{
/*   GridEntity entity( grid );
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
   }*/
    
    cout << "Not Implemented YET - 3D Traversar MIC" <<endl;

    
}

#endif /* TNLGRIDTRAVERSERMIC_IMPL_H */

