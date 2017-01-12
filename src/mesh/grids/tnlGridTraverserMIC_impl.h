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

#include "tnlGridTraverser.h"

#include <stdint.h>
#include <core/tnlMIC.h>

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
#define USE_MICSTRUCT
    
#ifdef USE_MICSTRUCT
    TNLMICSTRUCT(begin, const CoordinatesType);
    TNLMICSTRUCT(end, const CoordinatesType);
    TNLMICSTRUCT(entityOrientation,const CoordinatesType);
    TNLMICSTRUCT(entityBasis, const CoordinatesType);
    TNLMICSTRUCT(grid,const GridType);
    TNLMICSTRUCT(userData,UserData);
    
    #pragma offload target(mic) in(sbegin,send,sentityBasis,sentityOrientation,sgrid,suserData)
{
        
    TNLMICSTRUCTUSE(begin, const CoordinatesType);
    TNLMICSTRUCTUSE(end, const CoordinatesType);
    TNLMICSTRUCTUSE(entityOrientation,const CoordinatesType);
    TNLMICSTRUCTUSE(entityBasis, const CoordinatesType);
    TNLMICSTRUCTUSE(grid,const GridType);
    TNLMICSTRUCTUSE(userData,UserData);  
#endif

#ifdef USE_MICHIDE
   uint8_t * ubegin=(uint8_t*)&begin;
   uint8_t * uend=(uint8_t*)&end;
   uint8_t * uentityOrientation=(uint8_t*)&entityOrientation;
   uint8_t * ugrid=(uint8_t*)&grid;
   uint8_t * uentityBasis=(uint8_t*)&entityBasis;
   uint8_t * uuserData=(uint8_t*)&userData;
   
#pragma offload target(mic) in(ubegin:length(sizeof(CoordinatesType))) in(uend:length(sizeof(CoordinatesType))) in(uentityBasis:length(sizeof(CoordinatesType))) in(uentityOrientation:length(sizeof(CoordinatesType))) in(ugrid:length(sizeof(typename GridEntity::MeshType))) in(uuserData:length(sizeof(UserData)))
{
   typename GridEntity::MeshType * kernelgrid = (typename GridEntity::MeshType*) ugrid; 
   CoordinatesType* kernelbegin = (CoordinatesType*) ubegin;
   CoordinatesType* kernelend = (CoordinatesType*) uend ;
   CoordinatesType* kernelentityOrientation = (CoordinatesType*) uentityOrientation ;
   CoordinatesType* kernelentityBasis = (CoordinatesType*) uentityBasis;
   UserData* kerneluserData = (UserData*)uuserData;
  // typename GridEntity::MeshType * kernelgrid = (typename GridEntity::MeshType*) ugird; 

#endif
    
              
#pragma omp parallel firstprivate( kernelbegin, kernelend )
{       
    //    cout << "HOVNO" <<endl;
   GridEntity entity( *kernelgrid );
   entity.setOrientation( *kernelentityOrientation );
   entity.setBasis( *kernelentityBasis );

   if( processOnlyBoundaryEntities )
   {
       
      if( YOrthogonalBoundary )
      #pragma omp for
         for( auto k = kernelbegin->x();
              k <= kernelend->x();
              k ++ )
         {          
            entity.getCoordinates().x()=k; 
            entity.getCoordinates().y() = kernelbegin->y();
            entity.refresh();
            EntitiesProcessor::processEntity( entity.getMesh(), *(kerneluserData), entity );
            entity.getCoordinates().y() = kernelend->y();
            entity.refresh();
            EntitiesProcessor::processEntity( entity.getMesh(), *(kerneluserData), entity );
         }
     
     if( XOrthogonalBoundary )
     #pragma omp for
         for( auto k = kernelbegin->y();
              k <= kernelend->y();
              k ++ )
         {
            entity.getCoordinates().y() = k; 
            entity.getCoordinates().x() = kernelbegin->x();
            entity.refresh();
            EntitiesProcessor::processEntity( entity.getMesh(), *(kerneluserData), entity );
            entity.getCoordinates().x() = kernelend->x();
            entity.refresh();
            EntitiesProcessor::processEntity( entity.getMesh(), *(kerneluserData), entity );
         }
   }
   else
   {
#pragma omp for
      for(  auto k = kernelbegin->y();
           k <= kernelend->y();
           k ++ )
         for( entity.getCoordinates().x() = kernelbegin->x();
              entity.getCoordinates().x() <= kernelend->x();
              entity.getCoordinates().x() ++ )
         {
            entity.getCoordinates().y()=k;
            entity.refresh();
            EntitiesProcessor::processEntity( entity.getMesh(), *(kerneluserData), entity );
         }
   }
}
   
}   
 
   //like CUDA
   /*satanHider< const CoordinatesType >  kernelBegin;   
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
   kernelUserData.pointer = tnlMIC::passToDevice( userData );*/   
   
/*#pragma offload target(mic) in(kernelBegin,kernelEnd,kernelEntityOrientation, kernelEntityBasis,kernelGrid,kernelUserData)
{
 
#pragma omp parallel firstprivate( kernelBegin, kernelEnd )
{       
   GridEntity entity( *(kernelGrid.pointer) );
   entity.setOrientation( *kernelEntityOrientation.pointer );
   entity.setBasis( *kernelEntityBasis.pointer );

   if( processOnlyBoundaryEntities )
   {
       
      if( YOrthogonalBoundary )
      #pragma omp for
         for( auto k = kernelBegin.pointer->x();
              k <= kernelEnd.pointer->x();
              k ++ )
         {          
            entity.getCoordinates().x()=k; 
            entity.getCoordinates().y() = kernelBegin.pointer->y();
            entity.refresh();
            EntitiesProcessor::processEntity( entity.getMesh(), *(kernelUserData.pointer), entity );
            entity.getCoordinates().y() = kernelEnd.pointer->y();
            entity.refresh();
            EntitiesProcessor::processEntity( entity.getMesh(), *(kernelUserData.pointer), entity );
         }
     
     if( XOrthogonalBoundary )
     #pragma omp for
         for( auto k = kernelBegin.pointer->y();
              k <= kernelEnd.pointer->y();
              k ++ )
         {
            entity.getCoordinates().y() = k; 
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
#pragma omp for
      for(  auto k = kernelBegin.pointer->y();
           k <= kernelEnd.pointer->y();
           k ++ )
         for( entity.getCoordinates().x() = kernelBegin.pointer->x();
              entity.getCoordinates().x() <= kernelEnd.pointer->x();
              entity.getCoordinates().x() ++ )
         {
            entity.getCoordinates().y()=k;
            entity.refresh();
            EntitiesProcessor::processEntity( entity.getMesh(), *(kernelUserData.pointer), entity );
         }
   }
}   
   
   
}     
   tnlMIC::freeFromDevice( kernelGrid.pointer );
   tnlMIC::freeFromDevice( kernelBegin.pointer );
   tnlMIC::freeFromDevice( kernelEnd.pointer );
   tnlMIC::freeFromDevice( kernelEntityOrientation.pointer );
   tnlMIC::freeFromDevice( kernelEntityBasis.pointer );
   tnlMIC::freeFromDevice( kernelUserData.pointer );
   
*/        

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

