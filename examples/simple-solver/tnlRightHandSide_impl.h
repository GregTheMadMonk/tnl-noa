#ifndef TNLRIGHTHANDSIDE_IMPL_H
#define	TNLRIGHTHANDSIDE_IMPL_H

#include "tnlRightHandSide.h"

template<typename Real, typename Device, typename Index>
template<typename AnalyticSpaceFunction, typename TimeFunction>
void tnlRightHandSide<tnlGrid<1,Real,Device,Index,tnlIdenticalGridGeometry>>::
applyRHSValues(const MeshType& mesh, const RealType& time, DofVectorType& _fu,
               TimeFunction& timeFunction, AnalyticSpaceFunction& analyticSpaceFunction)
{        
      VertexType vertex;
      CoordinatesType coordinaates;

      #ifdef HAVE_OPENMP
         #pragma omp parallel for private(coordinates,vertex)
      #endif
      for(IndexType i=1; i<(_fu.getSize()-1); i++)
      {         
         mesh.getElementCoordinates(i,coordinates);
         mesh.getElementCenter(coordinates,vertex);

      _fu[i] += timeFunction.getDerivation(time)*analyticSpaceFunction.getF(vertex) -
                 timeFunction.getTimeValue(time)*analyticSpaceFunction.template getF<2,0,0>(vertex);
   }  
}

template<typename Real, typename Device, typename Index>
template<typename AnalyticSpaceFunction, typename TimeFunction>
void tnlRightHandSide<tnlGrid<2,Real,Device,Index,tnlIdenticalGridGeometry>>::
applyRHSValues(const MeshType& mesh, const RealType& time, DofVectorType& _fu,
               TimeFunction& timeFunction, AnalyticSpaceFunction& analyticSpaceFunction)
{       
   VertexType vertex;
   CoordinatesType coordinates;
   CoordinatesType dimensions = mesh.getDimensions();

   #ifdef HAVE_OPENMP
    #pragma omp parallel for private(coordinates,vertex)
   #endif
   for(IndexType i=1; i<(dimensions.x()-1); i++)
   {
      for(IndexType j=1; j<(dimensions.y()-1); j++)
      {  
         coordinates.x()=i;
         coordinates.y()=j;
         
         mesh.getElementCenter(coordinates,vertex);
         
         _fu[mesh.getElementIndex(i,j)] += timeFunction.getDerivation(time)*analyticSpaceFunction.getF(vertex)- 
                    timeFunction.getTimeValue(time)*(analyticSpaceFunction.template getF<2,0,0>(vertex)+
                    analyticSpaceFunction.template getF<0,2,0>(vertex));
      } 
   }
}

template<typename Real, typename Device, typename Index>
template<typename AnalyticSpaceFunction, typename TimeFunction>
void tnlRightHandSide<tnlGrid<3,Real,Device,Index,tnlIdenticalGridGeometry>>::
applyRHSValues(const MeshType& mesh, const RealType& time, DofVectorType& _fu, 
               TimeFunction& timeFunction, AnalyticSpaceFunction& analyticSpaceFunction)
{ 
   VertexType vertex;
   CoordinatesType coordinates;
   CoordinatesType dimensions = mesh.getDimensions();

   #ifdef HAVE_OPENMP
    #pragma omp parallel for private(coordinates,vertex)
   #endif
   for(IndexType i=1; i<(dimensions.x()-1); i++)
   {
      for(IndexType j=1; j<(dimensions.y()-1); j++)
      {  
         for(IndexType k=1; k<(dimensions.y()-1); k++)
         {
            coordinates.x()=i;
            coordinates.y()=j;
            coordinates.z()=k;
            
            mesh.getElementCenter(coordinates,vertex);
         
            _fu[mesh.getElementIndex(i,j,k)] += 
                       timeFunction.getDerivation(time)*analyticSpaceFunction.getF(vertex)-
                       timeFunction.getTimeValue(time)*(analyticSpaceFunction.template getF<2,0,0>(vertex)+
                       analyticSpaceFunction.template getF<0,2,0>(vertex)+
                       analyticSpaceFunction.template getF<0,0,2>(vertex));
         }
      } 
   }
}


#endif	/* TNLRIGHTHANDSIDE_IMPL_H */

