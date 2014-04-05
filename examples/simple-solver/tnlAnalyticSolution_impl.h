#ifndef TNLANALYTICSOLUTION_IMPL_H
#define	TNLANALYTICSOLUTION_IMPL_H

#include "tnlAnalyticSolution.h"

template<typename Real, typename Device, typename Index>
template< typename TimeFunction, typename AnalyticSpaceFunction>
void AnalyticSolution<tnlGrid<1,Real,Device,Index,tnlIdenticalGridGeometry>>::
compute(const MeshType& mesh, const RealType& time, SharedVector& output, 
        SharedVector& numerical, TimeFunction timeFunction, AnalyticSpaceFunction analyticSpaceFunction)
{  
      RealType timeFunctionValue = timeFunction.getTimeValue(time);
      
      VertexType vertex;
      CoordinatesType coordinates;
      
      for(IndexType i=0; i<output.getSize(); i++)
      {
         mesh.getElementCoordinates(i,coordinates);
         mesh.getElementCenter(coordinates,vertex);
         
         output[i] = timeFunctionValue*analyticSpaceFunction.getF(vertex);
      }
}

template<typename Real, typename Device, typename Index>
template< typename TimeFunction, typename AnalyticSpaceFunction>
void AnalyticSolution<tnlGrid<2,Real,Device,Index,tnlIdenticalGridGeometry>>::
compute(const MeshType& mesh, const RealType& time, SharedVector& output, 
        SharedVector& numerical, TimeFunction timeFunction, AnalyticSpaceFunction analyticSpaceFunction)
{
   RealType timeFunctionValue = timeFunction.getTimeValue(time);
   
   VertexType vertex;
   CoordinatesType coordinates;
   
   #ifdef HAVE_OPENMP
    #pragma omp parallel for private(coordinates,vertex)
   #endif
   for(IndexType i=0; i<output.getSize(); i++)
   {      
      mesh.getElementCoordinates(i,coordinates);
      mesh.getElementCenter(coordinates,vertex);
      
      output[i] = timeFunctionValue*analyticSpaceFunction.getF(vertex);
   }   
   
}

template<typename Real, typename Device, typename Index>
template< typename TimeFunction, typename AnalyticSpaceFunction>
void AnalyticSolution<tnlGrid<3,Real,Device,Index,tnlIdenticalGridGeometry>>::
compute(const MeshType& mesh, const RealType& time, SharedVector& output,
        SharedVector& numerical, TimeFunction timeFunction, AnalyticSpaceFunction analyticSpaceFunction)
{
   RealType timeFunctionValue = timeFunction.getTimeValue(time);
   
   VertexType vertex;
   CoordinatesType coordinates;
      
   for(IndexType i=0; i<output.getSize(); i++)
   {      
      mesh.getElementCoordinates(i,coordinates);
      mesh.getElementCenter(coordinates,vertex);
      
      output[i] = timeFunctionValue*analyticSpaceFunction.getF(vertex);
   }   
}


#endif	/* TNLANALYTICSOLUTION_IMPL_H */

