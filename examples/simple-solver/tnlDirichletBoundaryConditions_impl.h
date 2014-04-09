#ifndef TNLDIRICHLETBOUNDARYCONDITIONS_IMPL_H
#define	TNLDIRICHLETBOUNDARYCONDITIONS_IMPL_H

#include "tnlDirichletBoundaryConditions.h"

template<typename Real, typename Device, typename Index>
template<typename AnalyticSpaceFunction, typename TimeFunction, typename Vector>
void tnlDirichletBoundaryConditions<tnlGrid<1,Real,Device,Index,tnlIdenticalGridGeometry>>
::applyBoundaryConditions(const MeshType& mesh, Vector& u, const RealType& time,
                          TimeFunction& timeFunction,AnalyticSpaceFunction& analyticSpaceFunction)
{
   RealType timeFunctionValue = timeFunction.getTimeValue(time);
   
   CoordinatesType coordinates;
   VertexType vertex;
   
   mesh.getElementCoordinates(0,coordinates);
   mesh.getElementCenter(coordinates,vertex);
   
   u[0] = timeFunctionValue*analyticSpaceFunction.getF(vertex);
   
   mesh.getElementCoordinates(u.getSize()-1,coordinates);
   mesh.getElementCenter(coordinates,vertex);

   u[u.getSize()-1] = timeFunctionValue*analyticSpaceFunction.getF(vertex);
}

template<typename Real, typename Device, typename Index>
template<typename AnalyticSpaceFunction, typename TimeFunction, typename Vector>
void tnlDirichletBoundaryConditions<tnlGrid<1,Real,Device,Index,tnlIdenticalGridGeometry>>::
applyBoundaryTimeDerivation(const MeshType& mesh, Vector& u, const RealType& time,
                            TimeFunction& timeFunction, AnalyticSpaceFunction& analyticSpaceFunction)
{
   RealType timeFunctionDerivationValue = timeFunction.getDerivation(time);
   
   CoordinatesType coordinates;
   VertexType vertex;
   
   mesh.getElementCoordinates(0,coordinates);
   mesh.getElementCenter(coordinates,vertex);
   
   u[0] = timeFunctionDerivationValue*analyticSpaceFunction.getF(vertex);
   
   mesh.getElementCoordinates(u.getSize()-1,coordinates);
   mesh.getElementCenter(coordinates,vertex);

   u[u.getSize()-1] = timeFunctionDerivationValue*analyticSpaceFunction.getF(vertex);
}

template<typename Real, typename Device, typename Index>
template<typename AnalyticSpaceFunction, typename TimeFunction, typename Vector>
void tnlDirichletBoundaryConditions<tnlGrid<2,Real,Device,Index,tnlIdenticalGridGeometry>>::
applyBoundaryConditions(const MeshType& mesh, Vector& u, const RealType& time,
                        TimeFunction& timeFunction, AnalyticSpaceFunction& analyticSpaceFunction)
{  
   RealType timeFunctionValue = timeFunction.getTimeValue(time);   
   
   CoordinatesType dimensions = mesh.getDimensions();
   CoordinatesType coordinates1;
   CoordinatesType coordinates2;
   VertexType vertex;
   
   coordinates1.y()=0;
   coordinates2.y()=dimensions.y()-1;

   #ifdef HAVE_OPENMP
    #pragma omp parallel for firstprivate(coordinates1,coordinates2) private(vertex)
   #endif
   for(IndexType i=0; i<dimensions.x(); i++)
   {
      coordinates1.x()=coordinates2.x()=i;
      mesh.getElementCenter(coordinates1,vertex);
      u[i] = timeFunctionValue*analyticSpaceFunction.getF(vertex);
      
      mesh.getElementCenter(coordinates2,vertex);
      u[(dimensions.y()-1)*mesh.getDimensions().x()+i] = timeFunctionValue*analyticSpaceFunction.getF(vertex);
   }
   
   coordinates1.x()=0;
   coordinates2.x()=dimensions.x()-1;

   #ifdef HAVE_OPENMP
    #pragma omp parallel for firstprivate(coordinates1,coordinates2) private(vertex)
   #endif
   for(IndexType i=0; i<dimensions.y(); i++)
   {
      coordinates1.y()=coordinates2.y()=i;
           
      mesh.getElementCenter(coordinates1,vertex);
      u[i*mesh.getDimensions().x()] = timeFunctionValue*analyticSpaceFunction.getF(vertex);
      
      mesh.getElementCenter(coordinates2,vertex);
      u[i*mesh.getDimensions().x()+dimensions.x()-1] = timeFunctionValue*analyticSpaceFunction.getF(vertex);
   }
}

template<typename Real, typename Device, typename Index>
template<typename AnalyticSpaceFunction, typename TimeFunction, typename Vector>
void tnlDirichletBoundaryConditions<tnlGrid<2,Real,Device,Index,tnlIdenticalGridGeometry>>::
applyBoundaryTimeDerivation(const MeshType& mesh, Vector& u, const RealType& time,
                            TimeFunction& timeFunction, AnalyticSpaceFunction& analyticSpaceFunction)
{
   RealType timeFunctionDerivationValue = timeFunction.getDerivation(time);
   
   CoordinatesType dimensions = mesh.getDimensions();
   CoordinatesType coordinates1;
   CoordinatesType coordinates2;
   VertexType vertex;

   coordinates1.y()=0;
   coordinates2.y()=dimensions.y()-1;  

   #ifdef HAVE_OPENMP
    #pragma omp parallel for firstprivate(coordinates1,coordinates2) private(vertex)
   #endif
   for(IndexType i=0; i<dimensions.x(); i++)
   {
      coordinates1.x()=coordinates2.x()=i;
      mesh.getElementCenter(coordinates1,vertex);
      u[mesh.getElementIndex(i,0)] = timeFunctionDerivationValue*analyticSpaceFunction.getF(vertex);
      
      mesh.getElementCenter(coordinates2,vertex);
      u[mesh.getElementIndex(i,dimensions.y()-1)] = timeFunctionDerivationValue*analyticSpaceFunction.getF(vertex);
   }
   
   coordinates1.x()=0;
   coordinates2.x()=dimensions.x()-1; 

   #ifdef HAVE_OPENMP
    #pragma omp parallel for firstprivate(coordinates1,coordinates2) private(vertex)
   #endif
   for(IndexType i=0;i < dimensions.y(); i++)
   {
      coordinates1.y()=coordinates2.y()=i;
           
      mesh.getElementCenter(coordinates1,vertex);
      u[mesh.getElementIndex(0,i)] = timeFunctionDerivationValue*analyticSpaceFunction.getF(vertex);

      mesh.getElementCenter(coordinates2,vertex);
      u[mesh.getElementIndex(dimensions.x()-1,i)] = timeFunctionDerivationValue*analyticSpaceFunction.getF(vertex);
   }
}

template<typename Real, typename Device, typename Index>
template<typename AnalyticSpaceFunction, typename TimeFunction, typename Vector>
void tnlDirichletBoundaryConditions<tnlGrid<3,Real,Device,Index,tnlIdenticalGridGeometry>>::
applyBoundaryConditions(const MeshType& mesh, Vector& u, const RealType& time, 
                        TimeFunction& timeFunction, AnalyticSpaceFunction& analyticSpaceFunction)
{
   RealType timeFunctionValue = timeFunction.getTimeValue(time);
   
   CoordinatesType dimensions = mesh.getDimensions();
   CoordinatesType coordinates;
   VertexType vertex;

   #ifdef HAVE_OPENMP
    #pragma omp parallel for firstprivate(coordinates) private(vertex)
   #endif
   for(IndexType i=0; i<dimensions.x(); i++)
   {
      for(IndexType j=0; j<dimensions.y(); j++)
      {
         coordinates.x()=i;
         coordinates.y()=j;
         
         coordinates.z()=0;
         mesh.getElementCenter(coordinates,vertex);
         u[mesh.getElementIndex(i,j,0)] = timeFunctionValue*analyticSpaceFunction.getF(vertex);
         
         coordinates.z()=dimensions.z()-1;
         mesh.getElementCenter(coordinates,vertex);
         u[mesh.getElementIndex(i,j,dimensions.z()-1)] = timeFunctionValue*analyticSpaceFunction.getF(vertex);
      }
   }

   #ifdef HAVE_OPENMP
    #pragma omp parallel for firstprivate(coordinates) private(vertex)
   #endif
   for(IndexType i=0; i<dimensions.x(); i++)
   {
      for(IndexType j=0; j<dimensions.y(); j++)
      {
         coordinates.y()=i;
         coordinates.z()=j;
         
         coordinates.x()=0;
         mesh.getElementCenter(coordinates,vertex);
         u[mesh.getElementIndex(0,i,j)] = timeFunctionValue*analyticSpaceFunction.getF(vertex);
         
         coordinates.x()=dimensions.x()-1;
         mesh.getElementCenter(coordinates,vertex);
         u[mesh.getElementIndex(dimensions.x()-1,i,j)] = timeFunctionValue*analyticSpaceFunction.getF(vertex);
      }
   }

   #ifdef HAVE_OPENMP
    #pragma omp parallel for firstprivate(coordinates) private(vertex)
   #endif
   for(IndexType i=0; i<dimensions.x(); i++)
   {
      for(IndexType j=1; j<dimensions.y(); j++)
      {
         coordinates.x()=i;
         coordinates.z()=j;
         
         coordinates.y()=0;
         mesh.getElementCenter(coordinates,vertex);         
         u[mesh.getElementIndex(i,0,j)] = timeFunctionValue*analyticSpaceFunction.getF(vertex);
         
         coordinates.y()=dimensions.y()-1;
         mesh.getElementCenter(coordinates,vertex);   
         u[mesh.getElementIndex(i,dimensions.y()-1,j)] = timeFunctionValue*analyticSpaceFunction.getF(vertex);
      }
   } 
}

template<typename Real, typename Device, typename Index>
template<typename AnalyticSpaceFunction, typename TimeFunction, typename Vector>
void tnlDirichletBoundaryConditions<tnlGrid<3,Real,Device,Index,tnlIdenticalGridGeometry>>::
applyBoundaryTimeDerivation(const MeshType& mesh, Vector& u, const RealType& time, 
                            TimeFunction& timeFunction, AnalyticSpaceFunction& analyticSpaceFunction)
{
   RealType timeFunctionDerivationValue = timeFunction.getDerivation(time);
   
   CoordinatesType dimensions = mesh.getDimensions();
   CoordinatesType coordinates;
   VertexType vertex;

   #ifdef HAVE_OPENMP
    #pragma omp parallel for firstprivate(coordinates) private(vertex)
   #endif
   for(IndexType i=0; i<dimensions.x(); i++)
   {
      for(IndexType j=0; j<dimensions.y(); j++)
      {
         coordinates.x()=i;
         coordinates.y()=j;
         
         coordinates.z()=0;
         mesh.getElementCenter(coordinates,vertex);
         u[mesh.getElementIndex(i,j,0)] = timeFunctionDerivationValue*analyticSpaceFunction.getF(vertex);
         
         coordinates.z()=dimensions.z()-1;
         mesh.getElementCenter(coordinates,vertex);
         u[mesh.getElementIndex(i,j,dimensions.z()-1)] = timeFunctionDerivationValue*analyticSpaceFunction.getF(vertex);
      }
   }

   #ifdef HAVE_OPENMP
    #pragma omp parallel for firstprivate(coordinates) private(vertex)
   #endif
   for(IndexType i=0; i<dimensions.x(); i++)
   {
      for(IndexType j=0; j<dimensions.y(); j++)
      {
         coordinates.y()=i;
         coordinates.z()=j;
         
         coordinates.x()=0;
         mesh.getElementCenter(coordinates,vertex);
         u[mesh.getElementIndex(0,i,j)] = timeFunctionDerivationValue*analyticSpaceFunction.getF(vertex);
         
         coordinates.x()=dimensions.x()-1;
         mesh.getElementCenter(coordinates,vertex);
         u[mesh.getElementIndex(dimensions.x()-1,i,j)] = timeFunctionDerivationValue*analyticSpaceFunction.getF(vertex);
      }
   }

   #ifdef HAVE_OPENMP
    #pragma omp parallel for firstprivate(coordinates) private(vertex)
   #endif
   for(IndexType i=0; i<dimensions.x(); i++)
   {
      for(IndexType j=0; j<dimensions.y(); j++)
      {
         coordinates.x()=i;
         coordinates.z()=j;
         
         coordinates.y()=0;
         mesh.getElementCenter(coordinates,vertex);         
         u[mesh.getElementIndex(i,0,j)] = timeFunctionDerivationValue*analyticSpaceFunction.getF(vertex);
         
         coordinates.y()=dimensions.y()-1;
         mesh.getElementCenter(coordinates,vertex);   
         u[mesh.getElementIndex(i,dimensions.y()-1,j)] = timeFunctionDerivationValue*analyticSpaceFunction.getF(vertex);
      }
   } 
}


#endif	/* TNLDIRICHLETBOUNDARYCONDITIONS_IMPL_H */

