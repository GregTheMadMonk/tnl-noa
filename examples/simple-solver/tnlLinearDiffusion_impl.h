
#ifndef TNLLINEARDIFFUSION_IMP_H
#define	TNLLINEARDIFFUSION_IMP_H

#include "tnlLinearDiffusion.h"

template<typename Real, typename Device, typename Index>
void tnlLinearDiffusion<tnlGrid<1,Real,Device,Index,tnlIdenticalGridGeometry>> ::
getExplicitRHS( const MeshType& mesh, const RealType& time, const RealType& tau,
                const CoordinatesType& coordinates, DofVectorType& _u, DofVectorType& _fu)
{
   if((coordinates.x() <= 0) || (coordinates.x() >= (mesh.getDimensions().x()-1)))
   {
      cerr << "It is not possible to compute Laplace operator on the given dof. Coordinates are "
           <<coordinates.x()<<"in x-axis. ";
      return;
   }
   
   _fu[mesh.getElementIndex(coordinates.x())]=((_u[mesh.getElementIndex(coordinates.x()-1)]-
           2*_u[mesh.getElementIndex(coordinates.x())]+_u[mesh.getElementIndex(coordinates.x()+1)])/
           (mesh.getParametricStep().x()*mesh.getParametricStep().x()));
}
   
template<typename Real, typename Device, typename Index>
void tnlLinearDiffusion<tnlGrid<1,Real,Device,Index,tnlIdenticalGridGeometry>> ::
getExplicitRHS( const MeshType& mesh, const RealType& time, const RealType& tau,
                DofVectorType& _u, DofVectorType& _fu)
{
   RealType stepSquare = mesh.getParametricStep().x()*mesh.getParametricStep().x();
   CoordinatesType dimensions = mesh.getDimensions();
   
   for(IndexType i=1;i<(dimensions.x()-1);i++)
   {
      _fu[i]=(_u[i-1]-2*_u[i]+_u[i+1])/(stepSquare);
   } 
   
}

template<typename Real, typename Device, typename Index>
void tnlLinearDiffusion<tnlGrid<2,Real,Device,Index,tnlIdenticalGridGeometry>> :: 
getExplicitRHS( const MeshType& mesh, const RealType& time, const RealType& tau,
                const CoordinatesType& coordinates, DofVectorType& _u, DofVectorType& _fu)
{
   if((coordinates.x() <= 0) || (coordinates.x() >= (mesh.getDimensions().x()-1)) || 
      (coordinates.y() <= 0 ) || (coordinates.y() >= (mesh.getDimensions().y()-1)))
   {
      cerr << "It is not possible to compute Laplace operator on the given dof. Coordinates are "
           <<coordinates.x()<<"in x-axis, "<<coordinates.y()<<"in y-axis. ";
      return;
   }

   _fu[mesh.getElementIndex(coordinates.x(),coordinates.y())]=((_u[mesh.getElementIndex(coordinates.x()-1,coordinates.y())]-
           2*_u[mesh.getElementIndex(coordinates.x(),coordinates.y())]+_u[mesh.getElementIndex(coordinates.x()+1,coordinates.y())])/
           (mesh.getParametricStep().x()*mesh.getParametricStep().x()));
   _fu[mesh.getElementIndex(coordinates.x(),coordinates.y())]+=((_u[mesh.getElementIndex(coordinates.x(),coordinates.y()-1)]-
           2*_u[mesh.getElementIndex(coordinates.x(),coordinates.y())]+_u[mesh.getElementIndex(coordinates.x(),coordinates.y()+1)])/
           (mesh.getParametricStep().y()*mesh.getParametricStep().y()));
}
   
template<typename Real, typename Device, typename Index>
void tnlLinearDiffusion<tnlGrid<2,Real,Device,Index,tnlIdenticalGridGeometry>> :: 
getExplicitRHS( const MeshType& mesh, const RealType& time, const RealType& tau,
                DofVectorType& _u, DofVectorType& _fu)
{
   RealType stepXSquare = mesh.getParametricStep().x()*mesh.getParametricStep().x();
   RealType stepYSquare = mesh.getParametricStep().y()*mesh.getParametricStep().y();
   
   CoordinatesType dimensions = mesh.getDimensions();
   
   #ifdef HAVE_OPENMP
    #pragma omp parallel for shared(_fu,mesh,_u,stepXSquare,stepYSquare,dimensions) 
   #endif
   for(IndexType i=1;i<(dimensions.x()-1);i++)
   {
      for(IndexType j=1;j<(dimensions.y()-1);j++)
      {
         _fu[mesh.getElementIndex(i,j)]=(_u[mesh.getElementIndex(i-1,j)]-2*_u[mesh.getElementIndex(i,j)]+
             _u[mesh.getElementIndex(i+1,j)])/(stepXSquare)+(_u[mesh.getElementIndex(i,j-1)]-2*_u[mesh.getElementIndex(i,j)]+
             _u[mesh.getElementIndex(i,j+1)])/(stepYSquare);
      }
   }   
}

template<typename Real, typename Device, typename Index>
void tnlLinearDiffusion<tnlGrid<3,Real,Device,Index,tnlIdenticalGridGeometry>> :: 
getExplicitRHS( const MeshType& mesh, const RealType& time, const RealType& tau,
                const CoordinatesType& coordinates, DofVectorType& _u, DofVectorType& _fu)
{
   if((coordinates.x() <= 0) || (coordinates.x() >= (mesh.getDimensions().x()-1)) || 
      (coordinates.y() <= 0 ) || (coordinates.y() >= (mesh.getDimensions().y()-1)) || 
      (coordinates.y() <= 0 ) || (coordinates.z() >= (mesh.getDimensions().z()-1)))
   {
      cerr << "It is not possible to compute Laplace operator on given dof. Coordinates are "
           <<coordinates.x()<<"in x-axis, "<<coordinates.y()<<"in y-axis, "<<coordinates.z()<<"in z-axis. ";
   }
   
   _fu[mesh.getElementIndex(coordinates.x(),coordinates.y(),coordinates.z())]=
           ((_u[mesh.getElementIndex(coordinates.x()-1,coordinates.y(),coordinates.z())]-
           2*_u[mesh.getElementIndex(coordinates.x(),coordinates.y(),coordinates.z())]+
           _u[mesh.getElementIndex(coordinates.x()+1,coordinates.y(),coordinates.z())])/
           (mesh.getParametricStep().x()*mesh.getParametricStep().x()));
   _fu[mesh.getElementIndex(coordinates.x(),coordinates.y(),coordinates.z())]+=
           ((_u[mesh.getElementIndex(coordinates.x(),coordinates.y()-1,coordinates.z())]-
           2*_u[mesh.getElementIndex(coordinates.x(),coordinates.y(),coordinates.z())]+
           _u[mesh.getElementIndex(coordinates.x(),coordinates.y()+1,coordinates.z())])/
           (mesh.getParametricStep().y()*mesh.getParametricStep().y()));
   _fu[mesh.getElementIndex(coordinates.x(),coordinates.y(),coordinates.z())]+=
           ((_u[mesh.getElementIndex(coordinates.x(),coordinates.y(),coordinates.z()-1)]-
           2*_u[mesh.getElementIndex(coordinates.x(),coordinates.y(),coordinates.z())]+
           _u[mesh.getElementIndex(coordinates.x(),coordinates.y(),coordinates.z()+1)])/
           (mesh.getParametricStep().z()*mesh.getParametricStep().z()));
}
   
template<typename Real, typename Device, typename Index>
void tnlLinearDiffusion<tnlGrid<3,Real,Device,Index,tnlIdenticalGridGeometry>> :: 
getExplicitRHS( const MeshType& mesh, const RealType& time, const RealType& tau,
                DofVectorType& _u, DofVectorType& _fu)
{
   RealType stepXSquare = mesh.getParametricStep().x()*mesh.getParametricStep().x();
   RealType stepYSquare = mesh.getParametricStep().y()*mesh.getParametricStep().y();   
   RealType stepZSquare = mesh.getParametricStep().z()*mesh.getParametricStep().z();  
   
   CoordinatesType dimensions = mesh.getDimensions();
    
   for(IndexType i=1;i<(dimensions.x()-1);i++)
   {
      for(IndexType j=1;j<(dimensions.y()-1);j++)
      {
         for(IndexType k=1;k<(dimensions.z()-1);k++)
         {
            _fu[mesh.getElementIndex(i,j,k)]=(_u[mesh.getElementIndex(i-1,j,k)]-
                    2*_u[mesh.getElementIndex(i,j,k)]+_u[mesh.getElementIndex(i+1,j,k)])/
                    (stepXSquare) + (_u[mesh.getElementIndex(i,j-1,k)]-
                    2*_u[mesh.getElementIndex(i,j,k)]+_u[mesh.getElementIndex(i,j+1,k)])/
                    (stepYSquare) + (_u[mesh.getElementIndex(i,j,k-1)]-
                    2*_u[mesh.getElementIndex(i,j,k)]+_u[mesh.getElementIndex(i,j,k+1)])/
                    (stepZSquare);
         }
      }
   }   
}


#endif	/* TNLLINEARDIFFUSION_IMP_H */

