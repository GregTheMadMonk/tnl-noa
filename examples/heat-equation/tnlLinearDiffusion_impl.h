
#ifndef TNLLINEARDIFFUSION_IMP_H
#define	TNLLINEARDIFFUSION_IMP_H

#include "tnlLinearDiffusion.h"
#include <mesh/tnlGrid.h>

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
   template< typename Vector >
void tnlLinearDiffusion< tnlGrid< 1, MeshReal, Device, MeshIndex >, Real, Index >::
getExplicitRHS( const MeshType& mesh,
                const CoordinatesType& coordinates,
                Vector& _u,
                Vector& _fu )
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
   
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
   template< typename Vector >
void tnlLinearDiffusion< tnlGrid< 1, MeshReal, Device, MeshIndex >, Real, Index >::
getExplicitRHS( const MeshType& mesh,
                Vector& _u,
                Vector& _fu )
{
   RealType stepXSquare = mesh.getCellProportions().x()*mesh.getCellProportions().x();
   
   CoordinatesType dimensions = mesh.getDimensions();

   #ifdef HAVE_OPENMP
    #pragma omp parallel for
   #endif
   for(IndexType i=1;i<(dimensions.x()-1);i++)
   {
      _fu[i]=(_u[i-1]-2.0*_u[i]+_u[i+1])/(stepXSquare);
   }
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
   template< typename Vector >
void tnlLinearDiffusion< tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index >::
getExplicitRHS( const MeshType& mesh,
                const CoordinatesType& coordinates,
                Vector& _u,
                Vector& _fu )
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
   

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
   template< typename Vector >
void tnlLinearDiffusion< tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index>::
getExplicitRHS( const MeshType& mesh,
                Vector& _u,
                Vector& _fu )
{
   RealType stepXSquare = mesh.getCellProportions().x()*mesh.getCellProportions().x();
   RealType stepYSquare = mesh.getCellProportions().y()*mesh.getCellProportions().y(); 
   
   CoordinatesType dimensions = mesh.getDimensions();

   #ifdef HAVE_OPENMP
    #pragma omp parallel for
   #endif
   for(IndexType i=1;i<(dimensions.x()-1);i++)
   {
      for(IndexType j=1;j<(dimensions.y()-1);j++)
      {
         _fu[j*mesh.getDimensions().x()+i]=(_u[j*mesh.getDimensions().x()+i-1]-2*_u[j*mesh.getDimensions().x()+i]+
             _u[j*mesh.getDimensions().x()+i+1])/(stepXSquare)+(_u[(j-1)*mesh.getDimensions().x()+i]-2*_u[j*mesh.getDimensions().x()+i]+
             _u[(j+1)*mesh.getDimensions().x()+i])/(stepYSquare);
      }
   }
}


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
   template< typename Vector >
void tnlLinearDiffusion< tnlGrid< 3, MeshReal, Device, MeshIndex >, Real, Index >::
getExplicitRHS( const MeshType& mesh,
                const CoordinatesType& coordinates,
                Vector& _u,
                Vector& _fu )
{
   if((coordinates.x() <= 0) || (coordinates.x() >= (mesh.getDimensions().x()-1)) || 
      (coordinates.y() <= 0 ) || (coordinates.y() >= (mesh.getDimensions().y()-1)) || 
      (coordinates.y() <= 0 ) || (coordinates.z() >= (mesh.getDimensions().z()-1)))
   {
      cerr << "It is not possible to compute Laplace operator on given dof. Coordinates are "
           <<coordinates.x()<<"in x-axis, "<<coordinates.y()<<"in y-axis, "<<coordinates.z()<<"in z-axis. ";
      return;
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
   

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
   template< typename Vector >
void tnlLinearDiffusion< tnlGrid< 3, MeshReal, Device, MeshIndex >, Real, Index >::
getExplicitRHS( const MeshType& mesh,
                Vector& _u,
                Vector& _fu )
{
   RealType stepXSquare = mesh.getCellProportions().x()*mesh.getCellProportions().x();
   RealType stepYSquare = mesh.getCellProportions().y()*mesh.getCellProportions().y();   
   RealType stepZSquare = mesh.getCellProportions().z()*mesh.getCellProportions().z();  
   
   CoordinatesType dimensions = mesh.getDimensions();

   #ifdef HAVE_OPENMP
    #pragma omp parallel for
   #endif
   for(IndexType i=1;i<(dimensions.x()-1);i++)
   {
      for(IndexType j=1;j<(dimensions.y()-1);j++)
      {
         for(IndexType k=1;k<(dimensions.z()-1);k++)
         {
            _fu[ mesh.getCellIndex( CoordinatesType( i,j,k ) ) ] = (_u[mesh.getCellIndex( CoordinatesType( i-1,j,k) ) ]-
                    2*_u[mesh.getCellIndex( CoordinatesType( i,j,k) ) ]+_u[mesh.getCellIndex( CoordinatesType( i+1,j,k) )])/
                    (stepXSquare) + (_u[mesh.getCellIndex( CoordinatesType( i,j-1,k) ) ]-
                    2*_u[mesh.getCellIndex( CoordinatesType( i,j,k) ) ]+_u[mesh.getCellIndex( CoordinatesType( i,j+1,k) )])/
                    (stepYSquare) + (_u[mesh.getCellIndex( CoordinatesType( i,j,k-1) )]-
                    2*_u[mesh.getCellIndex( CoordinatesType( i,j,k) )]+_u[mesh.getCellIndex( CoordinatesType( i,j,k+1) )])/
                    (stepZSquare);
         }
      }
   }   
}


#endif	/* TNLLINEARDIFFUSION_IMP_H */
