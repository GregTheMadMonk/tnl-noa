
#ifndef TNLMEANCURVATIVEFLOWDIFFUSION_IMPL_H
#define	TNLMEANCURVATIVEFLOWDIFFUSION_IMPL_H

#include "tnlMeanCurvatureFlowDiffusion.h"
#include "tnlForwardFiniteDifference.h"
#include "tnlBackwardFiniteDifference.h"

#include <mesh/tnlGrid.h>

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
tnlString
tnlMeanCurvatureFlowDiffusion< tnlGrid< 1, MeshReal, Device, MeshIndex >, Real, Index >::
getType()
{
   return tnlString( "tnlForwardFiniteDifference< " ) +
          MeshType::getType() + ", " +
          ::getType< Real >() + ", " +
          ::getType< Index >() + " >";
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
tnlString
tnlMeanCurvatureFlowDiffusion< tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index >::
getType()
{
   return tnlString( "tnlForwardFiniteDifference< " ) +
          MeshType::getType() + ", " +
          ::getType< Real >() + ", " +
          ::getType< Index >() + " >";
}


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
template< typename Vector >
#ifdef HAVE_CUDA
__device__ __host__
#endif
Real
tnlMeanCurvatureFlowDiffusion< tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index >::
getValue( const MeshType& mesh,
          const IndexType cellIndex,
          const CoordinatesType& coordinates,
          const Vector& u,
          const Real& time ) const
{
   return operatorQ.getValueStriped(mesh,cellIndex,u)*((fDifference.getValueX(mesh,cellIndex,u)/operatorQ.getValue(mesh,cellIndex,u)
          -bDifference.getValueX(mesh,cellIndex,u)/operatorQ.getValue(mesh,mesh.template getCellNextToCell<-1,0>(cellIndex),u))
          *mesh.getHxInverse()+(fDifference.getValueY(mesh,cellIndex,u)/operatorQ.getValue(mesh,cellIndex,u)
          -bDifference.getValueY(mesh,cellIndex,u)/operatorQ.getValue(mesh,mesh.template getCellNextToCell<0,-1>(cellIndex),u))*mesh.getHyInverse());
}
//template< typename MeshReal,
//          typename Device,
//          typename MeshIndex,
//          typename Real,
//          typename Index >
//void tnlMeanCurvatureFlowDiffusion< tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index >::setupDofs ( const MeshType& mesh )
//{
//   dofs.setSize(mesh.getNumberOfCells());
//}
//
//template< typename MeshReal,
//          typename Device,
//          typename MeshIndex,
//          typename Real,
//          typename Index >
//template< typename Vector>
//void 
//tnlMeanCurvatureFlowDiffusion< tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index >::
//computeFirstGradient( const MeshType& mesh,
//                      const RealType& time,
//                      const Vector u)
//{
//   
//}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
tnlString
tnlMeanCurvatureFlowDiffusion< tnlGrid< 3, MeshReal, Device, MeshIndex >, Real, Index >::
getType()
{
   return tnlString( "tnlForwardFiniteDifference< " ) +
          MeshType::getType() + ", " +
          ::getType< Real >() + ", " +
          ::getType< Index >() + " >";
}

#endif	/* TNLMEANCURVATIVEFLOWDIFFUSION_IMPL_H */
