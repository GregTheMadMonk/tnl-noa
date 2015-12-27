
#ifndef TNLFINITEVOLUMENONLINEAROPERATOR__IMPL_H
#define	TNLFINITEVOLUMENONLINEAROPERATOR__IMPL_H

#include "tnlFiniteVolumeNonlinearOperator.h"

#include <mesh/tnlGrid.h>

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index,
          typename OperatorQ >
tnlString
tnlFiniteVolumeNonlinearOperator< tnlGrid< 1, MeshReal, Device, MeshIndex >, OperatorQ, Real, Index >::
getType()
{
   return tnlString( "tnlFiniteVolumeNonlinearOperator< " ) +
          MeshType::getType() + ", " +
          ::getType< Real >() + ", " +
          ::getType< Index >() + ", " +
	  OperatorQ::getType() + " >";
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index,
          typename OperatorQ >
template< typename Vector,
          typename MeshEntity >
__cuda_callable__
Real
tnlFiniteVolumeNonlinearOperator< tnlGrid< 1, MeshReal, Device, MeshIndex >, OperatorQ, Real, Index >::
getValue( const MeshType& mesh,
          const MeshEntity& entity,
          const Vector& u,
          const Real& time ) const
{
   return 0.0;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index,
          typename OperatorQ >
   template< typename MeshEntity >
__cuda_callable__
Index
tnlFiniteVolumeNonlinearOperator< tnlGrid< 1, MeshReal, Device, MeshIndex >, OperatorQ, Real, Index >::
getLinearSystemRowLength( const MeshType& mesh,
                          const IndexType& index,
                          const MeshEntity& entity ) const
{
   return 1;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index,
          typename OperatorQ >
template< typename Vector,
          typename MeshEntity,
          typename MatrixRow >
__cuda_callable__
void
tnlFiniteVolumeNonlinearOperator< tnlGrid< 1, MeshReal, Device, MeshIndex >, OperatorQ, Real, Index >::
updateLinearSystem( const RealType& time,
                    const RealType& tau,
                    const MeshType& mesh,
                    const IndexType& index,
                    const MeshEntity& entity,
                    Vector& u,
                    Vector& b,
                    MatrixRow& matrixRow ) const
{
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index,
	  typename OperatorQ >
tnlString
tnlFiniteVolumeNonlinearOperator< tnlGrid< 2, MeshReal, Device, MeshIndex >, OperatorQ, Real, Index >::
getType()
{
   return tnlString( "tnlFiniteVolumeNonlinearOperator< " ) +
          MeshType::getType() + ", " +
          ::getType< Real >() + ", " +
          ::getType< Index >() + ", " +
	  OperatorQ::getType() + " >";
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index,
          typename OperatorQ >
template< typename Vector,
          typename MeshEntity >
__cuda_callable__
Real
tnlFiniteVolumeNonlinearOperator< tnlGrid< 2, MeshReal, Device, MeshIndex >, OperatorQ, Real, Index >::
getValue( const MeshType& mesh,
          const MeshEntity& entity,
          const Vector& u,
          const Real& time ) const
{
   const typename EntityType::template NeighbourEntities< 2 >& neighbourEntities = entity.getNeighbourEntities();      
   return operatorQ.getValue( mesh, entity, u, time ) * 
      ( (  u[ neighbourEntities.template getEntityIndex<  1, 0 >() ] - u[ cellIndex ] ) * mesh.getHxSquareInverse() / operatorQ.getValue( mesh, entity, u, time, 1 )
      + (  u[ neighbourEntities.template getEntityIndex<  0, 1 >() ] - u[ cellIndex ] ) * mesh.getHySquareInverse() / operatorQ.getValue( mesh, entity, u, time, 0, 1 ) 
      - ( -u[ neighbourEntities.template getEntityIndex< -1, 0 >() ] + u[ cellIndex ] ) * mesh.getHxSquareInverse() / operatorQ.getValue( mesh, entity, u, time, -1)
      - ( -u[ neighbourEntities.template getEntityIndex<  0,-1 >() ] + u[ cellIndex ] ) * mesh.getHySquareInverse() / operatorQ.getValue( mesh, entity, u, time, 0, -1) );
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index,
          typename OperatorQ >
   template< typename MeshEntity >
__cuda_callable__
Index
tnlFiniteVolumeNonlinearOperator< tnlGrid< 2, MeshReal, Device, MeshIndex >, OperatorQ, Real, Index >::
getLinearSystemRowLength( const MeshType& mesh,
                          const IndexType& index,
                          const MeshEntity& entity ) const
{
   return 5;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index,
          typename OperatorQ >
template< typename Vector,
          typename MeshEntity, 
          typename MatrixRow >
__cuda_callable__
void
tnlFiniteVolumeNonlinearOperator< tnlGrid< 2, MeshReal, Device, MeshIndex >, OperatorQ, Real, Index >::
updateLinearSystem( const RealType& time,
                    const RealType& tau,
                    const MeshType& mesh,
                    const IndexType& index,
                    const MeshEntity& entity,
                    Vector& u,
                    Vector& b,
                    MatrixRow& matrixRow ) const
{
   const typename EntityType::template NeighbourEntities< 2 >& neighbourEntities = entity.getNeighbourEntities();
   const RealType aCoef = - tau * operatorQ.getValue(mesh, entity, u, time ) * mesh.getHySquareInverse() / 
                       operatorQ.getValue(mesh, entity, u, time, 0, -1 );
   const RealType bCoef = - tau * operatorQ.getValue(mesh, entity, u, time ) * mesh.getHxSquareInverse() / 
                       operatorQ.getValue(mesh, entity, u, time, -1 );
   const RealType cCoef = tau * operatorQ.getValue(mesh, entity, u, time ) * ( mesh.getHxSquareInverse() / 
                       operatorQ.getValue(mesh, entity, u, time, 1 ) + mesh.getHySquareInverse() / 
                       operatorQ.getValue(mesh, entity, u, time, 0, 1 )
                       + mesh.getHxSquareInverse() / operatorQ.getValue(mesh, entity, u, time, -1 ) + 
                       mesh.getHySquareInverse() / operatorQ.getValue(mesh, entity, u, time, 0, -1 ) );
   const RealType dCoef = - tau * operatorQ.getValue(mesh, index, entity, u, time ) * mesh.getHxSquareInverse() / 
                       operatorQ.getValue(mesh, index, entity, u, time, 1 );
   const RealType eCoef = - tau * operatorQ.getValue(mesh, index, entity, u, time ) * mesh.getHySquareInverse() / 
                       operatorQ.getValue(mesh, index, entity, u, time, 0, 1 );
   matrixRow.setElement( 0, mesh.template getCellNextToCell< 0,-1 >( index ),     aCoef );
   matrixRow.setElement( 1, mesh.template getCellNextToCell< -1,0 >( index ),     bCoef );
   matrixRow.setElement( 2, index                                           ,     cCoef );
   matrixRow.setElement( 3, mesh.template getCellNextToCell< 1,0 >( index ),      dCoef );
   matrixRow.setElement( 4, mesh.template getCellNextToCell< 0,1 >( index ),      eCoef );
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index,
 	  typename OperatorQ >
tnlString
tnlFiniteVolumeNonlinearOperator< tnlGrid< 3, MeshReal, Device, MeshIndex >, OperatorQ, Real, Index >::
getType()
{
   return tnlString( "tnlFiniteVolumeNonlinearOperator< " ) +
          MeshType::getType() + ", " +
          ::getType< Real >() + ", " +
          ::getType< Index >() + ", " +
	  OperatorQ::getType() + " >";
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index,
          typename OperatorQ >
template< typename Vector,
          typename MeshEntity >
__cuda_callable__
Real
tnlFiniteVolumeNonlinearOperator< tnlGrid< 3, MeshReal, Device, MeshIndex >, OperatorQ, Real, Index >::
getValue( const MeshType& mesh,
          const MeshEntity& entity,
          const Vector& u,
          const Real& time ) const
{
   const typename EntityType::template NeighbourEntities< 3 >& neighbourEntities = entity.getNeighbourEntities();
   return operatorQ.getValue( mesh, entity, u, time ) * 
      ( (u[ mesh.template getCellNextToCell< 1,0,0 >( cellIndex ) ] - u[ cellIndex ]) 
          * mesh.getHxSquareInverse() / operatorQ.getValue(mesh,cellIndex, entity, u, time, 1 )
          + ( u[ mesh.template getCellNextToCell< 0,1,0 >( cellIndex ) ] - u[ cellIndex ]) * mesh.getHySquareInverse()/
          operatorQ.getValue(mesh,cellIndex, entity, u, time, 0, 1 ) 
          + ( u[ mesh.template getCellNextToCell< 0,0,1 >( cellIndex ) ] - u[ cellIndex ]) * mesh.getHzSquareInverse()/
          operatorQ.getValue(mesh,cellIndex, entity, u, time, 0, 0, 1 ) 
          - ( - u[ mesh.template getCellNextToCell< -1,0,0 >( cellIndex ) ]  + u[ cellIndex ]) 
          * mesh.getHxSquareInverse()/operatorQ.getValue(mesh,cellIndex, entity, u, time, -1)
          -( - u[ mesh.template getCellNextToCell< 0,-1,0 >( cellIndex ) ] + u[ cellIndex ]) * mesh.getHySquareInverse()
          /operatorQ.getValue(mesh,cellIndex,entity, u, time, 0, -1) 
          -( - u[ mesh.template getCellNextToCell< 0,0,-1 >( cellIndex ) ] + u[ cellIndex ]) * mesh.getHzSquareInverse()
          /operatorQ.getValue(mesh,cellIndex,entity, u, time, 0, 0, -1) );
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index,
          typename OperatorQ >
   template< typename MeshEntity >
__cuda_callable__
Index
tnlFiniteVolumeNonlinearOperator< tnlGrid< 3, MeshReal, Device, MeshIndex >, OperatorQ, Real, Index >::
getLinearSystemRowLength( const MeshType& mesh,
                          const IndexType& index,
                          const MeshEntity& entity ) const
{
   return 7;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index,
          typename OperatorQ >
template< typename Vector,
          typename MeshEntity,
          typename MatrixRow >
#ifdef HAVE_CUDA
__cuda_callable__
#endif
void
tnlFiniteVolumeNonlinearOperator< tnlGrid< 3, MeshReal, Device, MeshIndex >, OperatorQ, Real, Index >::
updateLinearSystem( const RealType& time,
                    const RealType& tau,
                    const MeshType& mesh,
                    const IndexType& index,
                    const MeshEntity& entity,
                    Vector& u,
                    Vector& b,
                    MatrixRow& matrixRow ) const
{
   const typename EntityType::template NeighbourEntities< 3 >& neighbourEntities = entity.getNeighbourEntities();
   const RealType aCoef = - tau * operatorQ.getValue(mesh, index, entity, u, time ) * mesh.getHzSquareInverse() / 
                       operatorQ.getValue(mesh, index, entity, u, time, 0, 0, -1 );
   const RealType bCoef = - tau * operatorQ.getValue(mesh, index, entity, u, time ) * mesh.getHySquareInverse() / 
                       operatorQ.getValue(mesh, index, entity, u, time, 0, -1, 0 );
   const RealType cCoef = - tau * operatorQ.getValue(mesh, index, entity, u, time ) * mesh.getHxSquareInverse() / 
                       operatorQ.getValue(mesh, index, entity, u, time, -1, 0, 0 );
   const RealType dCoef = tau * operatorQ.getValue(mesh, index, entity, u, time ) * ( mesh.getHxSquareInverse() / 
                       operatorQ.getValue(mesh, index, entity, u, time, 1, 0, 0 ) + mesh.getHySquareInverse() / 
                       operatorQ.getValue(mesh, index, entity, u, time, 0, 1, 0 )
                       + mesh.getHzSquareInverse() / operatorQ.getValue(mesh, index, entity, u, time, 0, 0, 1 ) + 
                       mesh.getHxSquareInverse() / operatorQ.getValue(mesh, index, entity, u, time, -1, 0, 0 )
                       + mesh.getHySquareInverse() / operatorQ.getValue(mesh, index, entity, u, time, 0, -1, 0 ) + 
                       mesh.getHzSquareInverse() / operatorQ.getValue(mesh, index, entity, u, time, 0, 0, -1 ) );
   const RealType eCoef = - tau * operatorQ.getValue(mesh, index, entity, u, time ) * mesh.getHxSquareInverse() / 
                       operatorQ.getValue(mesh, index, entity, u, time, 1, 0, 0 );
   const RealType fCoef = - tau * operatorQ.getValue(mesh, index, entity, u, time ) * mesh.getHySquareInverse() / 
                       operatorQ.getValue(mesh, index, entity, u, time, 0, 1, 0 );
   const RealType gCoef = - tau * operatorQ.getValue(mesh, index, entity, u, time ) * mesh.getHzSquareInverse() / 
                       operatorQ.getValue(mesh, index, entity, u, time, 0, 0, 1 );
   matrixRow.setElement( 0, mesh.template getCellNextToCell< 0,0,-1 >( index ),     aCoef );
   matrixRow.setElement( 1, mesh.template getCellNextToCell< 0,-1,0 >( index ),     bCoef );
   matrixRow.setElement( 2, mesh.template getCellNextToCell< -1,0,0 >( index ),     cCoef );
   matrixRow.setElement( 3, index,                                                  dCoef );
   matrixRow.setElement( 4, mesh.template getCellNextToCell< 1,0,0 >( index ),      eCoef );
   matrixRow.setElement( 5, mesh.template getCellNextToCell< 0,1,0 >( index ),      fCoef );
   matrixRow.setElement( 6, mesh.template getCellNextToCell< 0,0,1 >( index ),      gCoef );
}
#endif	/* TNLFINITEVOLUMENONLINEAROPERATOR__IMPL_H */
