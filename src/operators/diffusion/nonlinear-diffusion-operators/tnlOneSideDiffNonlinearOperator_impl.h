
#ifndef TNLONESIDEDIFFNONLINEAROPERATOR_IMPL_H
#define	TNLONESIDEDIFFNONLINEAROPERATOR_IMPL_H

#include "tnlOneSideDiffNonlinearOperator.h"

#include <mesh/tnlGrid.h>

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index,
          typename OperatorQ >
tnlString
tnlOneSideDiffNonlinearOperator< tnlGrid< 1, MeshReal, Device, MeshIndex >, OperatorQ, Real, Index >::
getType()
{
   return tnlString( "tnlOneSideDiffNonlinearOperator< " ) +
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
tnlOneSideDiffNonlinearOperator< tnlGrid< 1, MeshReal, Device, MeshIndex >, OperatorQ, Real, Index >::
getValue( const MeshType& mesh,
          const IndexType cellIndex,
          const MeshEntity& entity,
          const Vector& u,
          const Real& time ) const
{
   const typename EntityType::template NeighbourEntities< 1 >& neighbourEntities = entity.getNeighbourEntities();
   return operatorQ.getValueStriped(mesh,cellIndex, entity, u, time)*((( u[ mesh.template getCellNextToCell< 1 >( cellIndex ) ] - u[ cellIndex ]) 
          * mesh.getHxInverse() / operatorQ.getValue(mesh,cellIndex, entity, u, time) - ( - u[ mesh.template getCellNextToCell< -1>( cellIndex ) ]
          + u[ cellIndex ] ) * mesh.getHxInverse() / operatorQ.getValue(mesh,mesh.template getCellNextToCell<-1>(cellIndex), entity, u, time))
          *mesh.getHxInverse());
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
tnlOneSideDiffNonlinearOperator< tnlGrid< 1, MeshReal, Device, MeshIndex >, OperatorQ, Real, Index >::
getLinearSystemRowLength( const MeshType& mesh,
                          const IndexType& index,
                          const MeshEntity& entity ) const
{
   return 3;
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
tnlOneSideDiffNonlinearOperator< tnlGrid< 1, MeshReal, Device, MeshIndex >, OperatorQ, Real, Index >::
updateLinearSystem( const RealType& time,
                    const RealType& tau,
                    const MeshType& mesh,
                    const IndexType& index,
                    const MeshEntity& entity,
                    Vector& u,
                    Vector& b,
                    MatrixRow& matrixRow ) const
{
   const typename EntityType::template NeighbourEntities< 1 >& neighbourEntities = entity.getNeighbourEntities();
   const RealType aCoef = - tau * operatorQ.getValueStriped(mesh, index, entity, u, time ) * mesh.getHxSquareInverse() / 
                       operatorQ.getValue(mesh, mesh.template getCellNextToCell< -1 >( index ), entity, u, time );
   const RealType bCoef = tau * operatorQ.getValueStriped(mesh, index, entity, u, time ) * ( mesh.getHxSquareInverse() / 
                          operatorQ.getValue(mesh, index, entity, u, time ) + mesh.getHxSquareInverse() / 
                          operatorQ.getValue(mesh, mesh.template getCellNextToCell< -1 >( index ), entity, u, time ) );
   const RealType cCoef = - tau * operatorQ.getValueStriped(mesh, index, entity, u, time ) * mesh.getHxSquareInverse() / 
                       operatorQ.getValue(mesh, index, entity, u, time );
   matrixRow.setElement( 0, mesh.template getCellNextToCell< -1 >( index ),     aCoef );
   matrixRow.setElement( 1, index,                             bCoef );
   matrixRow.setElement( 2, mesh.template getCellNextToCell< 1 >( index ),      cCoef );
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index,
	  typename OperatorQ >
tnlString
tnlOneSideDiffNonlinearOperator< tnlGrid< 2, MeshReal, Device, MeshIndex >, OperatorQ, Real, Index >::
getType()
{
   return tnlString( "tnlOneSideDiffNonlinearOperator< " ) +
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
tnlOneSideDiffNonlinearOperator< tnlGrid< 2, MeshReal, Device, MeshIndex >, OperatorQ, Real, Index >::
getValue( const MeshType& mesh,
          const IndexType cellIndex,
          const MeshEntity& entity,
          const Vector& u,
          const Real& time ) const
{
   const typename EntityType::template NeighbourEntities< 2 >& neighbourEntities = entity.getNeighbourEntities();
   return operatorQ.getValueStriped(mesh,cellIndex, entity, u, time)*(((u[ mesh.template getCellNextToCell< 1,0 >( cellIndex ) ] - u[ cellIndex ]) 
          * mesh.getHxInverse()/operatorQ.getValue(mesh,cellIndex, entity, u, time)
          -( - u[ mesh.template getCellNextToCell< -1,0 >( cellIndex ) ]+ u[ cellIndex ]) * mesh.getHxInverse()/
          operatorQ.getValue(mesh,mesh.template getCellNextToCell<-1,0>(cellIndex), entity, u, time))
          *mesh.getHxInverse()+(( u[ mesh.template getCellNextToCell< 0,1 >( cellIndex ) ]  - u[ cellIndex ]) 
          * mesh.getHyInverse()/operatorQ.getValue(mesh,cellIndex, entity, u, time)
          -( - u[ mesh.template getCellNextToCell< 0,-1 >( cellIndex ) ] + u[ cellIndex ]) * mesh.getHyInverse()
          /operatorQ.getValue(mesh,mesh.template getCellNextToCell<0,-1>(cellIndex),entity, u, time))*mesh.getHyInverse());
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
tnlOneSideDiffNonlinearOperator< tnlGrid< 2, MeshReal, Device, MeshIndex >, OperatorQ, Real, Index >::
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
tnlOneSideDiffNonlinearOperator< tnlGrid< 2, MeshReal, Device, MeshIndex >, OperatorQ, Real, Index >::
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
   const RealType aCoef = - tau * operatorQ.getValueStriped(mesh, index, entity, u, time ) * mesh.getHySquareInverse() / 
                       operatorQ.getValue(mesh, mesh.template getCellNextToCell< 0,-1 >( index ), entity, u, time );
   const RealType bCoef = - tau * operatorQ.getValueStriped(mesh, index, entity, u, time ) * mesh.getHxSquareInverse() / 
                       operatorQ.getValue(mesh, mesh.template getCellNextToCell< -1,0 >( index ), entity, u, time );
   const RealType cCoef = tau * operatorQ.getValueStriped(mesh, index, entity, u, time ) * ( mesh.getHySquareInverse() / 
                       operatorQ.getValue(mesh, index, entity, u, time ) + mesh.getHySquareInverse() / 
                       operatorQ.getValue(mesh, mesh.template getCellNextToCell< 0,-1 >( index ), entity, u, time )
                       + mesh.getHxSquareInverse() / operatorQ.getValue(mesh, index, entity, u, time ) + 
                       mesh.getHxSquareInverse() / operatorQ.getValue(mesh, mesh.template getCellNextToCell< -1,0 >( index ), entity, u, time ) );
   const RealType dCoef = - tau * operatorQ.getValueStriped(mesh, index, entity, u, time ) * mesh.getHxSquareInverse() / 
                       operatorQ.getValue(mesh, index, entity, u, time );
   const RealType eCoef = - tau * operatorQ.getValueStriped(mesh, index, entity, u, time ) * mesh.getHySquareInverse() / 
                       operatorQ.getValue(mesh, index, entity, u, time );
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
tnlOneSideDiffNonlinearOperator< tnlGrid< 3, MeshReal, Device, MeshIndex >, OperatorQ, Real, Index >::
getType()
{
   return tnlString( "tnlOneSideDiffNonlinearOperator< " ) +
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
tnlOneSideDiffNonlinearOperator< tnlGrid< 3, MeshReal, Device, MeshIndex >, OperatorQ, Real, Index >::
getValue( const MeshType& mesh,
          const IndexType cellIndex,
          const MeshEntity& entity,
          const Vector& u,
          const Real& time ) const
{
   const typename EntityType::template NeighbourEntities< 3 >& neighbourEntities = entity.getNeighbourEntities();
   return operatorQ.getValueStriped(mesh,cellIndex, entity, u, time)*(((u[ mesh.template getCellNextToCell< 1,0,0 >( cellIndex ) ] - u[ cellIndex ]) 
          * mesh.getHxInverse()/operatorQ.getValue(mesh,cellIndex, entity, u, time)
          -( - u[ mesh.template getCellNextToCell< -1,0,0 >( cellIndex ) ]+ u[ cellIndex ]) * mesh.getHxInverse()/
          operatorQ.getValue(mesh,mesh.template getCellNextToCell<-1,0,0>(cellIndex), entity, u, time))
          *mesh.getHxInverse()+(( u[ mesh.template getCellNextToCell< 0,1,0 >( cellIndex ) ]  - u[ cellIndex ]) 
          * mesh.getHyInverse()/operatorQ.getValue(mesh,cellIndex, entity, u, time)
          -( - u[ mesh.template getCellNextToCell< 0,-1,0 >( cellIndex ) ] + u[ cellIndex ]) * mesh.getHyInverse()
          /operatorQ.getValue(mesh,mesh.template getCellNextToCell<0,-1,0>(cellIndex),entity, u, time))*mesh.getHyInverse()
          +(( u[ mesh.template getCellNextToCell< 0,0,1 >( cellIndex ) ]  - u[ cellIndex ]) 
          * mesh.getHzInverse()/operatorQ.getValue(mesh,cellIndex, entity, u, time)
          -( - u[ mesh.template getCellNextToCell< 0,0,-1 >( cellIndex ) ] + u[ cellIndex ]) * mesh.getHzInverse()
          /operatorQ.getValue(mesh,mesh.template getCellNextToCell<0,0,-1>(cellIndex),entity, u, time))*mesh.getHzInverse());
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
tnlOneSideDiffNonlinearOperator< tnlGrid< 3, MeshReal, Device, MeshIndex >, OperatorQ, Real, Index >::
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
__cuda_callable__
void
tnlOneSideDiffNonlinearOperator< tnlGrid< 3, MeshReal, Device, MeshIndex >, OperatorQ, Real, Index >::
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
   const RealType aCoef = - tau * operatorQ.getValueStriped(mesh, index, entity, u, time ) * mesh.getHzSquareInverse() / 
                       operatorQ.getValue(mesh, mesh.template getCellNextToCell< 0,0,-1 >( index ), entity, u, time );
   const RealType bCoef = - tau * operatorQ.getValueStriped(mesh, index, entity, u, time ) * mesh.getHySquareInverse() / 
                       operatorQ.getValue(mesh, mesh.template getCellNextToCell< 0,-1,0 >( index ), entity, u, time );
   const RealType cCoef = - tau * operatorQ.getValueStriped(mesh, index, entity, u, time ) * mesh.getHxSquareInverse() / 
                       operatorQ.getValue(mesh, mesh.template getCellNextToCell< -1,0,0 >( index ), entity, u, time );
   const RealType dCoef = tau * operatorQ.getValueStriped(mesh, index, entity, u, time ) * ( mesh.getHySquareInverse() / 
                       operatorQ.getValue(mesh, index, entity, u, time ) + mesh.getHySquareInverse() / 
                       operatorQ.getValue(mesh, mesh.template getCellNextToCell< 0,-1,0 >( index ), entity, u, time )
                       + mesh.getHxSquareInverse() / operatorQ.getValue(mesh, index, entity, u, time ) + 
                       mesh.getHxSquareInverse() / operatorQ.getValue(mesh, mesh.template getCellNextToCell< -1,0,0 >( index ), entity, u, time )
                       + mesh.getHzSquareInverse() / operatorQ.getValue(mesh, index, entity, u, time ) + 
                       mesh.getHzSquareInverse() / operatorQ.getValue(mesh, mesh.template getCellNextToCell< 0,0,-1 >( index ), entity, u, time ) );
   const RealType eCoef = - tau * operatorQ.getValueStriped(mesh, index, entity, u, time ) * mesh.getHxSquareInverse() / 
                       operatorQ.getValue(mesh, index, entity, u, time );
   const RealType fCoef = - tau * operatorQ.getValueStriped(mesh, index, entity, u, time ) * mesh.getHySquareInverse() / 
                       operatorQ.getValue(mesh, index, entity, u, time );
   const RealType gCoef = - tau * operatorQ.getValueStriped(mesh, index, entity, u, time ) * mesh.getHzSquareInverse() / 
                       operatorQ.getValue(mesh, index, entity, u, time );
   matrixRow.setElement( 0, mesh.template getCellNextToCell< 0,0,-1 >( index ),     aCoef );
   matrixRow.setElement( 1, mesh.template getCellNextToCell< 0,-1,0 >( index ),     bCoef );
   matrixRow.setElement( 2, mesh.template getCellNextToCell< -1,0,0 >( index ),     cCoef );
   matrixRow.setElement( 3, index,                                                  dCoef );
   matrixRow.setElement( 4, mesh.template getCellNextToCell< 1,0,0 >( index ),      eCoef );
   matrixRow.setElement( 5, mesh.template getCellNextToCell< 0,1,0 >( index ),      fCoef );
   matrixRow.setElement( 6, mesh.template getCellNextToCell< 0,0,1 >( index ),      gCoef );
}
#endif	/* TNLONESIDEDIFFNONLINEAROPERATOR_IMPL_H */
