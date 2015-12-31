
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
getValue( const MeshEntity& entity,
          const Vector& u,
          const Real& time ) const
{
   const typename MeshEntity::template NeighbourEntities< 1 >& neighbourEntities = entity.getNeighbourEntities();
   const typename MeshEntity::MeshType& mesh = entity.getMesh();
   const IndexType& cellIndex = entity.getIndex();
   return operatorQ.getValueStriped( entity, u, time )*
      ( ( (  u[ neighbourEntities.template getEntityIndex<  1 >() ] - u[ cellIndex ] ) * mesh.template getSpaceStepsProducts< -1 >() / operatorQ.getValue( entity, u, time ) -
          ( -u[ neighbourEntities.template getEntityIndex< -1 >() ] + u[ cellIndex ] ) * mesh.template getSpaceStepsProducts< -1 >() / operatorQ.getValue( neighbourEntities.template getEntity< -1 >(), u, time ) 
        ) * mesh.template getSpaceStepsProducts< -1 >() );
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
          typename Matrix >
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
                    Matrix& matrix ) const
{
   typename Matrix::MatrixRow matrixRow = matrix.getRow( index );
   const typename MeshEntity::template NeighbourEntities< 1 >& neighbourEntities = entity.getNeighbourEntities();
   const RealType aCoef = -tau * operatorQ.getValueStriped( entity, u, time ) *
      mesh.template getSpaceStepsProducts< -2 >() / operatorQ.getValue( neighbourEntities.template getEntity< -1 >(), u, time );
   const RealType bCoef = tau * operatorQ.getValueStriped( entity, u, time ) * 
      ( mesh.template getSpaceStepsProducts< -2 >() / operatorQ.getValue( entity, u, time ) + 
        mesh.template getSpaceStepsProducts< -2 >() / operatorQ.getValue( neighbourEntities.template getEntity< -1 >(), u, time ) );
   const RealType cCoef = -tau * operatorQ.getValueStriped( entity, u, time ) * 
      mesh.template getSpaceStepsProducts< -2 >() / operatorQ.getValue( entity, u, time );
   matrixRow.setElement( 0, neighbourEntities.template getEntityIndex< -1 >(), aCoef );
   matrixRow.setElement( 1, entity.getIndex(),                                 bCoef );
   matrixRow.setElement( 2, neighbourEntities.template getEntityIndex<  1 >(), cCoef );
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
getValue( const MeshEntity& entity,
          const Vector& u,
          const Real& time ) const
{
   const IndexType& cellIndex = entity.getIndex();
   const typename MeshEntity::template NeighbourEntities< 2 >& neighbourEntities = entity.getNeighbourEntities();
   const typename MeshEntity::MeshType& mesh = entity.getMesh();
   return operatorQ.getValueStriped( entity, u, time ) *
      ( ( (  u[ neighbourEntities.template getEntityIndex<  1,  0 >() ] - u[ cellIndex ] ) * mesh.template getSpaceStepsProducts< -1, 0 >() / operatorQ.getValue( entity, u, time )
         -( -u[ neighbourEntities.template getEntityIndex< -1,  0 >() ] + u[ cellIndex ] ) * mesh.template getSpaceStepsProducts< -1, 0 >() / operatorQ.getValue( neighbourEntities.template getEntity< -1,  0 >(), u, time ) )*mesh.template getSpaceStepsProducts< -1, 0 >() 
       +( (  u[ neighbourEntities.template getEntityIndex<  0,  1 >() ] - u[ cellIndex ] ) * mesh.template getSpaceStepsProducts< 0, -1 >() / operatorQ.getValue( entity, u, time )
         -( -u[ neighbourEntities.template getEntityIndex<  0, -1 >() ] + u[ cellIndex ] ) * mesh.template getSpaceStepsProducts< 0, -1 >() / operatorQ.getValue( neighbourEntities.template getEntity<  0, -1 >(), u, time ) )*mesh.template getSpaceStepsProducts< 0, -1 >());
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
          typename Matrix >
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
                    Matrix& matrix ) const
{
   typename Matrix::MatrixRow matrixRow = matrix.getRow( index );
   const typename MeshEntity::template NeighbourEntities< 2 >& neighbourEntities = entity.getNeighbourEntities();
   const RealType aCoef = -tau * operatorQ.getValueStriped( entity, u, time ) *
      mesh.template getSpaceStepsProducts< 0, -2 >() / operatorQ.getValue( neighbourEntities.template getEntity< 0,-1 >(), u, time );
   const RealType bCoef = -tau * operatorQ.getValueStriped( entity, u, time ) * 
      mesh.template getSpaceStepsProducts< -2, 0 >() / operatorQ.getValue( neighbourEntities.template getEntity< -1,0 >(), u, time );
   const RealType cCoef = tau * operatorQ.getValueStriped( entity, u, time ) * 
      ( mesh.template getSpaceStepsProducts< 0, -2 >() / operatorQ.getValue( entity, u, time ) + 
        mesh.template getSpaceStepsProducts< 0, -2 >() / operatorQ.getValue( neighbourEntities.template getEntity< 0,-1 >(), u, time ) +
        mesh.template getSpaceStepsProducts< -2, 0 >() / operatorQ.getValue( entity, u, time ) + 
        mesh.template getSpaceStepsProducts< -2, 0 >() / operatorQ.getValue( neighbourEntities.template getEntity< -1,0 >(), u, time ) );
   const RealType dCoef = -tau * operatorQ.getValueStriped( entity, u, time ) * 
      mesh.template getSpaceStepsProducts< -2, 0 >() / operatorQ.getValue( entity, u, time );
   const RealType eCoef = -tau * operatorQ.getValueStriped( entity, u, time ) * 
      mesh.template getSpaceStepsProducts< 0, -2 >() / operatorQ.getValue( entity, u, time );
   matrixRow.setElement( 0, neighbourEntities.template getEntityIndex<  0, -1 >(), aCoef );
   matrixRow.setElement( 1, neighbourEntities.template getEntityIndex< -1,  0 >(), bCoef );
   matrixRow.setElement( 2, entity.getIndex()                                    , cCoef );
   matrixRow.setElement( 3, neighbourEntities.template getEntityIndex<  1,  0 >(), dCoef );
   matrixRow.setElement( 4, neighbourEntities.template getEntityIndex<  0,  1 >(), eCoef );
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
getValue( const MeshEntity& entity,
          const Vector& u,
          const Real& time ) const
{
   const typename MeshEntity::template NeighbourEntities< 3 >& neighbourEntities = entity.getNeighbourEntities();
   const IndexType& cellIndex = entity.getIndex();
   const typename MeshEntity::MeshType& mesh = entity.getMesh();
   return operatorQ.getValueStriped( entity, u, time ) *
      ( ( (  u[ neighbourEntities.template getEntityIndex<  1,  0, 0 >() ] - u[ cellIndex ] ) * mesh.template getSpaceStepsProducts< -1, 0, 0 >() / operatorQ.getValue( entity, u, time ) -
          ( -u[ neighbourEntities.template getEntityIndex< -1,  0, 0 >() ] + u[ cellIndex ] ) * mesh.template getSpaceStepsProducts< -1, 0, 0 >() / operatorQ.getValue( neighbourEntities.template getEntity< -1,  0,  0 >(), u, time ) ) * mesh.template getSpaceStepsProducts< -1, 0, 0 >() + 
        ( (  u[ neighbourEntities.template getEntityIndex<  0,  1, 0 >() ] - u[ cellIndex ] ) * mesh.template getSpaceStepsProducts< 0, -1,  0 >() / operatorQ.getValue( entity, u, time ) -
          ( -u[ neighbourEntities.template getEntityIndex<  0, -1, 0 >() ] + u[ cellIndex ] ) * mesh.template getSpaceStepsProducts< 0, -1,  0 >() / operatorQ.getValue( neighbourEntities.template getEntity<  0, -1,  0 >(), u, time ) ) * mesh.template getSpaceStepsProducts< 0, -1,  0 >() +
        ( (  u[ neighbourEntities.template getEntityIndex<  0,  0, 1 >() ] - u[ cellIndex ] ) * mesh.template getSpaceStepsProducts< 0, 0, -1 >() / operatorQ.getValue( entity, u, time ) -
          ( -u[ neighbourEntities.template getEntityIndex<  0,  0,-1 >() ] + u[ cellIndex ] ) * mesh.template getSpaceStepsProducts< 0, 0, -1 >() / operatorQ.getValue( neighbourEntities.template getEntity<  0,  0, -1 >(), u, time) ) * mesh.template getSpaceStepsProducts< 0, 0, -1 >() );
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
          typename Matrix >
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
                    Matrix& matrix ) const
{
   typename Matrix::MatrixRow matrixRow = matrix.getRow( index );
   const typename MeshEntity::template NeighbourEntities< 3 >& neighbourEntities = entity.getNeighbourEntities();
   const RealType aCoef = -tau * operatorQ.getValueStriped( entity, u, time ) * 
      mesh.template getSpaceStepsProducts< 0, 0, -2 >() / operatorQ.getValue( neighbourEntities.template getEntity<  0,  0, -1 >(), u, time );
   const RealType bCoef = -tau * operatorQ.getValueStriped( entity, u, time ) * 
      mesh.template getSpaceStepsProducts< 0, -2, 0 >() / operatorQ.getValue( neighbourEntities.template getEntity<  0, -1,  0 >(), u, time );
   const RealType cCoef = -tau * operatorQ.getValueStriped( entity, u, time ) *
      mesh.template getSpaceStepsProducts< -2, 0, 0 >() / operatorQ.getValue( neighbourEntities.template getEntity< -1,  0,  0 >(), u, time );
   const RealType dCoef = tau *  operatorQ.getValueStriped( entity, u, time ) * 
      ( mesh.template getSpaceStepsProducts< 0, -2, 0 >() / operatorQ.getValue( entity, u, time ) + 
        mesh.template getSpaceStepsProducts< 0, -2, 0 >() / operatorQ.getValue( neighbourEntities.template getEntity<  0, -1,  0 >(), u, time ) +
        mesh.template getSpaceStepsProducts< -2, 0, 0 >() / operatorQ.getValue( entity, u, time ) + 
        mesh.template getSpaceStepsProducts< -2, 0, 0 >() / operatorQ.getValue( neighbourEntities.template getEntity< -1,  0,  0 >(), u, time ) +
        mesh.template getSpaceStepsProducts< 0, 0, -2 >() / operatorQ.getValue( entity, u, time ) + 
        mesh.template getSpaceStepsProducts< 0, 0, -2 >() / operatorQ.getValue( neighbourEntities.template getEntity<  0,  0, -1 >(), u, time ) );
   const RealType eCoef = -tau * operatorQ.getValueStriped( entity, u, time ) *
      mesh.template getSpaceStepsProducts< -2, 0, 0 >() / operatorQ.getValue( entity, u, time );
   const RealType fCoef = -tau * operatorQ.getValueStriped( entity, u, time ) * 
      mesh.template getSpaceStepsProducts< 0, -2, 0 >() / operatorQ.getValue( entity, u, time );
   const RealType gCoef = -tau * operatorQ.getValueStriped( entity, u, time ) * 
      mesh.template getSpaceStepsProducts< 0, 0, -2 >() / operatorQ.getValue( entity, u, time );
   matrixRow.setElement( 0, neighbourEntities.template getEntityIndex<  0,  0, -1 >(), aCoef );
   matrixRow.setElement( 1, neighbourEntities.template getEntityIndex<  0, -1,  0 >(), bCoef );
   matrixRow.setElement( 2, neighbourEntities.template getEntityIndex< -1,  0,  0 >(), cCoef );
   matrixRow.setElement( 3, entity.getIndex(),                                         dCoef );
   matrixRow.setElement( 4, neighbourEntities.template getEntityIndex<  1,  0,  0 >(), eCoef );
   matrixRow.setElement( 5, neighbourEntities.template getEntityIndex<  0,  1,  0 >(), fCoef );
   matrixRow.setElement( 6, neighbourEntities.template getEntityIndex<  0,  0,  1 >(), gCoef );
}
#endif	/* TNLONESIDEDIFFNONLINEAROPERATOR_IMPL_H */
