/***************************************************************************
                          Traverser.h  -  description
                             -------------------
    begin                : Feb 17, 2016
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

/***
 * Authors:
 * Oberhuber Tomas, tomas.oberhuber@fjfi.cvut.cz
 * Szekely Ondrej, ondra.szekely@gmail.com
 */

#pragma once

#include "FiniteVolumeNonlinearOperator.h"
#include <TNL/Meshes/Grid.h>

namespace TNL {
namespace Operators {   

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index,
          typename OperatorQ >
String
FiniteVolumeNonlinearOperator< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, OperatorQ, Real, Index >::
getType()
{
   return String( "FiniteVolumeNonlinearOperator< " ) +
          MeshType::getType() + ", " +
         TNL::getType< Real >() + ", " +
         TNL::getType< Index >() + ", " +
	  OperatorQ::getType() + " >";
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index,
          typename OperatorQ >
template< typename MeshEntity,
          typename Vector >
__cuda_callable__
Real
FiniteVolumeNonlinearOperator< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, OperatorQ, Real, Index >::
operator()( const MeshEntity& entity,
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
FiniteVolumeNonlinearOperator< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, OperatorQ, Real, Index >::
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
template< typename MeshEntity,
          typename MeshFunction,
          typename Vector,
          typename Matrix >
__cuda_callable__
void
FiniteVolumeNonlinearOperator< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, OperatorQ, Real, Index >::
setMatrixElements( const RealType& time,
                    const RealType& tau,
                    const MeshType& mesh,
                    const IndexType& index,
                    const MeshEntity& entity,
                    const MeshFunction& u,
                    Vector& b,
                    Matrix& matrix ) const
{
   typename Matrix::MatrixRow matrixRow = matrix.getRow( index );
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index,
	  typename OperatorQ >
String
FiniteVolumeNonlinearOperator< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, OperatorQ, Real, Index >::
getType()
{
   return String( "FiniteVolumeNonlinearOperator< " ) +
          MeshType::getType() + ", " +
         TNL::getType< Real >() + ", " +
         TNL::getType< Index >() + ", " +
	  OperatorQ::getType() + " >";
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index,
          typename OperatorQ >
template< typename MeshEntity,
          typename Vector >
__cuda_callable__
Real
FiniteVolumeNonlinearOperator< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, OperatorQ, Real, Index >::
operator()( const MeshEntity& entity,
            const Vector& u,
            const Real& time ) const
{
   const typename MeshEntity::template NeighbourEntities< 2 >& neighbourEntities = entity.getNeighbourEntities();      
   const typename MeshEntity::MeshType& mesh = entity.getMesh();
   const IndexType& cellIndex = entity.getIndex();
   return operatorQ( entity, u, time ) * 
      ( (  u[ neighbourEntities.template getEntityIndex<  1, 0 >() ] - u[ cellIndex ] ) * mesh.template getSpaceStepsProducts< -2, 0 >() / operatorQ.operator()( entity, u, time, 1 )
      + (  u[ neighbourEntities.template getEntityIndex<  0, 1 >() ] - u[ cellIndex ] ) * mesh.template getSpaceStepsProducts< 0, -2 >() / operatorQ.operator()( entity, u, time, 0, 1 ) 
      - ( -u[ neighbourEntities.template getEntityIndex< -1, 0 >() ] + u[ cellIndex ] ) * mesh.template getSpaceStepsProducts< -2, 0 >() / operatorQ.operator()( entity, u, time, -1)
      - ( -u[ neighbourEntities.template getEntityIndex<  0,-1 >() ] + u[ cellIndex ] ) * mesh.template getSpaceStepsProducts< 0, -2 >() / operatorQ.operator()( entity, u, time, 0, -1) );
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
FiniteVolumeNonlinearOperator< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, OperatorQ, Real, Index >::
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
template< typename MeshEntity, 
          typename MeshFunction,
          typename Vector,
          typename Matrix >
__cuda_callable__
void
FiniteVolumeNonlinearOperator< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, OperatorQ, Real, Index >::
setMatrixElements( const RealType& time,
                    const RealType& tau,
                    const MeshType& mesh,
                    const IndexType& index,
                    const MeshEntity& entity,
                    const MeshFunction& u,
                    Vector& b,
                    Matrix& matrix ) const
{
   typename Matrix::MatrixRow matrixRow = matrix.getRow( index );
   const typename MeshEntity::template NeighbourEntities< 2 >& neighbourEntities = entity.getNeighbourEntities();
   const RealType aCoef = - tau * operatorQ.operator()( entity, u, time ) * mesh.template getSpaceStepsProducts< 0, -2 >() / 
                       operatorQ.operator()( entity, u, time, 0, -1 );
   const RealType bCoef = - tau * operatorQ.operator()( entity, u, time ) * mesh.template getSpaceStepsProducts< -2, 0 >() / 
                       operatorQ.operator()( entity, u, time, -1 );
   const RealType cCoef = tau * operatorQ.operator()( entity, u, time ) * ( mesh.template getSpaceStepsProducts< -2, 0 >() / 
                       operatorQ.operator()( entity, u, time, 1 ) + mesh.template getSpaceStepsProducts< 0, -2 >() / 
                       operatorQ.operator()( entity, u, time, 0, 1 )
                       + mesh.template getSpaceStepsProducts< -2, 0 >() / operatorQ.operator()( entity, u, time, -1 ) + 
                       mesh.template getSpaceStepsProducts< 0, -2 >() / operatorQ.operator()( entity, u, time, 0, -1 ) );
   const RealType dCoef = - tau * operatorQ.operator()( entity, u, time ) * mesh.template getSpaceStepsProducts< -2, 0 >() / 
                       operatorQ.operator()( entity, u, time, 1 );
   const RealType eCoef = - tau * operatorQ.operator()( entity, u, time ) * mesh.template getSpaceStepsProducts< 0, -2 >() / 
                       operatorQ.operator()(  entity, u, time, 0, 1 );
   matrixRow.setElement( 0, neighbourEntities.template getEntityIndex<  0, -1 >(), aCoef );
   matrixRow.setElement( 1, neighbourEntities.template getEntityIndex< -1,  0 >(), bCoef );
   matrixRow.setElement( 2, entity.getIndex(),                                     cCoef );
   matrixRow.setElement( 3, neighbourEntities.template getEntityIndex<  1,  0 >(), dCoef );
   matrixRow.setElement( 4, neighbourEntities.template getEntityIndex<  0,  1 >(), eCoef );
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index,
 	  typename OperatorQ >
String
FiniteVolumeNonlinearOperator< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, OperatorQ, Real, Index >::
getType()
{
   return String( "FiniteVolumeNonlinearOperator< " ) +
          MeshType::getType() + ", " +
         TNL::getType< Real >() + ", " +
         TNL::getType< Index >() + ", " +
	  OperatorQ::getType() + " >";
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index,
          typename OperatorQ >
template< typename MeshEntity,
          typename Vector >
__cuda_callable__
Real
FiniteVolumeNonlinearOperator< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, OperatorQ, Real, Index >::
operator()( const MeshEntity& entity,
            const Vector& u,
            const Real& time ) const
{
   const typename MeshEntity::template NeighbourEntities< 3 >& neighbourEntities = entity.getNeighbourEntities();
   const typename MeshEntity::MeshType& mesh = entity.getMesh();
   const IndexType& cellIndex = entity.getIndex();
   return operatorQ( entity, u, time ) * 
      ( (u[ neighbourEntities.template getEntityIndex< 1,0,0 >() ] - u[ cellIndex ]) 
          * mesh.template getSpaceStepsProducts< -2, 0, 0 >() / operatorQ( entity, u, time, 1 )
          + ( u[ neighbourEntities.template getEntityIndex< 0,1,0 >() ] - u[ cellIndex ]) * mesh.template getSpaceStepsProducts< 0, -2, 0 >()/
          operatorQ( entity, u, time, 0, 1 ) 
          + ( u[ neighbourEntities.template getEntityIndex< 0,0,1 >() ] - u[ cellIndex ]) * mesh.template getSpaceStepsProducts< 0, 0, -2 >()/
          operatorQ( entity, u, time, 0, 0, 1 ) 
          - ( - u[ neighbourEntities.template getEntityIndex< -1,0,0 >() ]  + u[ cellIndex ]) 
          * mesh.template getSpaceStepsProducts< -2, 0, 0 >() / operatorQ( entity, u, time, -1)
          -( - u[ neighbourEntities.template getEntityIndex< 0,-1,0 >() ] + u[ cellIndex ]) * mesh.template getSpaceStepsProducts< 0, -2, 0 >()
          /operatorQ( entity, u, time, 0, -1) 
          -( - u[ neighbourEntities.template getEntityIndex< 0,0,-1 >() ] + u[ cellIndex ]) * mesh.template getSpaceStepsProducts< 0, 0, -2 >()
          /operatorQ( entity, u, time, 0, 0, -1) );
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
FiniteVolumeNonlinearOperator< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, OperatorQ, Real, Index >::
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
template< typename MeshEntity,
          typename MeshFunction,
          typename Vector,
          typename Matrix >
#ifdef HAVE_CUDA
__cuda_callable__
#endif
void
FiniteVolumeNonlinearOperator< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, OperatorQ, Real, Index >::
setMatrixElements( const RealType& time,
                    const RealType& tau,
                    const MeshType& mesh,
                    const IndexType& index,
                    const MeshEntity& entity,
                    const MeshFunction& u,
                    Vector& b,
                    Matrix& matrix ) const
{
   typename Matrix::MatrixRow matrixRow = matrix.getRow( index );
   const typename MeshEntity::template NeighbourEntities< 3 >& neighbourEntities = entity.getNeighbourEntities();
   const RealType aCoef = - tau * operatorQ( entity, u, time ) *
                       mesh.template getSpaceStepsProducts< 0, 0, -2 >() / operatorQ.operator()( entity, u, time, 0, 0, -1 );
   const RealType bCoef = - tau * operatorQ( entity, u, time ) * 
                       mesh.template getSpaceStepsProducts< 0, -2, 0 >() / operatorQ.operator()( entity, u, time, 0, -1, 0 );
   const RealType cCoef = - tau * operatorQ( entity, u, time ) * 
                       mesh.template getSpaceStepsProducts< -2, 0, 0 >() / operatorQ.operator()( entity, u, time, -1, 0, 0 );
   const RealType dCoef = tau * operatorQ( entity, u, time ) * ( 
                       mesh.template getSpaceStepsProducts< -2, 0, 0 >() / operatorQ.operator()( entity, u, time, 1, 0, 0 ) + 
                       mesh.template getSpaceStepsProducts< 0, -2, 0 >() / operatorQ.operator()( entity, u, time, 0, 1, 0 ) +
                       mesh.template getSpaceStepsProducts< 0, 0, -2 >() / operatorQ.operator()( entity, u, time, 0, 0, 1 ) + 
                       mesh.template getSpaceStepsProducts< -2, 0, 0 >() / operatorQ.operator()( entity, u, time, -1, 0, 0 ) +
                       mesh.template getSpaceStepsProducts< 0, -2, 0 >() / operatorQ.operator()( entity, u, time, 0, -1, 0 ) + 
                       mesh.template getSpaceStepsProducts< 0, 0, -2 >() / operatorQ.operator()( entity, u, time, 0, 0, -1 ) );
   const RealType eCoef = - tau * operatorQ.operator()( entity, u, time ) *
                       mesh.template getSpaceStepsProducts< -2, 0, 0 >() / operatorQ.operator()( entity, u, time, 1, 0, 0 );
   const RealType fCoef = - tau * operatorQ.operator()( entity, u, time ) *
                       mesh.template getSpaceStepsProducts< 0, -2, 0 >() / operatorQ.operator()( entity, u, time, 0, 1, 0 );
   const RealType gCoef = - tau * operatorQ.operator()( entity, u, time ) * 
                       mesh.template getSpaceStepsProducts< 0, 0, -2 >() / operatorQ.operator()( entity, u, time, 0, 0, 1 );
   matrixRow.setElement( 0, neighbourEntities.template getEntityIndex< 0,0,-1 >(), aCoef );
   matrixRow.setElement( 1, neighbourEntities.template getEntityIndex< 0,-1,0 >(), bCoef );
   matrixRow.setElement( 2, neighbourEntities.template getEntityIndex< -1,0,0 >(), cCoef );
   matrixRow.setElement( 3, entity.getIndex(),                                     dCoef );
   matrixRow.setElement( 4, neighbourEntities.template getEntityIndex< 1,0,0 >(),  eCoef );
   matrixRow.setElement( 5, neighbourEntities.template getEntityIndex< 0,1,0 >(),  fCoef );
   matrixRow.setElement( 6, neighbourEntities.template getEntityIndex< 0,0,1 >(),  gCoef );
}

} // namespace Operators
} // namespace TNL
