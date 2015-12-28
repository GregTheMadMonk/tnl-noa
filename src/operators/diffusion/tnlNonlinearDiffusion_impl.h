
#ifndef TNLNONLINEARDIFFUSION_IMPL_H
#define	TNLNONLINEARDIFFUSION_IMPL_H

#include "tnlNonlinearDiffusion.h"

#include <mesh/tnlGrid.h>

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index,
          typename NonlinearDiffusionOperator >
tnlString
tnlNonlinearDiffusion< tnlGrid< 1, MeshReal, Device, MeshIndex >, NonlinearDiffusionOperator, Real, Index >::
getType()
{
   return tnlString( "tnlNonlinearDiffusion< " ) +
          MeshType::getType() + ", " +
          ::getType< Real >() + ", " +
          ::getType< Index >() + "," +
          NonlinearDiffusionOperator::getType() + " >";
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index,
          typename NonlinearDiffusionOperator >
template< typename MeshEntity,
          typename Vector >
__cuda_callable__
Real
tnlNonlinearDiffusion< tnlGrid< 1, MeshReal, Device, MeshIndex >, NonlinearDiffusionOperator, Real, Index >::
getValue( const MeshType& mesh,
          const MeshEntity& entity,
          const Vector& u,
          const Real& time ) const
{
    return nonlinearDiffusionOperator.getValue( mesh, entity, u, time );
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index,
          typename NonlinearDiffusionOperator >
   template< typename MeshEntity >          
__cuda_callable__
Index
tnlNonlinearDiffusion< tnlGrid< 1, MeshReal, Device, MeshIndex >, NonlinearDiffusionOperator, Real, Index >::
getLinearSystemRowLength( const MeshType& mesh,
                          const IndexType& index,
                          const MeshEntity& entity ) const
{
   return nonlinearDiffusionOperator.getLinearSystemRowLength( mesh, index, entity );
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index,
          typename NonlinearDiffusionOperator >
template< typename MeshEntity,
          typename Vector,           
          typename MatrixRow >
__cuda_callable__
void
tnlNonlinearDiffusion< tnlGrid< 1, MeshReal, Device, MeshIndex >, NonlinearDiffusionOperator, Real, Index >::
updateLinearSystem( const RealType& time,
                    const RealType& tau,
                    const MeshType& mesh,
                    const IndexType& index,
                    const MeshEntity& entity,
                    Vector& u,
                    Vector& b,
                    MatrixRow& matrixRow ) const
{
    nonlinearDiffusionOperator.updateLinearSystem( time, tau, mesh, index, entity, u, b, matrixRow );
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index,
          typename NonlinearDiffusionOperator >
tnlString
tnlNonlinearDiffusion< tnlGrid< 2, MeshReal, Device, MeshIndex >, NonlinearDiffusionOperator, Real, Index >::
getType()
{
   return tnlString( "tnlNonlinearDiffusion< " ) +
          MeshType::getType() + ", " +
          ::getType< Real >() + ", " +
          ::getType< Index >() + "," +
          NonlinearDiffusionOperator::getType() + " >";
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index,
          typename NonlinearDiffusionOperator >
template< typename MeshEntity,
          typename Vector >
__cuda_callable__
Real
tnlNonlinearDiffusion< tnlGrid< 2, MeshReal, Device, MeshIndex >, NonlinearDiffusionOperator, Real, Index >::
getValue( const MeshType& mesh,
          const MeshEntity& entity,
          const Vector& u,
          const Real& time ) const
{
    return nonlinearDiffusionOperator.getValue( mesh, entity, u, time );
}
       
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index,
          typename NonlinearDiffusionOperator >
   template< typename MeshEntity >
__cuda_callable__
Index
tnlNonlinearDiffusion< tnlGrid< 2, MeshReal, Device, MeshIndex >, NonlinearDiffusionOperator, Real, Index >::
getLinearSystemRowLength( const MeshType& mesh,
                          const IndexType& index,
                          const MeshEntity& entity ) const
{
   return nonlinearDiffusionOperator.getLinearSystemRowLength( mesh, index, entity );
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index,
          typename NonlinearDiffusionOperator >
template< typename MeshEntity,
          typename Vector,           
          typename MatrixRow >
__cuda_callable__
void
tnlNonlinearDiffusion< tnlGrid< 2, MeshReal, Device, MeshIndex >, NonlinearDiffusionOperator, Real, Index >::
updateLinearSystem( const RealType& time,
                    const RealType& tau,
                    const MeshType& mesh,
                    const IndexType& index,
                    const MeshEntity& entity,
                    Vector& u,
                    Vector& b,
                    MatrixRow& matrixRow ) const
{
    nonlinearDiffusionOperator.updateLinearSystem( time, tau, mesh, index, entity, u, b, matrixRow );
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index,
          typename NonlinearDiffusionOperator >
tnlString
tnlNonlinearDiffusion< tnlGrid< 3, MeshReal, Device, MeshIndex >, NonlinearDiffusionOperator, Real, Index >::
getType()
{
   return tnlString( "tnlNonlinearDiffusion< " ) +
          MeshType::getType() + ", " +
          ::getType< Real >() + ", " +
          ::getType< Index >() + "," +
          NonlinearDiffusionOperator::getType() + " >";
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index,
          typename NonlinearDiffusionOperator >
template< typename MeshEntity,
          typename Vector >
__cuda_callable__
Real
tnlNonlinearDiffusion< tnlGrid< 3, MeshReal, Device, MeshIndex >, NonlinearDiffusionOperator, Real, Index >::
getValue( const MeshType& mesh,
          const MeshEntity& entity,
          const Vector& u,
          const Real& time ) const
{
    return nonlinearDiffusionOperator.getValue( mesh, entity, u, time );
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index,
          typename NonlinearDiffusionOperator >
   template< typename MeshEntity >
__cuda_callable__
Index
tnlNonlinearDiffusion< tnlGrid< 3, MeshReal, Device, MeshIndex >, NonlinearDiffusionOperator, Real, Index >::
getLinearSystemRowLength( const MeshType& mesh,
                          const IndexType& index,
                          const MeshEntity& entity ) const
{
   return nonlinearDiffusionOperator.getLinearSystemRowLength( mesh, index, entity );
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index,
          typename NonlinearDiffusionOperator >
template< typename MeshEntity,
          typename Vector,          
          typename MatrixRow >
__cuda_callable__
void
tnlNonlinearDiffusion< tnlGrid< 3, MeshReal, Device, MeshIndex >, NonlinearDiffusionOperator, Real, Index >::
updateLinearSystem( const RealType& time,
                    const RealType& tau,
                    const MeshType& mesh,
                    const IndexType& index,
                    const MeshEntity& entity,
                    Vector& u,
                    Vector& b,
                    MatrixRow& matrixRow ) const
{
    nonlinearDiffusionOperator.updateLinearSystem( time, tau, mesh, index, entity, u, b, matrixRow );
}

#endif	/* TNLNONLINEARDIFFUSION_IMPL_H */
