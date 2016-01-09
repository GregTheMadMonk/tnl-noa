
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
getValue( const MeshEntity& entity,
          const Vector& u,
          const Real& time ) const
{
    return nonlinearDiffusionOperator.getValue( u, entity, time );
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
          typename MeshFunction,
          typename Vector,           
          typename Matrix >
__cuda_callable__
void
tnlNonlinearDiffusion< tnlGrid< 1, MeshReal, Device, MeshIndex >, NonlinearDiffusionOperator, Real, Index >::
updateLinearSystem( const RealType& time,
                    const RealType& tau,
                    const MeshType& mesh,
                    const IndexType& index,
                    const MeshEntity& entity,
                    const MeshFunction& u,
                    Vector& b,
                    Matrix& matrix ) const
{
    nonlinearDiffusionOperator.updateLinearSystem( time, tau, mesh, index, entity, u, b, matrix );
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
getValue( const MeshEntity& entity,
          const Vector& u,
          const Real& time ) const
{
    return nonlinearDiffusionOperator.getValue( u, entity, time );
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
          typename MeshFunction,
          typename Vector,           
          typename Matrix >
__cuda_callable__
void
tnlNonlinearDiffusion< tnlGrid< 2, MeshReal, Device, MeshIndex >, NonlinearDiffusionOperator, Real, Index >::
updateLinearSystem( const RealType& time,
                    const RealType& tau,
                    const MeshType& mesh,
                    const IndexType& index,
                    const MeshEntity& entity,
                    const MeshFunction& u,
                    Vector& b,
                    Matrix& matrix ) const
{
    nonlinearDiffusionOperator.updateLinearSystem( time, tau, mesh, index, entity, u, b, matrix );
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
getValue( const MeshEntity& entity,
          const Vector& u,
          const Real& time ) const
{
    return nonlinearDiffusionOperator.getValue( u, entity, time );
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
          typename MeshFunction,
          typename Vector,          
          typename Matrix >
__cuda_callable__
void
tnlNonlinearDiffusion< tnlGrid< 3, MeshReal, Device, MeshIndex >, NonlinearDiffusionOperator, Real, Index >::
updateLinearSystem( const RealType& time,
                    const RealType& tau,
                    const MeshType& mesh,
                    const IndexType& index,
                    const MeshEntity& entity,
                    const MeshFunction& u,
                    Vector& b,
                    Matrix& matrix ) const
{
    nonlinearDiffusionOperator.updateLinearSystem( time, tau, mesh, index, entity, u, b, matrix );
}

#endif	/* TNLNONLINEARDIFFUSION_IMPL_H */
