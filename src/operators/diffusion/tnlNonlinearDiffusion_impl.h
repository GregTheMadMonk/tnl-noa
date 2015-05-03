
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
template< typename Vector >
#ifdef HAVE_CUDA
__device__ __host__
#endif
Real
tnlNonlinearDiffusion< tnlGrid< 1, MeshReal, Device, MeshIndex >, NonlinearDiffusionOperator, Real, Index >::
getValue( const MeshType& mesh,
          const IndexType cellIndex,
          const CoordinatesType& coordinates,
          const Vector& u,
          const Real& time ) const
{
    return nonlinearDiffusionOperator.getValue( mesh, cellIndex, coordinates, u, time );
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index,
          typename NonlinearDiffusionOperator >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
Index
tnlNonlinearDiffusion< tnlGrid< 1, MeshReal, Device, MeshIndex >, NonlinearDiffusionOperator, Real, Index >::
getLinearSystemRowLength( const MeshType& mesh,
                          const IndexType& index,
                          const CoordinatesType& coordinates ) const
{
   return nonlinearDiffusionOperator.getLinearSystemRowLength( mesh, index, coordinates );
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index,
          typename NonlinearDiffusionOperator >
template< typename Vector, typename MatrixRow >
#ifdef HAVE_CUDA
__device__ __host__
#endif
void
tnlNonlinearDiffusion< tnlGrid< 1, MeshReal, Device, MeshIndex >, NonlinearDiffusionOperator, Real, Index >::
updateLinearSystem( const RealType& time,
                    const RealType& tau,
                    const MeshType& mesh,
                    const IndexType& index,
                    const CoordinatesType& coordinates,
                    Vector& u,
                    Vector& b,
                    MatrixRow& matrixRow ) const
{
    nonlinearDiffusionOperator.updateLinearSystem( time, tau, mesh, index, coordinates, u, b, matrixRow );
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
template< typename Vector >
#ifdef HAVE_CUDA
__device__ __host__
#endif
Real
tnlNonlinearDiffusion< tnlGrid< 2, MeshReal, Device, MeshIndex >, NonlinearDiffusionOperator, Real, Index >::
getValue( const MeshType& mesh,
          const IndexType cellIndex,
          const CoordinatesType& coordinates,
          const Vector& u,
          const Real& time ) const
{
    return nonlinearDiffusionOperator.getValue( mesh, cellIndex, coordinates, u, time );
}
       
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index,
          typename NonlinearDiffusionOperator >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
Index
tnlNonlinearDiffusion< tnlGrid< 2, MeshReal, Device, MeshIndex >, NonlinearDiffusionOperator, Real, Index >::
getLinearSystemRowLength( const MeshType& mesh,
                          const IndexType& index,
                          const CoordinatesType& coordinates ) const
{
   return nonlinearDiffusionOperator.getLinearSystemRowLength( mesh, index, coordinates );
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index,
          typename NonlinearDiffusionOperator >
template< typename Vector, typename MatrixRow >
#ifdef HAVE_CUDA
__device__ __host__
#endif
void
tnlNonlinearDiffusion< tnlGrid< 2, MeshReal, Device, MeshIndex >, NonlinearDiffusionOperator, Real, Index >::
updateLinearSystem( const RealType& time,
                    const RealType& tau,
                    const MeshType& mesh,
                    const IndexType& index,
                    const CoordinatesType& coordinates,
                    Vector& u,
                    Vector& b,
                    MatrixRow& matrixRow ) const
{
    nonlinearDiffusionOperator.updateLinearSystem( time, tau, mesh, index, coordinates, u, b, matrixRow );
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
template< typename Vector >
#ifdef HAVE_CUDA
__device__ __host__
#endif
Real
tnlNonlinearDiffusion< tnlGrid< 3, MeshReal, Device, MeshIndex >, NonlinearDiffusionOperator, Real, Index >::
getValue( const MeshType& mesh,
          const IndexType cellIndex,
          const CoordinatesType& coordinates,
          const Vector& u,
          const Real& time ) const
{
    return nonlinearDiffusionOperator.getValue( mesh, cellIndex, coordinates, u, time );
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index,
          typename NonlinearDiffusionOperator >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
Index
tnlNonlinearDiffusion< tnlGrid< 3, MeshReal, Device, MeshIndex >, NonlinearDiffusionOperator, Real, Index >::
getLinearSystemRowLength( const MeshType& mesh,
                          const IndexType& index,
                          const CoordinatesType& coordinates ) const
{
   return nonlinearDiffusionOperator.getLinearSystemRowLength( mesh, index, coordinates );
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index,
          typename NonlinearDiffusionOperator >
template< typename Vector, typename MatrixRow >
#ifdef HAVE_CUDA
__device__ __host__
#endif
void
tnlNonlinearDiffusion< tnlGrid< 3, MeshReal, Device, MeshIndex >, NonlinearDiffusionOperator, Real, Index >::
updateLinearSystem( const RealType& time,
                    const RealType& tau,
                    const MeshType& mesh,
                    const IndexType& index,
                    const CoordinatesType& coordinates,
                    Vector& u,
                    Vector& b,
                    MatrixRow& matrixRow ) const
{
    nonlinearDiffusionOperator.updateLinearSystem( time, tau, mesh, index, coordinates, u, b, matrixRow );
}

#endif	/* TNLNONLINEARDIFFUSION_IMPL_H */
