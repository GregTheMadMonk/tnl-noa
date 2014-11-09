
#ifndef TNLLINEARDIFFUSION_IMP_H
#define	TNLLINEARDIFFUSION_IMP_H

#include <operators/diffusion/tnlLinearDiffusion.h>
#include <mesh/tnlGrid.h>

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
tnlString
tnlLinearDiffusion< tnlGrid< 1, MeshReal, Device, MeshIndex >, Real, Index >::
getType()
{
   return tnlString( "tnlLinearDiffusion< " ) +
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
tnlLinearDiffusion< tnlGrid< 1, MeshReal, Device, MeshIndex >, Real, Index >::
getValue( const MeshType& mesh,
          const IndexType cellIndex,
          const CoordinatesType& coordinates,
          const Vector& u,
          const Real& time ) const
{
   return ( u[ mesh.getCellXPredecessor( cellIndex ) ]
            - 2.0 * u[ cellIndex ]
            + u[ mesh.getCellXSuccessor( cellIndex ) ] ) * mesh.getHxSquareInverse();
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
Index
tnlLinearDiffusion< tnlGrid< 1, MeshReal, Device, MeshIndex >, Real, Index >::
getLinearSystemRowLength( const MeshType& mesh,
                          const IndexType& index,
                          const CoordinatesType& coordinates ) const
{
   return 3;
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
void
tnlLinearDiffusion< tnlGrid< 1, MeshReal, Device, MeshIndex >, Real, Index >::
updateLinearSystem( const RealType& time,
                    const RealType& tau,
                    const MeshType& mesh,
                    const IndexType& index,
                    const CoordinatesType& coordinates,
                    Vector& u,
                    Vector& b,
                    IndexType* columns,
                    RealType* values,
                    IndexType& rowLength ) const
{
   const RealType lambdaX = tau * mesh.getHxSquareInverse();
   columns[ 0 ] = mesh.getCellXPredecessor( index );
   columns[ 1 ] = index;
   columns[ 2 ] = mesh.getCellXSuccessor( index );
   values[ 0 ] = -lambdaX;
   values[ 1 ] = 2.0 * lambdaX;
   values[ 2 ] = -lambdaX;
   rowLength = 3;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
tnlString
tnlLinearDiffusion< tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index >::
getType()
{
   return tnlString( "tnlLinearDiffusion< " ) +
          MeshType::getType() + ", " +
          ::getType< Real >() + ", " +
          ::getType< Index >() + " >";
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
Index
tnlLinearDiffusion< tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index >::
getLinearSystemRowLength( const MeshType& mesh,
                          const IndexType& index,
                          const CoordinatesType& coordinates ) const
{
   return 5;
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
tnlLinearDiffusion< tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index >::
getValue( const MeshType& mesh,
          const IndexType cellIndex,
          const CoordinatesType& coordinates,
          const Vector& u,
          const Real& time ) const
{
   return ( u[ mesh.getCellXPredecessor( cellIndex ) ]
            - 2.0 * u[ cellIndex ]
            + u[ mesh.getCellXSuccessor( cellIndex ) ] ) * mesh.getHxSquareInverse() +
           ( u[ mesh.getCellYPredecessor( cellIndex ) ]
             - 2.0 * u[ cellIndex ]
             + u[ mesh.getCellYSuccessor( cellIndex ) ] ) * mesh.getHySquareInverse();
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
void
tnlLinearDiffusion< tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index >::
updateLinearSystem( const RealType& time,
                    const RealType& tau,
                    const MeshType& mesh,
                    const IndexType& index,
                    const CoordinatesType& coordinates,
                    Vector& u,
                    Vector& b,
                    IndexType* columns,
                    RealType* values,
                    IndexType& rowLength ) const
{
   const RealType lambdaX = tau * mesh.getHxSquareInverse();
   const RealType lambdaY = tau * mesh.getHySquareInverse();
   columns[ 0 ] = mesh.getCellYPredecessor( index );
   columns[ 1 ] = mesh.getCellXPredecessor( index );
   columns[ 2 ] = index;
   columns[ 3 ] = mesh.getCellXSuccessor( index );
   columns[ 4 ] = mesh.getCellYSuccessor( index );
   values[ 0 ] = -lambdaY;
   values[ 1 ] = -lambdaX;
   values[ 2 ] = 2.0 * ( lambdaX + lambdaY );
   values[ 3 ] = -lambdaX;
   values[ 4 ] = -lambdaY;
   rowLength = 5;
}


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
tnlString
tnlLinearDiffusion< tnlGrid< 3, MeshReal, Device, MeshIndex >, Real, Index >::
getType()
{
   return tnlString( "tnlLinearDiffusion< " ) +
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
tnlLinearDiffusion< tnlGrid< 3, MeshReal, Device, MeshIndex >, Real, Index >::
getValue( const MeshType& mesh,
          const IndexType cellIndex,
          const CoordinatesType& coordinates,
          const Vector& u,
          const Real& time ) const
{
   return ( u[ mesh.getCellXPredecessor( cellIndex ) ]
            - 2.0 * u[ cellIndex ]
            + u[ mesh.getCellXSuccessor( cellIndex ) ] ) * mesh.getHxSquareInverse() +
          ( u[ mesh.getCellYPredecessor( cellIndex ) ]
            - 2.0 * u[ cellIndex ]
            + u[ mesh.getCellYSuccessor( cellIndex ) ] ) * mesh.getHySquareInverse() +
          ( u[ mesh.getCellZPredecessor( cellIndex ) ]
            - 2.0 * u[ cellIndex ]
            + u[ mesh.getCellZSuccessor( cellIndex ) ] ) * mesh.getHzSquareInverse();
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
Index
tnlLinearDiffusion< tnlGrid< 3, MeshReal, Device, MeshIndex >, Real, Index >::
getLinearSystemRowLength( const MeshType& mesh,
                          const IndexType& index,
                          const CoordinatesType& coordinates ) const
{
   return 7;
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
void
tnlLinearDiffusion< tnlGrid< 3, MeshReal, Device, MeshIndex >, Real, Index >::
updateLinearSystem( const RealType& time,
                    const RealType& tau,
                    const MeshType& mesh,
                    const IndexType& index,
                    const CoordinatesType& coordinates,
                    Vector& u,
                    Vector& b,
                    IndexType* columns,
                    RealType* values,
                    IndexType& rowLength ) const
{
   const RealType lambdaX = tau * mesh.getHxSquareInverse();
   const RealType lambdaY = tau * mesh.getHySquareInverse();
   const RealType lambdaZ = tau * mesh.getHzSquareInverse();
   columns[ 0 ] = mesh.getCellZPredecessor( index );
   columns[ 1 ] = mesh.getCellYPredecessor( index );
   columns[ 2 ] = mesh.getCellXPredecessor( index );
   columns[ 3 ] = index;
   columns[ 4 ] = mesh.getCellXSuccessor( index );
   columns[ 5 ] = mesh.getCellYSuccessor( index );
   columns[ 6 ] = mesh.getCellZSuccessor( index );
   values[ 0 ] = -lambdaZ;
   values[ 1 ] = -lambdaY;
   values[ 2 ] = -lambdaX;
   values[ 3 ] = 2.0 * ( lambdaX + lambdaY + lambdaZ );
   values[ 4 ] = -lambdaX;
   values[ 5 ] = -lambdaY;
   values[ 6 ] = -lambdaZ;
   rowLength = 7;
}



#endif	/* TNLLINEARDIFFUSION_IMP_H */
