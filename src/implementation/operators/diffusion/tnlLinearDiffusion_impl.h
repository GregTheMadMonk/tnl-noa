
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
   template< typename Vector, typename MatrixRow >
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
                    MatrixRow& matrixRow ) const
{
   const RealType lambdaX = tau * mesh.getHxSquareInverse();
   //printf( "tau = %f lambda = %f dx_sqr = %f dx = %f, \n", tau, lambdaX, mesh.getHxSquareInverse(), mesh.getHx() );
   matrixRow.setElement( 0, mesh.getCellXPredecessor( index ),     - lambdaX );
   matrixRow.setElement( 1, index,                             2.0 * lambdaX );
   matrixRow.setElement( 2, mesh.getCellXSuccessor( index ),       - lambdaX );
   //printf( "Linear diffusion index %d columns %d %d %d \n", index, columns[ 0 ], columns[ 1 ], columns[ 2 ] );
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
   template< typename Vector, typename MatrixRow >
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
                    MatrixRow& matrixRow ) const
{
   const RealType lambdaX = tau * mesh.getHxSquareInverse();
   const RealType lambdaY = tau * mesh.getHySquareInverse();
   matrixRow.setElement( 0, mesh.getCellYPredecessor( index ), -lambdaY );
   matrixRow.setElement( 1, mesh.getCellXPredecessor( index ), -lambdaX );
   matrixRow.setElement( 2, index,                             2.0 * ( lambdaX + lambdaY ) );
   matrixRow.setElement( 3, mesh.getCellXSuccessor( index ),   -lambdaX );
   matrixRow.setElement( 4, mesh.getCellYSuccessor( index ),   -lambdaY );
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
   template< typename Vector, typename MatrixRow >
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
                    MatrixRow& matrixRow ) const
{
   const RealType lambdaX = tau * mesh.getHxSquareInverse();
   const RealType lambdaY = tau * mesh.getHySquareInverse();
   const RealType lambdaZ = tau * mesh.getHzSquareInverse();
   matrixRow.setElement( 0, mesh.getCellZPredecessor( index ), -lambdaZ );
   matrixRow.setElement( 1, mesh.getCellYPredecessor( index ), -lambdaY );
   matrixRow.setElement( 2, mesh.getCellXPredecessor( index ), -lambdaX );
   matrixRow.setElement( 3, index,                             2.0 * ( lambdaX + lambdaY + lambdaZ ) );
   matrixRow.setElement( 4, mesh.getCellXSuccessor( index ),   -lambdaX );
   matrixRow.setElement( 5, mesh.getCellYSuccessor( index ),   -lambdaY );
   matrixRow.setElement( 6, mesh.getCellZSuccessor( index ),   -lambdaZ );
}



#endif	/* TNLLINEARDIFFUSION_IMP_H */
