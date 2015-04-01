
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
   return ( u[ mesh.template getCellNextToCell< - 1 >( cellIndex ) ]
            - 2.0 * u[ cellIndex ]
            + u[ mesh.template getCellNextToCell< 1 >( cellIndex ) ] ) * mesh.getHxSquareInverse();
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
   matrixRow.setElement( 0, mesh.template getCellNextToCell< -1 >( index ),     - lambdaX );
   matrixRow.setElement( 1, index,                             2.0 * lambdaX );
   matrixRow.setElement( 2, mesh.template getCellNextToCell< 1 >( index ),       - lambdaX );
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
   return ( u[ mesh.template getCellNextToCell< -1, 0 >( cellIndex ) ]
            - 2.0 * u[ cellIndex ]
            + u[ mesh.template getCellNextToCell< 1, 0 >( cellIndex ) ] ) * mesh.getHxSquareInverse() +
           ( u[ mesh.template getCellNextToCell< 0, -1 >( cellIndex ) ]
             - 2.0 * u[ cellIndex ]
             + u[ mesh.template getCellNextToCell< 0, 1 >( cellIndex ) ] ) * mesh.getHySquareInverse();
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
   matrixRow.setElement( 0, mesh.template getCellNextToCell< 0, -1 >( index ), -lambdaY );
   matrixRow.setElement( 1, mesh.template getCellNextToCell< -1, 0 >( index ), -lambdaX );
   matrixRow.setElement( 2, index,                                             2.0 * ( lambdaX + lambdaY ) );
   matrixRow.setElement( 3, mesh.template getCellNextToCell< 1, 0 >( index ),   -lambdaX );
   matrixRow.setElement( 4, mesh.template getCellNextToCell< 0, 1 >( index ),   -lambdaY );
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
   return ( u[ mesh.template getCellNextToCell< -1, 0, 0 >( cellIndex ) ]
            - 2.0 * u[ cellIndex ]
            + u[ mesh.template getCellNextToCell< 1, 0, 0 >( cellIndex ) ] ) * mesh.getHxSquareInverse() +
          ( u[ mesh.template getCellNextToCell< 0, -1, 0 >( cellIndex ) ]
            - 2.0 * u[ cellIndex ]
            + u[ mesh.template getCellNextToCell< 0, 1, 0 >( cellIndex ) ] ) * mesh.getHySquareInverse() +
          ( u[ mesh.template getCellNextToCell< 0, 0, -1 >( cellIndex ) ]
            - 2.0 * u[ cellIndex ]
            + u[ mesh.template getCellNextToCell< 0, 0, 1 >( cellIndex ) ] ) * mesh.getHzSquareInverse();
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
   matrixRow.setElement( 0, mesh.template getCellNextToCell< 0, 0, -1 >( index ), -lambdaZ );
   matrixRow.setElement( 1, mesh.template getCellNextToCell< 0, -1, 0 >( index ), -lambdaY );
   matrixRow.setElement( 2, mesh.template getCellNextToCell< -1, 0, 0 >( index ), -lambdaX );
   matrixRow.setElement( 3, index,                             2.0 * ( lambdaX + lambdaY + lambdaZ ) );
   matrixRow.setElement( 4, mesh.template getCellNextToCell< 1, 0, 0 >( index ),   -lambdaX );
   matrixRow.setElement( 5, mesh.template getCellNextToCell< 0, 1, 0 >( index ),   -lambdaY );
   matrixRow.setElement( 6, mesh.template getCellNextToCell< 0, 0, 1 >( index ),   -lambdaZ );
}



#endif	/* TNLLINEARDIFFUSION_IMP_H */
