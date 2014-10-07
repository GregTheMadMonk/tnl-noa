
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

/*template< typename MeshReal,
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
explicitUpdate( const RealType& time,
                const RealType& tau,
                const MeshType& mesh,
                const IndexType cellIndex,
                const CoordinatesType& coordinates,
                Vector& u,
                Vector& fu ) const
{
   fu[ cellIndex ] = ( u[ mesh.getCellXPredecessor( cellIndex ) ]
                       - 2.0 * u[ cellIndex ]
                       + u[ mesh.getCellXSuccessor( cellIndex ) ] ) * mesh.getHxSquareInverse();
}*/

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
tnlString
tnlLinearDiffusion< tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index >::
getType()
{
   return tnlString( "tnlLinearDiffusion< " ) +
          MeshType::getType() + ", " +
          ::getType< Real >() + ", " +
          ::getType< Index >() + " >";
}

/*template< typename MeshReal,
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
explicitUpdate( const RealType& time,
                const RealType& tau,
                const MeshType& mesh,
                const IndexType cellIndex,
                const CoordinatesType& coordinates,
                Vector& u,
                Vector& fu ) const
{
   fu[ cellIndex ] = ( u[ mesh.getCellXPredecessor( cellIndex ) ]
                       - 2.0 * u[ cellIndex ]
                       + u[ mesh.getCellXSuccessor( cellIndex ) ] ) * mesh.getHxSquareInverse() +
                     ( u[ mesh.getCellYPredecessor( cellIndex ) ]
                       - 2.0 * u[ cellIndex ]
                       + u[ mesh.getCellYSuccessor( cellIndex ) ] ) * mesh.getHySquareInverse();
}*/

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
tnlString
tnlLinearDiffusion< tnlGrid< 3, MeshReal, Device, MeshIndex >, Real, Index >::
getType()
{
   return tnlString( "tnlLinearDiffusion< " ) +
          MeshType::getType() + ", " +
          ::getType< Real >() + ", " +
          ::getType< Index >() + " >";
}

/*template< typename MeshReal,
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
explicitUpdate( const RealType& time,
                const RealType& tau,
                const MeshType& mesh,
                const IndexType cellIndex,
                const CoordinatesType& coordinates,
                Vector& u,
                Vector& fu ) const
{
   fu[ cellIndex ] = ( u[ mesh.getCellXPredecessor( cellIndex ) ]
                       - 2.0 * u[ cellIndex ]
                       + u[ mesh.getCellXSuccessor( cellIndex ) ] ) * mesh.getHxSquareInverse() +
                     ( u[ mesh.getCellYPredecessor( cellIndex ) ]
                       - 2.0 * u[ cellIndex ]
                       + u[ mesh.getCellYSuccessor( cellIndex ) ] ) * mesh.getHySquareInverse() +
                     ( u[ mesh.getCellZPredecessor( cellIndex ) ]
                       - 2.0 * u[ cellIndex ]
                       + u[ mesh.getCellZSuccessor( cellIndex ) ] ) * mesh.getHzSquareInverse();
}*/

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


#endif	/* TNLLINEARDIFFUSION_IMP_H */
