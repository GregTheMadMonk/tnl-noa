
#ifndef TTNLOPERATORQ_IMPL_H
#define	TTNLOPERATORQ_IMPL_H

#include "tnlOperatorQ.h"
#include <mesh/tnlGrid.h>

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
tnlString
tnlOperatorQ< tnlGrid< 1, MeshReal, Device, MeshIndex >, Real, Index >::
getType()
{
   return tnlString( "tnlOperatorQ< " ) +
          MeshType::getType() + ", " +
          ::getType< Real >() + ", " +
          ::getType< Index >() + " >";
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
tnlString
tnlOperatorQ< tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index >::
getType()
{
   return tnlString( "tnlOperatorQ< " ) +
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
tnlOperatorQ< tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index >::
getValue( const MeshType& mesh,
          const IndexType cellIndex,
          const Vector& u) const
{
   return sqrt( 1.0 + (fDifference.getValueX(mesh,cellIndex,u))*(fDifference.getValueX(mesh,cellIndex,u))
          + (fDifference.getValueY(mesh,cellIndex,u))*(fDifference.getValueY(mesh,cellIndex,u)) );
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
tnlOperatorQ< tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index >::
getValueStriped( const MeshType& mesh,
          const IndexType cellIndex,
          const Vector& u) const
{
   return sqrt( 1.0 + 0.5*( (fDifference.getValueX(mesh,cellIndex,u))*(fDifference.getValueX(mesh,cellIndex,u))
          + (fDifference.getValueY(mesh,cellIndex,u))*(fDifference.getValueY(mesh,cellIndex,u)) 
          + (bDifference.getValueX(mesh,cellIndex,u))*(bDifference.getValueX(mesh,cellIndex,u))
          + (bDifference.getValueY(mesh,cellIndex,u))*(bDifference.getValueY(mesh,cellIndex,u)) ) );
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
tnlString
tnlOperatorQ< tnlGrid< 3, MeshReal, Device, MeshIndex >, Real, Index >::
getType()
{
   return tnlString( "tnlOperatorQ< " ) +
          MeshType::getType() + ", " +
          ::getType< Real >() + ", " +
          ::getType< Index >() + " >";
}

#endif	/* TTNLOPERATORQ_IMPL_H */
