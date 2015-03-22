
#ifndef TNLBACKWARDFINITEDIFFERENCE_IMPL_H
#define	TNLBACKWARDFINITEDIFFERENCE_IMPL_H

#include "tnlBackwardFiniteDifference.h"
#include <mesh/tnlGrid.h>

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
tnlString
tnlBackwardFiniteDifference< tnlGrid< 1, MeshReal, Device, MeshIndex >, Real, Index >::
getType()
{
   return tnlString( "tnlBackwardFiniteDifference< " ) +
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
tnlBackwardFiniteDifference< tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index >::
getType()
{
   return tnlString( "tnlBackwardFiniteDifference< " ) +
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
tnlBackwardFiniteDifference< tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index >::
getValueX( const MeshType& mesh,
          const IndexType cellIndex,
          const Vector& u) const
{
   return ( - u[ mesh.template getCellNextToCell< -1,0 >( cellIndex ) ]
            + u[ cellIndex ]) * mesh.getHxInverse();   
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
tnlBackwardFiniteDifference< tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index >::
getValueY( const MeshType& mesh,
          const IndexType cellIndex,
          const Vector& u) const
{
   return ( - u[ mesh.template getCellNextToCell< 0,-1 >( cellIndex ) ]
            + u[ cellIndex ]) * mesh.getHyInverse();   
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
tnlString
tnlBackwardFiniteDifference< tnlGrid< 3, MeshReal, Device, MeshIndex >, Real, Index >::
getType()
{
   return tnlString( "tnlBackwardFiniteDifference< " ) +
          MeshType::getType() + ", " +
          ::getType< Real >() + ", " +
          ::getType< Index >() + " >";
}

#endif	/* TNLBACKWARDFINITEDIFFERENCE_IMPL_H */
