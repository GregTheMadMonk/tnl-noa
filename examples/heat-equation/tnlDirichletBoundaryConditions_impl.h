#ifndef TNLDIRICHLETBOUNDARYCONDITIONS_IMPL_H
#define	TNLDIRICHLETBOUNDARYCONDITIONS_IMPL_H

#include "tnlDirichletBoundaryConditions.h"

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void
tnlDirichletBoundaryConditions< tnlGrid< 1,MeshReal, Device, MeshIndex >, Real, Index >::
setBoundaryConditions( const RealType& time,
                       const RealType& tau,
                       const MeshType& mesh,
                       const IndexType index,
                       const CoordinatesType& coordinates,
                       DofVectorType& u,
                       DofVectorType& fu )
{
   fu[ index ] = 0;
   u[ index ] = 0;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void
tnlDirichletBoundaryConditions< tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index >::
setBoundaryConditions( const RealType& time,
                       const RealType& tau,
                       const MeshType& mesh,
                       const IndexType index,
                       const CoordinatesType& coordinates,
                       DofVectorType& u,
                       DofVectorType& fu )
{
   fu[ index ] = 0;
   u[ index ] = 0;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void
tnlDirichletBoundaryConditions< tnlGrid< 3, MeshReal, Device, MeshIndex >, Real, Index >::
setBoundaryConditions( const RealType& time,
                       const RealType& tau,
                       const MeshType& mesh,
                       const IndexType index,
                       const CoordinatesType& coordinates,
                       DofVectorType& u,
                       DofVectorType& fu )
{
   fu[ index ] = 0;
   u[ index ] = 0;
}


#endif	/* TNLDIRICHLETBOUNDARYCONDITIONS_IMPL_H */

