#ifndef TNLDIRICHLETBOUNDARYCONDITIONS_IMPL_H
#define	TNLDIRICHLETBOUNDARYCONDITIONS_IMPL_H


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Function,
          typename Real,
          typename Index >
bool
tnlDirichletBoundaryConditions< tnlGrid< 1,MeshReal, Device, MeshIndex >, Function, Real, Index >::
init( const tnlParameterContainer& parameters )
{
   return function.init( parameters, "boundary-conditions-" );
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Function,
          typename Real,
          typename Index >
void
tnlDirichletBoundaryConditions< tnlGrid< 1,MeshReal, Device, MeshIndex >, Function, Real, Index >::
setBoundaryConditions( const RealType& time,
                       const RealType& tau,
                       const MeshType& mesh,
                       const IndexType index,
                       const CoordinatesType& coordinates,
                       DofVectorType& u,
                       DofVectorType& fu )
{
   fu[ index ] = 0;
   u[ index ] = function.getValue( mesh.getVertex( coordinates ), time );
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Function,
          typename Real,
          typename Index >
bool
tnlDirichletBoundaryConditions< tnlGrid< 2,MeshReal, Device, MeshIndex >, Function, Real, Index >::
init( const tnlParameterContainer& parameters )
{
   return function.init( parameters, "boundary-conditions-" );
}


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Function,
          typename Real,
          typename Index >
void
tnlDirichletBoundaryConditions< tnlGrid< 2, MeshReal, Device, MeshIndex >, Function, Real, Index >::
setBoundaryConditions( const RealType& time,
                       const RealType& tau,
                       const MeshType& mesh,
                       const IndexType index,
                       const CoordinatesType& coordinates,
                       DofVectorType& u,
                       DofVectorType& fu )
{
   fu[ index ] = 0;
   u[ index ] = function.getValue( mesh.getVertex( coordinates ), time );;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Function,
          typename Real,
          typename Index >
bool
tnlDirichletBoundaryConditions< tnlGrid< 3,MeshReal, Device, MeshIndex >, Function, Real, Index >::
init( const tnlParameterContainer& parameters )
{
   return function.init( parameters, "boundary-conditions-" );
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Function,
          typename Real,
          typename Index >
void
tnlDirichletBoundaryConditions< tnlGrid< 3, MeshReal, Device, MeshIndex >, Function, Real, Index >::
setBoundaryConditions( const RealType& time,
                       const RealType& tau,
                       const MeshType& mesh,
                       const IndexType index,
                       const CoordinatesType& coordinates,
                       DofVectorType& u,
                       DofVectorType& fu )
{
   fu[ index ] = 0;
   u[ index ] = function.getValue( mesh.getVertex( coordinates ), time );;
}


#endif	/* TNLDIRICHLETBOUNDARYCONDITIONS_IMPL_H */

