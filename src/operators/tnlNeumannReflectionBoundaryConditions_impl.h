#pragma once

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
bool
tnlNeumannReflectionBoundaryConditions< tnlGrid< 1, MeshReal, Device, MeshIndex >, Real, Index >::
setup( const Config::ParameterContainer& parameters,
      const String& prefix )
{
   return true;
}



template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
   template< typename EntityType,
             typename MeshFunction >
__cuda_callable__
const Real
tnlNeumannBoundaryConditions< tnlGrid< 1, MeshReal, Device, MeshIndex >, Function, Real, Index >::
operator()( const MeshFunction& u,
            const EntityType& entity,
            const RealType& time ) const

#ifdef HAVE_CUDA
   __device__ __host__
#endif
void tnlNeumannReflectionBoundaryConditions< tnlGrid< 1,MeshReal, Device, MeshIndex >, Real, Index >::
setBoundaryConditions( const RealType& time,
                       const MeshType& mesh,
                       const IndexType index,
                       const CoordinatesType& coordinates,
                       DofVectorType& u,
                       DofVectorType& fu )
{
	tmp = coordinates;

   if(coordinates.x() == 0)
	   tmp.x() = 1;
   else if(coordinates.x() == mesh. getDimensions().x() - 1)
	   tmp.x() = coordinates.x() - 2;

   u[ index ] = u[mesh.getCellIndex( tmp )];
}



template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
bool
tnlNeumannReflectionBoundaryConditions< tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index >::
setup( const Config::ParameterContainer& parameters,
      const String& prefix )
{
   return true;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
   template< typename EntityType,
             typename MeshFunction >
__cuda_callable__
const Real
tnlNeumannBoundaryConditions< tnlGrid< 1, MeshReal, Device, MeshIndex >, Function, Real, Index >::
operator()( const MeshFunction& u,
            const EntityType& entity,
            const RealType& time ) const

#ifdef HAVE_CUDA
   __device__ __host__
#endif
void tnlNeumannReflectionBoundaryConditions< tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index >::
setBoundaryConditions( const RealType& time,
                       const MeshType& mesh,
                       const IndexType index,
                       const CoordinatesType& coordinates,
                       DofVectorType& u,
                       DofVectorType& fu )
{

	tmp = coordinates;

   if(coordinates.x() == 0)
	   tmp.x() = coordinates.x() + 2;
   else if(coordinates.x() == mesh. getDimensions().x() - 1)
	   tmp.x() = coordinates.x() - 2;

   if(coordinates.y() == 0)
	   tmp.y() = coordinates.y() + 2;
   else if(coordinates.y() == mesh. getDimensions().y() - 1)
	   tmp.y() = coordinates.y() - 2;

   u[ index ] = u[mesh.getCellIndex( tmp )];
}


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
bool
tnlNeumannReflectionBoundaryConditions< tnlGrid< 3, MeshReal, Device, MeshIndex >, Real, Index >::
setup( const Config::ParameterContainer& parameters,
      const String& prefix )
{
   return true;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
   template< typename EntityType,
             typename MeshFunction >
__cuda_callable__
const Real
tnlNeumannBoundaryConditions< tnlGrid< 1, MeshReal, Device, MeshIndex >, Function, Real, Index >::
operator()( const MeshFunction& u,
            const EntityType& entity,
            const RealType& time ) const

#ifdef HAVE_CUDA
   __device__ __host__
#endif
void tnlNeumannReflectionBoundaryConditions< tnlGrid< 3, MeshReal, Device, MeshIndex >, Real, Index >::
setBoundaryConditions( const RealType& time,
                       const MeshType& mesh,
                       const IndexType index,
                       const CoordinatesType& coordinates,
                       DofVectorType& u,
                       DofVectorType& fu )
{
	tmp = coordinates;

   if(coordinates.x() == 0)
	   tmp.x() = coordinates.x() + 2;
   else if(coordinates.x() == mesh. getDimensions().x() - 1)
	   tmp.x() = coordinates.x() - 2;

   if(coordinates.y() == 0)
	   tmp.y() = coordinates.y() + 2;
   else if(coordinates.y() == mesh. getDimensions().y() - 1)
	   tmp.y() = coordinates.y() - 2;

   if(coordinates.z() == 0)
	   tmp.z() = coordinates.z() + 2;
   else if(coordinates.z() == mesh. getDimensions().z() - 1)
	   tmp.z() = coordinates.z() - 2;

   u[ index ] = u[mesh.getCellIndex( tmp )];
}



#endif	/* TNLNEUMANNREFLECTIONBOUNDARYCONDITIONS_IMPL_H */

