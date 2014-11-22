/***************************************************************************
                          tnlAnalyticNeumannBoundaryConditions_impl.h  -  description
                             -------------------
    begin                : Nov 22, 2014
    copyright            : (C) 2014 by oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef TNLANALYTICNEUMANNBOUNDARYCONDITIONS_IMPL_H_
#define TNLANALYTICNEUMANNBOUNDARYCONDITIONS_IMPL_H_

/****
 * Base
 */
template< typename Function >
void
tnlAnalyticNeumannBoundaryConditionsBase< Function >::
configSetup( tnlConfigDescription& config,
             const tnlString& prefix )
{
   Function::configSetup( config, prefix );
}

template< typename Function >
bool
tnlAnalyticNeumannBoundaryConditionsBase< Function >::
setup( const tnlParameterContainer& parameters,
       const tnlString& prefix )
{
   return function.setup( parameters, prefix );
}

template< typename Function >
void
tnlAnalyticNeumannBoundaryConditionsBase< Function >::
setFunction( const Function& function )
{
   return function->function;
}

/****
 * 1D grid
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Function,
          typename Real,
          typename Index >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
void
tnlAnalyticNeumannBoundaryConditions< tnlGrid< 1, MeshReal, Device, MeshIndex >, Function, Real, Index >::
setBoundaryConditions( const RealType& time,
                       const MeshType& mesh,
                       const IndexType index,
                       const CoordinatesType& coordinates,
                       DofVectorType& u,
                       DofVectorType& fu ) const
{
   fu[ index ] = 0;
   const Real functionValue = this->function.getValue( mesh.template getCellCenter< VertexType >( coordinates ), time );
   if( coordinates.x() == 0 )
      u[ index ] = u[ mesh.getCellXSuccessor( index ) ] - mesh.getHx() * functionValue;
   else
      u[ index ] = u[ mesh.getCellXPredecessor( index ) ] + mesh.getHx() * functionValue;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Function,
          typename Real,
          typename Index >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
Index
tnlAnalyticNeumannBoundaryConditions< tnlGrid< 1, MeshReal, Device, MeshIndex >, Function, Real, Index >::
getLinearSystemRowLength( const MeshType& mesh,
                          const IndexType& index,
                          const CoordinatesType& coordinates ) const
{
   return 2;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Function,
          typename Real,
          typename Index >
#ifdef HAVE_CUDA
__device__ __host__
#endif
void
tnlAnalyticNeumannBoundaryConditions< tnlGrid< 1, MeshReal, Device, MeshIndex >, Function, Real, Index >::
updateLinearSystem( const RealType& time,
                    const MeshType& mesh,
                    const IndexType& index,
                    const CoordinatesType& coordinates,
                    DofVectorType& u,
                    DofVectorType& b,
                    IndexType* columns,
                    RealType* values,
                    IndexType& rowLength ) const
{
   const Real functionValue = this->function.getValue( mesh.template getCellCenter< VertexType >( coordinates ), time );
   if( coordinates.x() == 0 )
   {
      columns[ 0 ] = index;
      columns[ 1 ] = mesh.getCellXSuccessor( index );
      values[ 0 ] = 1.0;
      values[ 1 ] = -1.0;
      b[ index ] = - mesh.getHx() * functionValue;
   }
   else
   {
      columns[ 0 ] = mesh.getCellXPredecessor( index );
      columns[ 1 ] = index;
      values[ 0 ] = -1.0;
      values[ 1 ] = 1.0;
      b[ index ] = mesh.getHx() * functionValue;
   }
   rowLength = 2;
}

/****
 * 2D grid
 */

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Function,
          typename Real,
          typename Index >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
void
tnlAnalyticNeumannBoundaryConditions< tnlGrid< 2, MeshReal, Device, MeshIndex >, Function, Real, Index >::
setBoundaryConditions( const RealType& time,
                       const MeshType& mesh,
                       const IndexType index,
                       const CoordinatesType& coordinates,
                       DofVectorType& u,
                       DofVectorType& fu ) const
{
   fu[ index ] = 0;
   const Real functionValue = this->function.getValue( mesh.template getCellCenter< VertexType >( coordinates ), time );
   if( coordinates.x() == 0 )
   {
      u[ index ] = u[ mesh.getCellXSuccessor( index ) ] - mesh.getHx() * functionValue;
      return;
   }
   if( coordinates.x() == mesh.getDimensions().x() - 1 )
   {
      u[ index ] = u[ mesh.getCellXPredecessor( index ) ] + mesh.getHx() * functionValue;
      return;
   }
   if( coordinates.y() == 0 )
   {
      u[ index ] = u[ mesh.getCellYSuccessor( index ) ] - mesh.getHy() * functionValue;
      return;
   }
   if( coordinates.y() == mesh.getDimensions().y() - 1 )
   {
      u[ index ] = u[ mesh.getCellYPredecessor( index ) ] + mesh.getHy() * functionValue;
      return;
   }
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Function,
          typename Real,
          typename Index >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
Index
tnlAnalyticNeumannBoundaryConditions< tnlGrid< 2, MeshReal, Device, MeshIndex >, Function, Real, Index >::
getLinearSystemRowLength( const MeshType& mesh,
                          const IndexType& index,
                          const CoordinatesType& coordinates ) const
{
   return 2;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Function,
          typename Real,
          typename Index >
#ifdef HAVE_CUDA
__device__ __host__
#endif
void
tnlAnalyticNeumannBoundaryConditions< tnlGrid< 2, MeshReal, Device, MeshIndex >, Function, Real, Index >::
updateLinearSystem( const RealType& time,
                    const MeshType& mesh,
                    const IndexType& index,
                    const CoordinatesType& coordinates,
                    DofVectorType& u,
                    DofVectorType& b,
                    IndexType* columns,
                    RealType* values,
                    IndexType& rowLength ) const
{
   const Real functionValue = this->function.getValue( mesh.template getCellCenter< VertexType >( coordinates ), time );
   if( coordinates.x() == 0 )
   {
      columns[ 0 ] = index;
      columns[ 1 ] = mesh.getCellXSuccessor( index );
      values[ 0 ] = 1.0;
      values[ 1 ] = -1.0;
      b[ index ] = - mesh.getHx() * functionValue;
   }
   if( coordinates.x() == mesh.getDimensions().x() - 1 )
   {
      columns[ 0 ] = mesh.getCellXPredecessor( index );
      columns[ 1 ] = index;
      values[ 0 ] = -1.0;
      values[ 1 ] = 1.0;
      b[ index ] = mesh.getHx() * functionValue;
   }
   if( coordinates.y() == 0 )
   {
      columns[ 0 ] = index;
      columns[ 1 ] = mesh.getCellYSuccessor( index );
      values[ 0 ] = 1.0;
      values[ 1 ] = -1.0;
      b[ index ] = - mesh.getHy() * functionValue;
   }
   if( coordinates.y() == mesh.getDimensions().y() - 1 )
   {
      columns[ 0 ] = mesh.getCellYPredecessor( index );
      columns[ 1 ] = index;
      values[ 0 ] = -1.0;
      values[ 1 ] = 1.0;
      b[ index ] = mesh.getHy() * functionValue;
   }
   rowLength = 2;
}

/****
 * 3D grid
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Function,
          typename Real,
          typename Index >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
void
tnlAnalyticNeumannBoundaryConditions< tnlGrid< 3, MeshReal, Device, MeshIndex >, Function, Real, Index >::
setBoundaryConditions( const RealType& time,
                       const MeshType& mesh,
                       const IndexType index,
                       const CoordinatesType& coordinates,
                       DofVectorType& u,
                       DofVectorType& fu ) const
{
   fu[ index ] = 0;
   fu[ index ] = 0;
   const Real functionValue = this->function.getValue( mesh.template getCellCenter< VertexType >( coordinates ), time );
   if( coordinates.x() == 0 )
   {
      u[ index ] = u[ mesh.getCellXSuccessor( index ) ] - mesh.getHx() * functionValue;
      return;
   }
   if( coordinates.x() == mesh.getDimensions().x() - 1 )
   {
      u[ index ] = u[ mesh.getCellXPredecessor( index ) ] + mesh.getHx() * functionValue;
      return;
   }
   if( coordinates.y() == 0 )
   {
      u[ index ] = u[ mesh.getCellYSuccessor( index ) ] - mesh.getHy() * functionValue;
      return;
   }
   if( coordinates.y() == mesh.getDimensions().y() - 1 )
   {
      u[ index ] = u[ mesh.getCellYPredecessor( index ) ] + mesh.getHy() * functionValue;
      return;
   }
   if( coordinates.z() == 0 )
   {
      u[ index ] = u[ mesh.getCellZSuccessor( index ) ] - mesh.getHz() * functionValue;
      return;
   }
   if( coordinates.z() == mesh.getDimensions().z() - 1 )
   {
      u[ index ] = u[ mesh.getCellZPredecessor( index ) ] + mesh.getHz() * functionValue;
      return;
   }
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Function,
          typename Real,
          typename Index >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
Index
tnlAnalyticNeumannBoundaryConditions< tnlGrid< 3, MeshReal, Device, MeshIndex >, Function, Real, Index >::
getLinearSystemRowLength( const MeshType& mesh,
                          const IndexType& index,
                          const CoordinatesType& coordinates ) const
{
   return 2;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Function,
          typename Real,
          typename Index >
#ifdef HAVE_CUDA
__device__ __host__
#endif
void
tnlAnalyticNeumannBoundaryConditions< tnlGrid< 3, MeshReal, Device, MeshIndex >, Function, Real, Index >::
updateLinearSystem( const RealType& time,
                    const MeshType& mesh,
                    const IndexType& index,
                    const CoordinatesType& coordinates,
                    DofVectorType& u,
                    DofVectorType& b,
                    IndexType* columns,
                    RealType* values,
                    IndexType& rowLength ) const
{
   const Real functionValue = this->function.getValue( mesh.template getCellCenter< VertexType >( coordinates ), time );
   if( coordinates.x() == 0 )
   {
      columns[ 0 ] = index;
      columns[ 1 ] = mesh.getCellXSuccessor( index );
      values[ 0 ] = 1.0;
      values[ 1 ] = -1.0;
      b[ index ] = - mesh.getHx() * functionValue;
   }
   if( coordinates.x() == mesh.getDimensions().x() - 1 )
   {
      columns[ 0 ] = mesh.getCellXPredecessor( index );
      columns[ 1 ] = index;
      values[ 0 ] = -1.0;
      values[ 1 ] = 1.0;
      b[ index ] = mesh.getHx() * functionValue;
   }
   if( coordinates.y() == 0 )
   {
      columns[ 0 ] = index;
      columns[ 1 ] = mesh.getCellYSuccessor( index );
      values[ 0 ] = 1.0;
      values[ 1 ] = -1.0;
      b[ index ] = - mesh.getHy() * functionValue;
   }
   if( coordinates.y() == mesh.getDimensions().y() - 1 )
   {
      columns[ 0 ] = mesh.getCellYPredecessor( index );
      columns[ 1 ] = index;
      values[ 0 ] = -1.0;
      values[ 1 ] = 1.0;
      b[ index ] = mesh.getHy() * functionValue;
   }
   if( coordinates.z() == 0 )
   {
      columns[ 0 ] = index;
      columns[ 1 ] = mesh.getCellZSuccessor( index );
      values[ 0 ] = 1.0;
      values[ 1 ] = -1.0;
      b[ index ] = - mesh.getHz() * functionValue;
   }
   if( coordinates.z() == mesh.getDimensions().z() - 1 )
   {
      columns[ 0 ] = mesh.getCellZPredecessor( index );
      columns[ 1 ] = index;
      values[ 0 ] = -1.0;
      values[ 1 ] = 1.0;
      b[ index ] = mesh.getHz() * functionValue;
   }
   rowLength = 2;
}


#endif /* TNLANALYTICNEUMANNBOUNDARYCONDITIONS_IMPL_H_ */
