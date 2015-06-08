
#ifndef TNLFINITEVOLUMEOPERATORQ_IMPL_H
#define	TNLFINITEVOLUMEOPERATORQ_IMPL_H

#include <operators/operator-Q/tnlFiniteVolumeOperatorQ.h>
#include <mesh/tnlGrid.h>

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
tnlString
tnlFiniteVolumeOperatorQ< tnlGrid< 1, MeshReal, Device, MeshIndex >, Real, Index, 0 >::
getType()
{
   return tnlString( "tnlFiniteVolumeOperatorQ< " ) +
          MeshType::getType() + ", " +
          ::getType< Real >() + ", " +
          ::getType< Index >() + ", 0 >";
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
tnlString
tnlFiniteVolumeOperatorQ< tnlGrid< 1, MeshReal, Device, MeshIndex >, Real, Index, 1 >::
getType()
{
   return tnlString( "tnlFiniteVolumeOperatorQ< " ) +
          MeshType::getType() + ", " +
          ::getType< Real >() + ", " +
          ::getType< Index >() + ", 1 >";
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
bool tnlFiniteVolumeOperatorQ< tnlGrid< 1, MeshReal, Device, MeshIndex >, Real, Index, 0 >::setEps( const Real& eps )
{
  this->eps = eps;
  
  return true;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
bool tnlFiniteVolumeOperatorQ< tnlGrid< 1, MeshReal, Device, MeshIndex >, Real, Index, 1 >::setEps( const Real& eps )
{
  this->eps = eps;
  
  return true;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >   
#ifdef HAVE_CUDA
   __device__ __host__
#endif
void  
tnlFiniteVolumeOperatorQ< tnlGrid< 1, MeshReal, Device, MeshIndex >, Real, Index, 1 >::
update( const MeshType& mesh, const RealType& time )
{
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
template< typename Vector, int AxeX, int AxeY, int AxeZ >
#ifdef HAVE_CUDA
__device__ __host__
#endif
Real
tnlFiniteVolumeOperatorQ< tnlGrid< 1, MeshReal, Device, MeshIndex >, Real, Index, 0 >::
boundaryDerivative( const MeshType& mesh,
          const IndexType cellIndex,
          const CoordinatesType& coordinates,
          const Vector& u,
          const Real& time,
          const IndexType& dx, 
          const IndexType& dy,
          const IndexType& dz ) const
{
    return 0.0;
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
tnlFiniteVolumeOperatorQ< tnlGrid< 1, MeshReal, Device, MeshIndex >, Real, Index, 0 >::
getValue( const MeshType& mesh,
          const IndexType cellIndex,
          const CoordinatesType& coordinates,
          const Vector& u,
          const Real& time,
          const IndexType& dx, 
          const IndexType& dy,
          const IndexType& dz ) const
{
    return 0.0;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
template< typename Vector, int AxeX, int AxeY, int AxeZ >
#ifdef HAVE_CUDA
__device__ __host__
#endif
Real
tnlFiniteVolumeOperatorQ< tnlGrid< 1, MeshReal, Device, MeshIndex >, Real, Index, 1 >::
boundaryDerivative( const MeshType& mesh,
          const IndexType cellIndex,
          const CoordinatesType& coordinates,
          const Vector& u,
          const Real& time,
          const IndexType& dx, 
          const IndexType& dy,
          const IndexType& dz ) const
{
    return 0.0;
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
tnlFiniteVolumeOperatorQ< tnlGrid< 1, MeshReal, Device, MeshIndex >, Real, Index, 1 >::
getValue( const MeshType& mesh,
          const IndexType cellIndex,
          const CoordinatesType& coordinates,
          const Vector& u,
          const Real& time,
          const IndexType& dx, 
          const IndexType& dy,
          const IndexType& dz ) const
{
    return 0.0;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
tnlString
tnlFiniteVolumeOperatorQ< tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index, 0 >::
getType()
{
   return tnlString( "tnlFiniteVolumeOperatorQ< " ) +
          MeshType::getType() + ", " +
          ::getType< Real >() + ", " +
          ::getType< Index >() + ", 0 >";
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
tnlString
tnlFiniteVolumeOperatorQ< tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index, 1 >::
getType()
{
   return tnlString( "tnlFiniteVolumeOperatorQ< " ) +
          MeshType::getType() + ", " +
          ::getType< Real >() + ", " +
          ::getType< Index >() + ", 1 >";
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
bool tnlFiniteVolumeOperatorQ< tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index, 1 >::setEps( const Real& eps )
{
  this->eps = eps;
  
  return true;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
bool tnlFiniteVolumeOperatorQ< tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index, 0 >::setEps( const Real& eps )
{
  this->eps = eps;
  
  return true;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
template< typename Vector >
Index 
tnlFiniteVolumeOperatorQ< tnlGrid< 1, MeshReal, Device, MeshIndex >, Real, Index, 1 >::
bind( Vector& u) 
{
    return 0;
}

#ifdef HAVE_CUDA
   __device__ __host__
#endif
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void 
tnlFiniteVolumeOperatorQ< tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index, 1 >::
update( const MeshType& mesh, const RealType& time )
{
    CoordinatesType dimensions = mesh.getDimensions();
    CoordinatesType coordinates;
    
    for( coordinates.x()=1; coordinates.x() < dimensions.x()-1; coordinates.x()++ )
        for( coordinates.y()=1; coordinates.y() < dimensions.y()-1; coordinates.y()++  )
        {
            q.setElement( mesh.getCellIndex(coordinates), getValue( mesh, mesh.getCellIndex(coordinates), coordinates, u, time ) ); 
        }
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
template< typename Vector, int AxeX, int AxeY, int AxeZ >
#ifdef HAVE_CUDA
__device__ __host__
#endif
Real
tnlFiniteVolumeOperatorQ< tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index, 0 >::
boundaryDerivative( const MeshType& mesh,
          const IndexType cellIndex,
          const CoordinatesType& coordinates,
          const Vector& u,
          const Real& time,
          const IndexType& dx, 
          const IndexType& dy,
          const IndexType& dz ) const
{
    if ( ( AxeX == 1 ) && ( AxeY == 0 ) && ( AxeZ == 0 ) )
    {
        if ( ( dx == 1 ) && ( dy == 0 ) && ( dz == 0 ) )
            return mesh.getHxInverse() * ( u[ mesh.template getCellNextToCell< 1,0 >( cellIndex ) ] - u[ cellIndex ] );
        if ( ( dx == -1 ) && ( dy == 0 ) && ( dz == 0 ) )
            return mesh.getHxInverse() * ( u[ cellIndex ] - u[ mesh.template getCellNextToCell< -1,0 >( cellIndex ) ] );
        if ( ( dx == 0 ) && ( dy == 1 ) && ( dz == 0 ) )
            return mesh.getHxInverse() * 0.25 * ( u[ mesh.template getCellNextToCell< 1,0 >( cellIndex ) ] + 
                   u[ mesh.template getCellNextToCell< 1,1 >( cellIndex ) ] - u[ mesh.template getCellNextToCell< -1,0 >( cellIndex ) ] -
                   u[ mesh.template getCellNextToCell< -1,1 >( cellIndex ) ] );
        if ( ( dx == 0 ) && ( dy == -1 ) && ( dz == 0 ) )
            return mesh.getHxInverse() * 0.25 * ( u[ mesh.template getCellNextToCell< 1,0 >( cellIndex ) ] + 
                   u[ mesh.template getCellNextToCell< 1,-1 >( cellIndex ) ] - u[ mesh.template getCellNextToCell< -1,0 >( cellIndex ) ] -
                   u[ mesh.template getCellNextToCell< -1,-1 >( cellIndex ) ] );
    }
    if ( ( AxeX == 0 ) && ( AxeY == 1 ) && ( AxeZ == 0 ) )
    {
        if ( ( dx == 0 ) && ( dy == 1 ) && ( dz == 0 ) )
            return mesh.getHyInverse() * ( u[ mesh.template getCellNextToCell< 0,1 >( cellIndex ) ] - u[ cellIndex ] );
        if ( ( dx == 0 ) && ( dy == -1 ) && ( dz == 0 ) )
            return mesh.getHyInverse() * ( u[ cellIndex ] - u[ mesh.template getCellNextToCell< 0,-1 >( cellIndex ) ] );
        if ( ( dx == 1 ) && ( dy == 0 ) && ( dz == 0 ) )
            return mesh.getHyInverse() * 0.25 * ( u[ mesh.template getCellNextToCell< 0,1 >( cellIndex ) ] + 
                   u[ mesh.template getCellNextToCell< 1,1 >( cellIndex ) ] - u[ mesh.template getCellNextToCell< 0,-1 >( cellIndex ) ] -
                   u[ mesh.template getCellNextToCell< 1,-1 >( cellIndex ) ] );
        if ( ( dx == -1 ) && ( dy == 0 ) && ( dz == 0 ) )
            return mesh.getHyInverse() * 0.25 * ( u[ mesh.template getCellNextToCell< 0,1 >( cellIndex ) ] + 
                   u[ mesh.template getCellNextToCell< -1,1 >( cellIndex ) ] - u[ mesh.template getCellNextToCell< 0,-1 >( cellIndex ) ] -
                   u[ mesh.template getCellNextToCell< -1,-1 >( cellIndex ) ] );
    }
    return 0.0;
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
tnlFiniteVolumeOperatorQ< tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index, 0 >::
getValue( const MeshType& mesh,
          const IndexType cellIndex,
          const CoordinatesType& coordinates,
          const Vector& u,
          const Real& time,
          const IndexType& dx, 
          const IndexType& dy,
          const IndexType& dz ) const
{
    if ( ( dx == 0 ) && ( dy == 0 ) && ( dz == 0 ) )
        return sqrt( this->eps + ( u[ mesh.template getCellNextToCell< 0,1 >( cellIndex ) ] - u[ cellIndex ] ) * 
                ( u[ mesh.template getCellNextToCell< 0,1 >( cellIndex ) ] - u[ cellIndex ] )
                * mesh.getHyInverse() * mesh.getHyInverse() + ( u[ mesh.template getCellNextToCell< 1,0 >( cellIndex ) ] - u[ cellIndex ] ) 
                * ( u[ mesh.template getCellNextToCell< 1,0 >( cellIndex ) ] - u[ cellIndex ] ) * mesh.getHxInverse() * mesh.getHxInverse() );
    if ( ( dx == 1 ) && ( dy == 0 ) && ( dz == 0 ) )
        return sqrt( this->eps + this->template boundaryDerivative< Vector,1,0 >( mesh, cellIndex, coordinates, u, time, 1, 0 ) * 
               this->template boundaryDerivative< Vector,1,0 >( mesh, cellIndex, coordinates, u, time, 1, 0 ) + 
               this->template boundaryDerivative< Vector,0,1 >( mesh, cellIndex, coordinates, u, time, 1, 0 ) * 
               this->template boundaryDerivative< Vector,0,1 >( mesh, cellIndex, coordinates, u, time, 1, 0 ) );
    if ( ( dx == -1 ) && ( dy == 0 ) && ( dz == 0 ) )
        return sqrt( this->eps + this->template boundaryDerivative< Vector,1,0 >( mesh, cellIndex, coordinates, u, time, -1, 0 ) * 
               this->template boundaryDerivative< Vector,1,0 >( mesh, cellIndex, coordinates, u, time, -1, 0 ) + 
               this->template boundaryDerivative< Vector,0,1 >( mesh, cellIndex, coordinates, u, time, -1, 0 ) * 
               this->template boundaryDerivative< Vector,0,1 >( mesh, cellIndex, coordinates, u, time, -1, 0 ) );
    if ( ( dx == 0 ) && ( dy == 1 ) && ( dz == 0 ) )
        return sqrt( this->eps + this->template boundaryDerivative< Vector,1,0 >( mesh, cellIndex, coordinates, u, time, 0, 1 ) * 
               this->template boundaryDerivative< Vector,1,0 >( mesh, cellIndex, coordinates, u, time, 0, 1 ) + 
               this->template boundaryDerivative< Vector,0,1 >( mesh, cellIndex, coordinates, u, time, 0, 1 ) * 
               this->template boundaryDerivative< Vector,0,1 >( mesh, cellIndex, coordinates, u, time, 0, 1 ) );
    if ( ( dx == 0 ) && ( dy == -1 ) && ( dz == 0 ) )
        return sqrt( this->eps + this->template boundaryDerivative< Vector,1,0 >( mesh, cellIndex, coordinates, u, time, 0, -1 ) * 
               this->template boundaryDerivative< Vector,1,0 >( mesh, cellIndex, coordinates, u, time, 0, -1 ) + 
               this->template boundaryDerivative< Vector,0,1 >( mesh, cellIndex, coordinates, u, time, 0, -1 ) * 
               this->template boundaryDerivative< Vector,0,1 >( mesh, cellIndex, coordinates, u, time, 0, -1 ) );
    return 0.0;
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
tnlFiniteVolumeOperatorQ< tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index, 1 >::
getValue( const MeshType& mesh,
          const IndexType cellIndex,
          const CoordinatesType& coordinates,
          const Vector& u,
          const Real& time,
          const IndexType& dx, 
          const IndexType& dy,
          const IndexType& dz ) const
{
   return q.getElement(cellIndex);
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
tnlString
tnlFiniteVolumeOperatorQ< tnlGrid< 3, MeshReal, Device, MeshIndex >, Real, Index, 0 >::
getType()
{
   return tnlString( "tnlFiniteVolumeOperatorQ< " ) +
          MeshType::getType() + ", " +
          ::getType< Real >() + ", " +
          ::getType< Index >() + ", 0 >";
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
tnlString
tnlFiniteVolumeOperatorQ< tnlGrid< 3, MeshReal, Device, MeshIndex >, Real, Index, 1 >::
getType()
{
   return tnlString( "tnlFiniteVolumeOperatorQ< " ) +
          MeshType::getType() + ", " +
          ::getType< Real >() + ", " +
          ::getType< Index >() + ", 1 >";
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
bool tnlFiniteVolumeOperatorQ< tnlGrid< 3, MeshReal, Device, MeshIndex >, Real, Index, 1 >::setEps( const Real& eps )
{
  this->eps = eps;
  
  return true;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
bool tnlFiniteVolumeOperatorQ< tnlGrid< 3, MeshReal, Device, MeshIndex >, Real, Index, 0 >::setEps( const Real& eps )
{
  this->eps = eps;
  
  return true;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
template< typename Vector >
Index 
tnlFiniteVolumeOperatorQ< tnlGrid< 3, MeshReal, Device, MeshIndex >, Real, Index, 1 >::
bind( Vector& u) 
{
    this->u.bind(u);
    if(q.setSize(u.getSize()))
        return 1;
    q.setValue(0);
    return 0;
}

#ifdef HAVE_CUDA
   __device__ __host__
#endif
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void 
tnlFiniteVolumeOperatorQ< tnlGrid< 3, MeshReal, Device, MeshIndex >, Real, Index, 1 >::
update( const MeshType& mesh, const RealType& time )
{
    CoordinatesType dimensions = mesh.getDimensions();
    CoordinatesType coordinates;
    
    for( coordinates.x()=1; coordinates.x() < dimensions.x()-1; coordinates.x()++ )
        for( coordinates.y()=1; coordinates.y() < dimensions.y()-1; coordinates.y()++ )
            for( coordinates.z()=1; coordinates.z() < dimensions.z()-1; coordinates.z()++ )
                q.setElement( mesh.getCellIndex(coordinates), getValue( mesh, mesh.getCellIndex(coordinates), coordinates, u, time ) ); 
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
template< typename Vector, int AxeX, int AxeY, int AxeZ >
#ifdef HAVE_CUDA
__device__ __host__
#endif
Real
tnlFiniteVolumeOperatorQ< tnlGrid< 3, MeshReal, Device, MeshIndex >, Real, Index, 0 >::
boundaryDerivative( const MeshType& mesh,
          const IndexType cellIndex,
          const CoordinatesType& coordinates,
          const Vector& u,
          const Real& time,
          const IndexType& dx, 
          const IndexType& dy,
          const IndexType& dz  ) const
{
    if ( ( AxeX == 1 ) && ( AxeY == 0 ) && ( AxeZ == 0 ) )
    {
        if ( ( dx == 1 ) && ( dy == 0 ) && ( dz == 0 ) )
            return mesh.getHxInverse() * ( u[ mesh.template getCellNextToCell< 1,0,0 >( cellIndex ) ] - u[ cellIndex ] );
        if ( ( dx == -1 ) && ( dy == 0 ) && ( dz == 0 ) )
            return mesh.getHxInverse() * ( u[ cellIndex ] - u[ mesh.template getCellNextToCell< -1,0,0 >( cellIndex ) ] );
        if ( ( dx == 0 ) && ( dy == 1 ) && ( dz == 0 ) )
            return mesh.getHxInverse() * 0.125 * ( u[ mesh.template getCellNextToCell< 1,0,0 >( cellIndex ) ] + 
                   u[ mesh.template getCellNextToCell< 1,1,0 >( cellIndex ) ] - u[ mesh.template getCellNextToCell< -1,0,0 >( cellIndex ) ] -
                   u[ mesh.template getCellNextToCell< -1,1,0 >( cellIndex ) ] );
        if ( ( dx == 0 ) && ( dy == -1 ) && ( dz == 0 ) )
            return mesh.getHxInverse() * 0.125 * ( u[ mesh.template getCellNextToCell< 1,0,0 >( cellIndex ) ] + 
                   u[ mesh.template getCellNextToCell< 1,-1,0 >( cellIndex ) ] - u[ mesh.template getCellNextToCell< -1,0,0 >( cellIndex ) ] -
                   u[ mesh.template getCellNextToCell< -1,-1,0 >( cellIndex ) ] );
        if ( ( dx == 0 ) && ( dy == 0 ) && ( dz == 1 ) )
            return mesh.getHxInverse() * 0.125 * ( u[ mesh.template getCellNextToCell< 1,0,0 >( cellIndex ) ] + 
                   u[ mesh.template getCellNextToCell< 1,0,1 >( cellIndex ) ] - u[ mesh.template getCellNextToCell< -1,0,0 >( cellIndex ) ] -
                   u[ mesh.template getCellNextToCell< -1,0,1 >( cellIndex ) ] );
        if ( ( dx == 0 ) && ( dy == 0 ) && ( dz == -1 ) )
            return mesh.getHxInverse() * 0.125 * ( u[ mesh.template getCellNextToCell< 1,0,0 >( cellIndex ) ] + 
                   u[ mesh.template getCellNextToCell< 1,0,-1 >( cellIndex ) ] - u[ mesh.template getCellNextToCell< -1,0,0 >( cellIndex ) ] -
                   u[ mesh.template getCellNextToCell< -1,0,-1 >( cellIndex ) ] );
    }
    if ( ( AxeX == 0 ) && ( AxeY == 1 ) && ( AxeZ == 0 ) )
    {
        if ( ( dx == 0 ) && ( dy == 1 ) && ( dz == 0 ) )
            return mesh.getHyInverse() * ( u[ mesh.template getCellNextToCell< 0,1,0 >( cellIndex ) ] - u[ cellIndex ] );
        if ( ( dx == 0 ) && ( dy == -1 ) && ( dz == 0 ) )
            return mesh.getHyInverse() * ( u[ cellIndex ] - u[ mesh.template getCellNextToCell< 0,-1,0 >( cellIndex ) ] );
        if ( ( dx == 1 ) && ( dy == 0 ) && ( dz == 0 ) )
            return mesh.getHyInverse() * 0.125 * ( u[ mesh.template getCellNextToCell< 0,1,0 >( cellIndex ) ] + 
                   u[ mesh.template getCellNextToCell< 1,1,0 >( cellIndex ) ] - u[ mesh.template getCellNextToCell< 0,-1,0 >( cellIndex ) ] -
                   u[ mesh.template getCellNextToCell< 1,-1,0 >( cellIndex ) ] );
        if ( ( dx == -1 ) && ( dy == 0 ) && ( dz == 0 ) )
            return mesh.getHyInverse() * 0.125 * ( u[ mesh.template getCellNextToCell< 0,1,0 >( cellIndex ) ] + 
                   u[ mesh.template getCellNextToCell< -1,1,0 >( cellIndex ) ] - u[ mesh.template getCellNextToCell< 0,-1,0 >( cellIndex ) ] -
                   u[ mesh.template getCellNextToCell< -1,-1,0 >( cellIndex ) ] );
        if ( ( dx == 0 ) && ( dy == 0 ) && ( dz == 1 ) )
            return mesh.getHyInverse() * 0.125 * ( u[ mesh.template getCellNextToCell< 0,1,0 >( cellIndex ) ] + 
                   u[ mesh.template getCellNextToCell< 0,1,1 >( cellIndex ) ] - u[ mesh.template getCellNextToCell< 0,-1,0 >( cellIndex ) ] -
                   u[ mesh.template getCellNextToCell< 0,-1,1 >( cellIndex ) ] );
        if ( ( dx == 0 ) && ( dy == 0 ) && ( dz == -1 ) )
            return mesh.getHyInverse() * 0.125 * ( u[ mesh.template getCellNextToCell< 0,1,0 >( cellIndex ) ] + 
                   u[ mesh.template getCellNextToCell< 0,1,-1 >( cellIndex ) ] - u[ mesh.template getCellNextToCell< 0,-1,0 >( cellIndex ) ] -
                   u[ mesh.template getCellNextToCell< 0,-1,-1 >( cellIndex ) ] );
    }
    if ( ( AxeX == 0 ) && ( AxeY == 0 ) && ( AxeZ == 1 ) )
    {
        if ( ( dx == 0 ) && ( dy == 0 ) && ( dz == 1 ) )
            return mesh.getHzInverse() * ( u[ mesh.template getCellNextToCell< 0,0,1 >( cellIndex ) ] - u[ cellIndex ] );
        if ( ( dx == 0 ) && ( dy == 0 ) && ( dz == -1 ) )
            return mesh.getHzInverse() * ( u[ cellIndex ] - u[ mesh.template getCellNextToCell< 0,0,-1 >( cellIndex ) ] );
        if ( ( dx == 1 ) && ( dy == 0 ) && ( dz == 0 ) )
            return mesh.getHzInverse() * 0.125 * ( u[ mesh.template getCellNextToCell< 0,0,1 >( cellIndex ) ] + 
                   u[ mesh.template getCellNextToCell< 1,0,1 >( cellIndex ) ] - u[ mesh.template getCellNextToCell< 0,0,-1 >( cellIndex ) ] -
                   u[ mesh.template getCellNextToCell< 1,0,-1 >( cellIndex ) ] );
        if ( ( dx == -1 ) && ( dy == 0 ) && ( dz == 0 ) )
            return mesh.getHzInverse() * 0.125 * ( u[ mesh.template getCellNextToCell< 0,0,1 >( cellIndex ) ] + 
                   u[ mesh.template getCellNextToCell< -1,0,1 >( cellIndex ) ] - u[ mesh.template getCellNextToCell< 0,0,-1 >( cellIndex ) ] -
                   u[ mesh.template getCellNextToCell< -1,0,-1 >( cellIndex ) ] );
        if ( ( dx == 0 ) && ( dy == 1 ) && ( dz == 0 ) )
            return mesh.getHzInverse() * 0.125 * ( u[ mesh.template getCellNextToCell< 0,0,1 >( cellIndex ) ] + 
                   u[ mesh.template getCellNextToCell< 0,1,1 >( cellIndex ) ] - u[ mesh.template getCellNextToCell< 0,0,-1 >( cellIndex ) ] -
                   u[ mesh.template getCellNextToCell< 0,1,-1 >( cellIndex ) ] );
        if ( ( dx == 0 ) && ( dy == -1 ) && ( dz == 0 ) )
            return mesh.getHzInverse() * 0.125 * ( u[ mesh.template getCellNextToCell< 0,0,1 >( cellIndex ) ] + 
                   u[ mesh.template getCellNextToCell< 0,-1,1 >( cellIndex ) ] - u[ mesh.template getCellNextToCell< 0,0,-1 >( cellIndex ) ] -
                   u[ mesh.template getCellNextToCell< 0,-1,-1 >( cellIndex ) ] );
    }
    return 0.0;
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
tnlFiniteVolumeOperatorQ< tnlGrid< 3, MeshReal, Device, MeshIndex >, Real, Index, 0 >::
getValue( const MeshType& mesh,
          const IndexType cellIndex,
          const CoordinatesType& coordinates,
          const Vector& u,
          const Real& time,
          const IndexType& dx, 
          const IndexType& dy,
          const IndexType& dz ) const
{
    if ( ( dx == 0 ) && ( dy == 0 ) && ( dz == 0 ) )
        return sqrt( this->eps + ( u[ mesh.template getCellNextToCell< 0,1,0 >( cellIndex ) ] - u[ cellIndex ] ) * 
                ( u[ mesh.template getCellNextToCell< 0,1,0 >( cellIndex ) ] - u[ cellIndex ] )
                * mesh.getHyInverse() * mesh.getHyInverse() + ( u[ mesh.template getCellNextToCell< 1,0,0 >( cellIndex ) ] - u[ cellIndex ] ) 
                * ( u[ mesh.template getCellNextToCell< 1,0,0 >( cellIndex ) ] - u[ cellIndex ] ) * mesh.getHxInverse() * mesh.getHxInverse()
                + ( u[ mesh.template getCellNextToCell< 0,0,1 >( cellIndex ) ] - u[ cellIndex ] ) 
                * ( u[ mesh.template getCellNextToCell< 0,0,1 >( cellIndex ) ] - u[ cellIndex ] ) * mesh.getHzInverse() * mesh.getHzInverse() );
    if ( ( dx == 1 ) && ( dy == 0 ) && ( dz == 0 ) )
        return sqrt( this->eps + this->template boundaryDerivative< Vector,1,0,0 >( mesh, cellIndex, coordinates, u, time, 1, 0, 0 ) * 
               this->template boundaryDerivative< Vector,1,0,0 >( mesh, cellIndex, coordinates, u, time, 1, 0, 0 ) + 
               this->template boundaryDerivative< Vector,0,1,0 >( mesh, cellIndex, coordinates, u, time, 1, 0, 0 ) * 
               this->template boundaryDerivative< Vector,0,1,0 >( mesh, cellIndex, coordinates, u, time, 1, 0, 0 ) + 
               this->template boundaryDerivative< Vector,0,0,1 >( mesh, cellIndex, coordinates, u, time, 1, 0, 0 ) * 
               this->template boundaryDerivative< Vector,0,0,1 >( mesh, cellIndex, coordinates, u, time, 1, 0, 0 ) );
    if ( ( dx == -1 ) && ( dy == 0 ) && ( dz == 0 ) )
        return sqrt( this->eps + this->template boundaryDerivative< Vector,1,0,0 >( mesh, cellIndex, coordinates, u, time, -1, 0, 0 ) * 
               this->template boundaryDerivative< Vector,1,0,0 >( mesh, cellIndex, coordinates, u, time, -1, 0, 0 ) + 
               this->template boundaryDerivative< Vector,0,1,0 >( mesh, cellIndex, coordinates, u, time, -1, 0, 0 ) * 
               this->template boundaryDerivative< Vector,0,1,0 >( mesh, cellIndex, coordinates, u, time, -1, 0, 0 ) +
               this->template boundaryDerivative< Vector,0,0,1 >( mesh, cellIndex, coordinates, u, time, -1, 0, 0 ) * 
               this->template boundaryDerivative< Vector,0,0,1 >( mesh, cellIndex, coordinates, u, time, -1, 0, 0 ) );
    if ( ( dx == 0 ) && ( dy == 1 ) && ( dz == 0 ) )
        return sqrt( this->eps + this->template boundaryDerivative< Vector,1,0,0 >( mesh, cellIndex, coordinates, u, time, 0, 1, 0 ) * 
               this->template boundaryDerivative< Vector,1,0,0 >( mesh, cellIndex, coordinates, u, time, 0, 1, 0 ) + 
               this->template boundaryDerivative< Vector,0,1,0 >( mesh, cellIndex, coordinates, u, time, 0, 1, 0 ) * 
               this->template boundaryDerivative< Vector,0,1,0 >( mesh, cellIndex, coordinates, u, time, 0, 1, 0 ) +
               this->template boundaryDerivative< Vector,0,0,1 >( mesh, cellIndex, coordinates, u, time, 0, 1, 0 ) * 
               this->template boundaryDerivative< Vector,0,0,1 >( mesh, cellIndex, coordinates, u, time, 0, 1, 0 ));
    if ( ( dx == 0 ) && ( dy == -1 ) && ( dz == 0 ) )
        return sqrt( this->eps + this->template boundaryDerivative< Vector,1,0,0 >( mesh, cellIndex, coordinates, u, time, 0, -1, 0 ) * 
               this->template boundaryDerivative< Vector,1,0,0 >( mesh, cellIndex, coordinates, u, time, 0, -1, 0 ) + 
               this->template boundaryDerivative< Vector,0,1,0 >( mesh, cellIndex, coordinates, u, time, 0, -1, 0 ) * 
               this->template boundaryDerivative< Vector,0,1,0 >( mesh, cellIndex, coordinates, u, time, 0, -1, 0 ) +
               this->template boundaryDerivative< Vector,0,0,1 >( mesh, cellIndex, coordinates, u, time, 0, -1, 0 ) * 
               this->template boundaryDerivative< Vector,0,0,1 >( mesh, cellIndex, coordinates, u, time, 0, -1, 0 ) );
    if ( ( dx == 0 ) && ( dy == 0 ) && ( dz == 1 ) )
        return sqrt( this->eps + this->template boundaryDerivative< Vector,1,0,0 >( mesh, cellIndex, coordinates, u, time, 0, 0, 1 ) * 
               this->template boundaryDerivative< Vector,1,0,0 >( mesh, cellIndex, coordinates, u, time, 0, 0, 1 ) + 
               this->template boundaryDerivative< Vector,0,1,0 >( mesh, cellIndex, coordinates, u, time, 0, 0, 1 ) * 
               this->template boundaryDerivative< Vector,0,1,0 >( mesh, cellIndex, coordinates, u, time, 0, 0, 1 ) +
               this->template boundaryDerivative< Vector,0,0,1 >( mesh, cellIndex, coordinates, u, time, 0, 0, 1 ) * 
               this->template boundaryDerivative< Vector,0,0,1 >( mesh, cellIndex, coordinates, u, time, 0, 0, 1 ));
    if ( ( dx == 0 ) && ( dy == 0 ) && ( dz == -1 ) )
        return sqrt( this->eps + this->template boundaryDerivative< Vector,1,0,0 >( mesh, cellIndex, coordinates, u, time, 0, 0, -1 ) * 
               this->template boundaryDerivative< Vector,1,0,0 >( mesh, cellIndex, coordinates, u, time, 0, 0, -1 ) + 
               this->template boundaryDerivative< Vector,0,1,0 >( mesh, cellIndex, coordinates, u, time, 0, 0, -1 ) * 
               this->template boundaryDerivative< Vector,0,1,0 >( mesh, cellIndex, coordinates, u, time, 0, 0, -1 ) +
               this->template boundaryDerivative< Vector,0,0,1 >( mesh, cellIndex, coordinates, u, time, 0, 0, -1 ) * 
               this->template boundaryDerivative< Vector,0,0,1 >( mesh, cellIndex, coordinates, u, time, 0, 0, -1 ) );
    return 0.0;
}
#endif	/* TNLFINITEVOLUMEOPERATORQ_IMPL_H */
