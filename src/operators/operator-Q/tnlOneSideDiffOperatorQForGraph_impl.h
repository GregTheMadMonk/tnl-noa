
#ifndef TNLONESIDEDIFFOPERATORQFORGRAPH_IMPL_H
#define	TNLONESIDEDIFFOPERATORQFORGRAPH_IMPL_H

#include <operators/operator-Q/tnlOneSideDiffOperatorQForGraph.h>
#include <mesh/tnlGrid.h>

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
tnlString
tnlOneSideDiffOperatorQForGraph< tnlGrid< 1, MeshReal, Device, MeshIndex >, Real, Index, 0 >::
getType()
{
   return tnlString( "tnlOneSideDiffOperatorQForGraph< " ) +
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
tnlOneSideDiffOperatorQForGraph< tnlGrid< 1, MeshReal, Device, MeshIndex >, Real, Index, 1 >::
getType()
{
   return tnlString( "tnlOneSideDiffOperatorQForGraph< " ) +
          MeshType::getType() + ", " +
          ::getType< Real >() + ", " +
          ::getType< Index >() + ", 1 >";
}

template< typename MeshReal,
        typename Device, 
        typename MeshIndex,
        typename Real,
        typename Index >
template< typename Vector >
Index 
tnlOneSideDiffOperatorQForGraph< tnlGrid< 1, MeshReal, Device, MeshIndex >, Real, Index, 1 >::
bind( Vector& u )
{
    this->u.bind(u);
    if(q.setSize(u.getSize()))
        return 1;
    if(qStriped.setSize(u.getSize()))
        return 1;
    q.setValue(0);
    qStriped.setValue(0);
    return 0;    
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
tnlOneSideDiffOperatorQForGraph< tnlGrid< 1, MeshReal, Device, MeshIndex >, Real, Index, 1 >::
update( const MeshType& mesh, const RealType& time )
{
    CoordinatesType dimensions = mesh.getDimensions();
    CoordinatesType coordinates;
    for( coordinates.x()=1; coordinates.x() < dimensions.x()-1; coordinates.x()++ )
    {
        q.setElement( coordinates.x(), getValue( mesh, mesh.getCellIndex(coordinates), coordinates, u, time ) ); 
    }
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
tnlOneSideDiffOperatorQForGraph< tnlGrid< 1, MeshReal, Device, MeshIndex >, Real, Index, 0 >::
getValue( const MeshType& mesh,
          const IndexType cellIndex,
          const CoordinatesType& coordinates,
          const Vector& u,
          const Real& time ) const
{
   return sqrt( 1.0 + ( u[ mesh.template getCellNextToCell< 1 >( cellIndex ) ] - u[ cellIndex ]) * 
          ( u[ mesh.template getCellNextToCell< 1 >( cellIndex ) ] - u[ cellIndex ]) *
          mesh.getHxInverse() * mesh.getHxInverse() );
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
tnlOneSideDiffOperatorQForGraph< tnlGrid< 1, MeshReal, Device, MeshIndex >, Real, Index, 1 >::
getValue( const MeshType& mesh,
          const IndexType cellIndex,
          const CoordinatesType& coordinates,
          const Vector& u,
          const Real& time ) const
{
   return q.getElement(cellIndex);
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
tnlOneSideDiffOperatorQForGraph< tnlGrid< 1, MeshReal, Device, MeshIndex >, Real, Index, 0 >::
getValueStriped( const MeshType& mesh,
          const IndexType cellIndex,
          const CoordinatesType& coordinates,
          const Vector& u,
          const Real& time ) const
{
   return sqrt( 1.0 + 0.5*( ( u[ mesh.template getCellNextToCell< 1 >( cellIndex ) ] - u[ cellIndex ]) * 
          ( u[ mesh.template getCellNextToCell< 1 >( cellIndex ) ] - u[ cellIndex ]) *
          mesh.getHxInverse() * mesh.getHxInverse() + ( - u[ mesh.template getCellNextToCell< -1 >( cellIndex ) ] + u[ cellIndex ] ) 
          * ( - u[ mesh.template getCellNextToCell< -1 >( cellIndex ) ] + u[ cellIndex ] ) * mesh.getHxInverse() * mesh.getHxInverse() ) );
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
tnlOneSideDiffOperatorQForGraph< tnlGrid< 1, MeshReal, Device, MeshIndex >, Real, Index, 1 >::
getValueStriped( const MeshType& mesh,
          const IndexType cellIndex,
          const CoordinatesType& coordinates,
          const Vector& u,
          const Real& time ) const
{
   return qStriped.getElement(cellIndex);
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
tnlString
tnlOneSideDiffOperatorQForGraph< tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index, 0 >::
getType()
{
   return tnlString( "tnlOneSideDiffOperatorQForGraph< " ) +
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
tnlOneSideDiffOperatorQForGraph< tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index, 1 >::
getType()
{
   return tnlString( "tnlOneSideDiffOperatorQForGraph< " ) +
          MeshType::getType() + ", " +
          ::getType< Real >() + ", " +
          ::getType< Index >() + ", 1 >";
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
template< typename Vector >
Index 
tnlOneSideDiffOperatorQForGraph< tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index, 1 >::
bind( Vector& u) 
{
    this->u.bind(u);
    if(q.setSize(u.getSize()))
        return 1;
    if(qStriped.setSize(u.getSize()))
        return 1;
    q.setValue(0);
    qStriped.setValue(0);
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
tnlOneSideDiffOperatorQForGraph< tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index, 1 >::
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
template< typename Vector >
#ifdef HAVE_CUDA
__device__ __host__
#endif
Real
tnlOneSideDiffOperatorQForGraph< tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index, 0 >::
getValue( const MeshType& mesh,
          const IndexType cellIndex,
          const CoordinatesType& coordinates,
          const Vector& u,
          const Real& time ) const
{
   return sqrt( 1.0 + ( u[ mesh.template getCellNextToCell< 0,1 >( cellIndex ) ] - u[ cellIndex ] ) * 
          ( u[ mesh.template getCellNextToCell< 0,1 >( cellIndex ) ] - u[ cellIndex ] )
          * mesh.getHyInverse() * mesh.getHyInverse() + ( u[ mesh.template getCellNextToCell< 1,0 >( cellIndex ) ] - u[ cellIndex ] ) 
          * ( u[ mesh.template getCellNextToCell< 1,0 >( cellIndex ) ] - u[ cellIndex ] ) * mesh.getHxInverse() * mesh.getHxInverse() );
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
tnlOneSideDiffOperatorQForGraph< tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index, 1 >::
getValue( const MeshType& mesh,
          const IndexType cellIndex,
          const CoordinatesType& coordinates,
          const Vector& u,
          const Real& time ) const
{
   return q.getElement(cellIndex);
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
tnlOneSideDiffOperatorQForGraph< tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index, 0 >::
getValueStriped( const MeshType& mesh,
          const IndexType cellIndex,
          const CoordinatesType& coordinates,
          const Vector& u,
          const Real& time ) const
{
   return sqrt( 1.0 + 0.5*( ( u[ mesh.template getCellNextToCell< 0,1 >( cellIndex ) ] - u[ cellIndex ] ) * 
          ( u[ mesh.template getCellNextToCell< 0,1 >( cellIndex ) ] - u[ cellIndex ] )
          * mesh.getHyInverse() * mesh.getHyInverse() + ( u[ mesh.template getCellNextToCell< 1,0 >( cellIndex ) ] - u[ cellIndex ] ) 
          * ( u[ mesh.template getCellNextToCell< 1,0 >( cellIndex ) ] - u[ cellIndex ] ) * mesh.getHxInverse() * mesh.getHxInverse()
          + ( - u[ mesh.template getCellNextToCell< -1,0 >( cellIndex ) ] + u[ cellIndex ]) * ( - u[ mesh.template getCellNextToCell< -1,0 >( cellIndex ) ] + u[ cellIndex ]) 
          * mesh.getHxInverse() * mesh.getHxInverse()
          + ( - u[ mesh.template getCellNextToCell< 0,-1 >( cellIndex ) ] + u[ cellIndex ]) * ( - u[ mesh.template getCellNextToCell< 0,-1 >( cellIndex ) ] + u[ cellIndex ]) * 
          mesh.getHyInverse() * mesh.getHyInverse() ) );
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
tnlOneSideDiffOperatorQForGraph< tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index, 1 >::
getValueStriped( const MeshType& mesh,
          const IndexType cellIndex,
          const CoordinatesType& coordinates,
          const Vector& u,
          const Real& time ) const
{
   return qStriped.getElement(cellIndex);
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
tnlString
tnlOneSideDiffOperatorQForGraph< tnlGrid< 3, MeshReal, Device, MeshIndex >, Real, Index, 0 >::
getType()
{
   return tnlString( "tnlOneSideDiffOperatorQForGraph< " ) +
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
tnlOneSideDiffOperatorQForGraph< tnlGrid< 3, MeshReal, Device, MeshIndex >, Real, Index, 1 >::
getType()
{
   return tnlString( "tnlOneSideDiffOperatorQForGraph< " ) +
          MeshType::getType() + ", " +
          ::getType< Real >() + ", " +
          ::getType< Index >() + ", 1 >";
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
template< typename Vector >
Index 
tnlOneSideDiffOperatorQForGraph< tnlGrid< 3, MeshReal, Device, MeshIndex >, Real, Index, 1 >::
bind( Vector& u) 
{
    this->u.bind(u);
    if(q.setSize(u.getSize()))
        return 1;
    if(qStriped.setSize(u.getSize()))
        return 1;
    q.setValue(0);
    qStriped.setValue(0);
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
tnlOneSideDiffOperatorQForGraph< tnlGrid< 3, MeshReal, Device, MeshIndex >, Real, Index, 1 >::
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
template< typename Vector >
#ifdef HAVE_CUDA
__device__ __host__
#endif
Real
tnlOneSideDiffOperatorQForGraph< tnlGrid< 3, MeshReal, Device, MeshIndex >, Real, Index, 0 >::
getValue( const MeshType& mesh,
          const IndexType cellIndex,
          const CoordinatesType& coordinates,
          const Vector& u,
          const Real& time ) const
{
   return sqrt( 1.0 + ( u[ mesh.template getCellNextToCell< 0,1,0 >( cellIndex ) ] - u[ cellIndex ] ) * 
          ( u[ mesh.template getCellNextToCell< 0,1,0 >( cellIndex ) ] - u[ cellIndex ] )
          * mesh.getHyInverse() * mesh.getHyInverse() + ( u[ mesh.template getCellNextToCell< 1,0,0 >( cellIndex ) ] - u[ cellIndex ] ) 
          * ( u[ mesh.template getCellNextToCell< 1,0,0 >( cellIndex ) ] - u[ cellIndex ] ) * mesh.getHxInverse() * mesh.getHxInverse() 
          + ( u[ mesh.template getCellNextToCell< 0,0,1 >( cellIndex ) ] - u[ cellIndex ] ) * ( u[ mesh.template getCellNextToCell< 0,0,1 >( cellIndex ) ] - u[ cellIndex ] )
            * mesh.getHzInverse() * mesh.getHzInverse() );
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
tnlOneSideDiffOperatorQForGraph< tnlGrid< 3, MeshReal, Device, MeshIndex >, Real, Index, 0 >::
getValueStriped( const MeshType& mesh,
          const IndexType cellIndex,
          const CoordinatesType& coordinates,
          const Vector& u,
          const Real& time ) const
{
   return sqrt( 1.0 + 0.5*( ( u[ mesh.template getCellNextToCell< 0,1,0 >( cellIndex ) ] - u[ cellIndex ] ) * 
           ( u[ mesh.template getCellNextToCell< 0,1,0 >( cellIndex ) ] - u[ cellIndex ] ) * mesh.getHyInverse() * mesh.getHyInverse() 
           + ( u[ mesh.template getCellNextToCell< 1,0,0 >( cellIndex ) ] - u[ cellIndex ] ) * ( u[ mesh.template getCellNextToCell< 1,0,0 >( cellIndex ) ] - u[ cellIndex ] ) 
           * mesh.getHxInverse() * mesh.getHxInverse() + ( - u[ mesh.template getCellNextToCell< -1,0,0 >( cellIndex ) ] + u[ cellIndex ]) 
           * ( - u[ mesh.template getCellNextToCell< -1,0,0 >( cellIndex ) ] + u[ cellIndex ]) * mesh.getHxInverse() * mesh.getHxInverse()
           + ( - u[ mesh.template getCellNextToCell< 0,-1,0 >( cellIndex ) ] + u[ cellIndex ]) * 
           ( - u[ mesh.template getCellNextToCell< 0,-1,0 >( cellIndex ) ] + u[ cellIndex ]) * mesh.getHyInverse() * mesh.getHyInverse() 
           + ( u[ mesh.template getCellNextToCell< 0,0,1 >( cellIndex ) ] - u[ cellIndex ] ) * ( u[ mesh.template getCellNextToCell< 0,0,1 >( cellIndex ) ] - u[ cellIndex ] )
           * mesh.getHzInverse() * mesh.getHzInverse() + ( - u[ mesh.template getCellNextToCell< 0,0,-1 >( cellIndex ) ] + u[ cellIndex ]) * 
           ( - u[ mesh.template getCellNextToCell< 0,0,-1 >( cellIndex ) ] + u[ cellIndex ]) * mesh.getHzInverse() * mesh.getHzInverse()
           ) );
}   
#endif	/* TNLONESIDEDIFFOPERATORQFORGRAPH_IMPL_H */
