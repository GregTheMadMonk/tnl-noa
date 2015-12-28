
#ifndef TNLONESIDEDIFFOPERATORQ_IMPL_H
#define	TNLONESIDEDIFFOPERATORQ_IMPL_H

#include <operators/operator-Q/tnlOneSideDiffOperatorQ.h>
#include <mesh/tnlGrid.h>

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
tnlString
tnlOneSideDiffOperatorQ< tnlGrid< 1, MeshReal, Device, MeshIndex >, Real, Index, 0 >::
getType()
{
   return tnlString( "tnlOneSideDiffOperatorQ< " ) +
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
tnlOneSideDiffOperatorQ< tnlGrid< 1, MeshReal, Device, MeshIndex >, Real, Index, 1 >::
getType()
{
   return tnlString( "tnlOneSideDiffOperatorQ< " ) +
          MeshType::getType() + ", " +
          ::getType< Real >() + ", " +
          ::getType< Index >() + ", 1 >";
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
bool tnlOneSideDiffOperatorQ< tnlGrid< 1, MeshReal, Device, MeshIndex >, Real, Index, 0 >::setEps( const Real& eps )
{
  this->eps = eps;
  
  return true;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
bool tnlOneSideDiffOperatorQ< tnlGrid< 1, MeshReal, Device, MeshIndex >, Real, Index, 1 >::setEps( const Real& eps )
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
tnlOneSideDiffOperatorQ< tnlGrid< 1, MeshReal, Device, MeshIndex >, Real, Index, 1 >::
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
__cuda_callable__
void  
tnlOneSideDiffOperatorQ< tnlGrid< 1, MeshReal, Device, MeshIndex >, Real, Index, 1 >::
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
template< typename MeshEntity, typename Vector >
__cuda_callable__
Real
tnlOneSideDiffOperatorQ< tnlGrid< 1, MeshReal, Device, MeshIndex >, Real, Index, 0 >::
getValue( const MeshType& mesh,
          const MeshEntity& entity,
          const Vector& u,
          const Real& time ) const
{
   const IndexType& cellIndex = entity.getIndex();
   const typename MeshEntity::template NeighbourEntities< 1 >& neighbourEntities = entity.getNeighbourEntities();      
   // TODO: fix this
   return sqrt( this->eps + 
      ( u[ neighbourEntities.template getEntityIndex< 1 >() ] - u[ cellIndex ]) * 
      ( u[ neighbourEntities.template getEntityIndex< 1 >() ] - u[ cellIndex ]) *
          mesh.template getSpaceStepsProducts< -2 >() );
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
template< typename MeshEntity, typename Vector >
__cuda_callable__
Real
tnlOneSideDiffOperatorQ< tnlGrid< 1, MeshReal, Device, MeshIndex >, Real, Index, 1 >::
getValue( const MeshType& mesh,
          const MeshEntity& entity,
          const Vector& u,
          const Real& time ) const
{
   return q.getElement( entity.getIndex() );
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
template< typename MeshEntity, typename Vector >
__cuda_callable__
Real
tnlOneSideDiffOperatorQ< tnlGrid< 1, MeshReal, Device, MeshIndex >, Real, Index, 0 >::
getValueStriped( const MeshType& mesh,
                 const MeshEntity& entity,
                 const Vector& u,
                 const Real& time ) const
{
   const IndexType& cellIndex = entity.getIndex();
   const typename MeshEntity::template NeighbourEntities< 1 >& neighbourEntities = entity.getNeighbourEntities();      
   return sqrt( this->eps + 0.5*( ( u[ neighbourEntities.template getEntityIndex< 1 >() ] - u[ cellIndex ]) * 
          (  u[ neighbourEntities.template getEntityIndex<  1 >() ] - u[ cellIndex ]) * mesh.template getSpaceStepsProducts< -1 >() * mesh.template getSpaceStepsProducts< -1 >() + 
          ( -u[ neighbourEntities.template getEntityIndex< -1 >() ] + u[ cellIndex ] ) 
          * ( - u[ neighbourEntities.template getEntityIndex< -1 >() ] + u[ cellIndex ] ) * mesh.template getSpaceStepsProducts< -1 >() * mesh.template getSpaceStepsProducts< -1 >() ) );
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
template< typename MeshEntity, typename Vector >
__cuda_callable__
Real
tnlOneSideDiffOperatorQ< tnlGrid< 1, MeshReal, Device, MeshIndex >, Real, Index, 1 >::
getValueStriped( const MeshType& mesh,
          const IndexType cellIndex,
          const MeshEntity& entity,
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
tnlOneSideDiffOperatorQ< tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index, 0 >::
getType()
{
   return tnlString( "tnlOneSideDiffOperatorQ< " ) +
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
tnlOneSideDiffOperatorQ< tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index, 1 >::
getType()
{
   return tnlString( "tnlOneSideDiffOperatorQ< " ) +
          MeshType::getType() + ", " +
          ::getType< Real >() + ", " +
          ::getType< Index >() + ", 1 >";
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
bool tnlOneSideDiffOperatorQ< tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index, 1 >::setEps( const Real& eps )
{
  this->eps = eps;
  
  return true;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
bool tnlOneSideDiffOperatorQ< tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index, 0 >::setEps( const Real& eps )
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
tnlOneSideDiffOperatorQ< tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index, 1 >::
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

__cuda_callable__
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void 
tnlOneSideDiffOperatorQ< tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index, 1 >::
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
template< typename MeshEntity, typename Vector >
__cuda_callable__
Real
tnlOneSideDiffOperatorQ< tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index, 0 >::
getValue( const MeshType& mesh,          
          const MeshEntity& entity,
          const Vector& u,
          const Real& time ) const
{
   const IndexType& cellIndex = entity.getIndex();
   const typename MeshEntity::template NeighbourEntities< 2 >& neighbourEntities = entity.getNeighbourEntities();      
   return sqrt( this->eps + ( u[ neighbourEntities.template getEntityIndex< 0,1 >() ] - u[ cellIndex ] ) * 
          ( u[ neighbourEntities.template getEntityIndex< 0, 1 >() ] - u[ cellIndex ] )
          * mesh.template getSpaceStepsProducts< 0, -1 >() * mesh.template getSpaceStepsProducts< 0, -1 >() + ( u[ neighbourEntities.template getEntityIndex< 1,0 >() ] - u[ cellIndex ] ) 
          * ( u[ neighbourEntities.template getEntityIndex< 1, 0 >() ] - u[ cellIndex ] ) * mesh.template getSpaceStepsProducts< -1, 0 >() * mesh.template getSpaceStepsProducts< -1, 0 >() );
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
template< typename MeshEntity, typename Vector >
__cuda_callable__
Real
tnlOneSideDiffOperatorQ< tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index, 1 >::
getValue( const MeshType& mesh,
          const IndexType cellIndex,
          const MeshEntity& entity,
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
template< typename MeshEntity, typename Vector >
__cuda_callable__
Real
tnlOneSideDiffOperatorQ< tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index, 0 >::
getValueStriped( const MeshType& mesh,          
                 const MeshEntity& entity,
                 const Vector& u,
                 const Real& time ) const
{
   const IndexType& cellIndex = entity.getIndex();
   const typename MeshEntity::template NeighbourEntities< 2 >& neighbourEntities = entity.getNeighbourEntities();      
   return sqrt( this->eps + 0.5*( ( u[ neighbourEntities.template getEntityIndex< 0,1 >() ] - u[ cellIndex ] ) * 
          ( u[ neighbourEntities.template getEntityIndex< 0, 1 >() ] - u[ cellIndex ] )
          * mesh.template getSpaceStepsProducts< 0, -1 >() * mesh.template getSpaceStepsProducts< 0, -1 >() + ( u[ neighbourEntities.template getEntityIndex< 1,0 >() ] - u[ cellIndex ] ) 
          * ( u[ neighbourEntities.template getEntityIndex< 1, 0 >() ] - u[ cellIndex ] ) * mesh.template getSpaceStepsProducts< -1, 0 >() * mesh.template getSpaceStepsProducts< -1, 0 >()
          + ( - u[ neighbourEntities.template getEntityIndex< -1, 0 >() ] + u[ cellIndex ]) * ( - u[ neighbourEntities.template getEntityIndex< -1,0 >() ] + u[ cellIndex ]) 
          * mesh.template getSpaceStepsProducts< -1, 0 >() * mesh.template getSpaceStepsProducts< -1, 0 >()
          + ( - u[ neighbourEntities.template getEntityIndex< 0,-1 >() ] + u[ cellIndex ]) * ( - u[ neighbourEntities.template getEntityIndex< 0,-1 >() ] + u[ cellIndex ]) * 
          mesh.template getSpaceStepsProducts< 0, -1 >() * mesh.template getSpaceStepsProducts< 0, -1 >() ) );
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
template< typename MeshEntity, typename Vector >
__cuda_callable__
Real
tnlOneSideDiffOperatorQ< tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index, 1 >::
getValueStriped( const MeshType& mesh,
          const MeshEntity& entity,
          const Vector& u,
          const Real& time ) const
{
   return qStriped.getElement( entity.getIndex() );
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
tnlString
tnlOneSideDiffOperatorQ< tnlGrid< 3, MeshReal, Device, MeshIndex >, Real, Index, 0 >::
getType()
{
   return tnlString( "tnlOneSideDiffOperatorQ< " ) +
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
tnlOneSideDiffOperatorQ< tnlGrid< 3, MeshReal, Device, MeshIndex >, Real, Index, 1 >::
getType()
{
   return tnlString( "tnlOneSideDiffOperatorQ< " ) +
          MeshType::getType() + ", " +
          ::getType< Real >() + ", " +
          ::getType< Index >() + ", 1 >";
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
bool tnlOneSideDiffOperatorQ< tnlGrid< 3, MeshReal, Device, MeshIndex >, Real, Index, 1 >::setEps( const Real& eps )
{
  this->eps = eps;
  
  return true;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
bool tnlOneSideDiffOperatorQ< tnlGrid< 3, MeshReal, Device, MeshIndex >, Real, Index, 0 >::setEps( const Real& eps )
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
tnlOneSideDiffOperatorQ< tnlGrid< 3, MeshReal, Device, MeshIndex >, Real, Index, 1 >::
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

__cuda_callable__
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void 
tnlOneSideDiffOperatorQ< tnlGrid< 3, MeshReal, Device, MeshIndex >, Real, Index, 1 >::
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
template< typename MeshEntity, typename Vector >
__cuda_callable__
Real
tnlOneSideDiffOperatorQ< tnlGrid< 3, MeshReal, Device, MeshIndex >, Real, Index, 0 >::
getValue( const MeshType& mesh,
          const MeshEntity& entity,
          const Vector& u,
          const Real& time ) const
{
   const IndexType& cellIndex = entity.getIndex();
   const typename MeshEntity::template NeighbourEntities< 3 >& neighbourEntities = entity.getNeighbourEntities();      
   return sqrt( 1.0 + ( u[ neighbourEntities.template getEntityIndex< 0,1,0 >() ] - u[ cellIndex ] ) * 
          ( u[ neighbourEntities.template getEntityIndex< 0,1,0 >() ] - u[ cellIndex ] )
          * mesh.template getSpaceStepsProducts< 0, -1, 0 >() * mesh.template getSpaceStepsProducts< 0, -1, 0 >() + ( u[ neighbourEntities.template getEntityIndex< 1,0,0 >() ] - u[ cellIndex ] ) 
          * ( u[ neighbourEntities.template getEntityIndex< 1,0,0 >() ] - u[ cellIndex ] ) * mesh.template getSpaceStepsProducts< -1, 0, 0 >() * mesh.template getSpaceStepsProducts< -1, 0, 0 >() 
          + ( u[ neighbourEntities.template getEntityIndex< 0,0,1 >() ] - u[ cellIndex ] ) * ( u[ neighbourEntities.template getEntityIndex< 0,0,1 >() ] - u[ cellIndex ] )
            * mesh.template getSpaceStepsProducts< 0, 0, -1 >() * mesh.template getSpaceStepsProducts< 0, 0, -1 >() );
}
   
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
template< typename MeshEntity, typename Vector >
__cuda_callable__
Real
tnlOneSideDiffOperatorQ< tnlGrid< 3, MeshReal, Device, MeshIndex >, Real, Index, 0 >::
getValueStriped( const MeshType& mesh,          
                 const MeshEntity& entity,
                 const Vector& u,
                 const Real& time ) const
{
   const IndexType& cellIndex = entity.getIndex();
   const typename MeshEntity::template NeighbourEntities< 3 >& neighbourEntities = entity.getNeighbourEntities();      
   return sqrt( this->eps + 0.5*( ( u[ neighbourEntities.template getEntityIndex< 0,1,0 >() ] - u[ cellIndex ] ) * 
           ( u[ neighbourEntities.template getEntityIndex< 0,1,0 >() ] - u[ cellIndex ] ) * mesh.template getSpaceStepsProducts< 0, -1, 0 >() * mesh.template getSpaceStepsProducts< 0, -1, 0 >() 
           + ( u[ neighbourEntities.template getEntityIndex< 1,0,0 >() ] - u[ cellIndex ] ) * ( u[ neighbourEntities.template getEntityIndex< 1,0,0 >() ] - u[ cellIndex ] ) 
           * mesh.template getSpaceStepsProducts< -1, 0, 0 >() * mesh.template getSpaceStepsProducts< -1, 0, 0 >() + ( - u[ neighbourEntities.template getEntityIndex< -1,0,0 >() ] + u[ cellIndex ]) 
           * ( - u[ neighbourEntities.template getEntityIndex< -1,0,0 >() ] + u[ cellIndex ]) * mesh.template getSpaceStepsProducts< -1, 0, 0 >() * mesh.template getSpaceStepsProducts< -1, 0, 0 >()
           + ( - u[ neighbourEntities.template getEntityIndex< 0,-1,0 >() ] + u[ cellIndex ]) * 
           ( - u[ neighbourEntities.template getEntityIndex< 0,-1,0 >() ] + u[ cellIndex ]) * mesh.template getSpaceStepsProducts< 0, -1, 0 >() * mesh.template getSpaceStepsProducts< 0, -1, 0 >() 
           + ( u[ neighbourEntities.template getEntityIndex< 0,0,1 >() ] - u[ cellIndex ] ) * ( u[ neighbourEntities.template getEntityIndex< 0,0,1 >() ] - u[ cellIndex ] )
           * mesh.template getSpaceStepsProducts< 0, 0, -1 >() * mesh.template getSpaceStepsProducts< 0, 0, -1 >() + ( - u[ neighbourEntities.template getEntityIndex< 0,0,-1 >() ] + u[ cellIndex ]) * 
           ( - u[ neighbourEntities.template getEntityIndex< 0,0,-1 >() ] + u[ cellIndex ]) * mesh.template getSpaceStepsProducts< 0, 0, -1 >() * mesh.template getSpaceStepsProducts< 0, 0, -1 >()
           ) );
}   
#endif	/* TNLONESIDEDIFFOPERATORQ_IMPL_H */
