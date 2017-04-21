/***************************************************************************
                          GridEntity_impl.h  -  description
                             -------------------
    begin                : Nov 20, 2015
    copyright            : (C) 2015 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */


#pragma once

#include <TNL/Meshes/GridDetails/BoundaryGridEntityChecker.h>
#include <TNL/Meshes/GridDetails/GridEntityCenterGetter.h>
#include <TNL/Meshes/GridDetails/GridEntityMeasureGetter.h>
#include <TNL/Meshes/GridEntity.h>

namespace TNL {
namespace Meshes {

/*template< int Dimension,
          typename Real,
          typename Device,
          typename Index,          typename Config,
          int EntityDimension >
__cuda_callable__ inline
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, EntityDimension >::
GridEntity()
{
}*/

template< int Dimension,
          typename Real,
          typename Device,
          typename Index,
          int EntityDimension,
          typename Config >
__cuda_callable__ inline
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, EntityDimension, Config >::
GridEntity( const Meshes::Grid< Dimension, Real, Device, Index >& grid )
: grid( grid ),
  entityIndex( -1 ),
  coordinates( 0 ),
  orientation( 0 ),
  basis( 0 ),
  neighbourEntitiesStorage( *this )
{
}

template< int Dimension,
          typename Real,
          typename Device,
          typename Index,
          int EntityDimension,
          typename Config >
__cuda_callable__ inline
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, EntityDimension, Config >::
GridEntity( const Meshes::Grid< Dimension, Real, Device, Index >& grid,
               const CoordinatesType& coordinates,
               const EntityOrientationType& orientation,
               const EntityBasisType& basis )
: grid( grid ),
  entityIndex( -1 ),
  coordinates( coordinates ),
  orientation( orientation ),
  basis( basis ),
  neighbourEntitiesStorage( *this )
{
}

template< int Dimension,
          typename Real,
          typename Device,
          typename Index,
          int EntityDimension,
          typename Config >
__cuda_callable__ inline
const typename GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, EntityDimension, Config >::CoordinatesType&
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, EntityDimension, Config >::
getCoordinates() const
{
   return this->coordinates;
}

template< int Dimension,
          typename Real,
          typename Device,
          typename Index,
          int EntityDimension,
          typename Config >
__cuda_callable__ inline
typename GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, EntityDimension, Config >::CoordinatesType&
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, EntityDimension, Config >::
getCoordinates()
{
   return this->coordinates;
}

template< int Dimension,
          typename Real,
          typename Device,
          typename Index,
          int EntityDimension,
          typename Config >
__cuda_callable__ inline
void
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, EntityDimension, Config >::
setCoordinates( const CoordinatesType& coordinates )
{
   this->coordinates = coordinates;
}

template< int Dimension,
          typename Real,
          typename Device,
          typename Index,
          int EntityDimension,
          typename Config >
__cuda_callable__ inline
void
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, EntityDimension, Config >::
refresh()
{
   this->entityIndex = this->grid.getEntityIndex( *this );
   this->neighbourEntitiesStorage.refresh( this->grid, this->entityIndex );
}

template< int Dimension,
          typename Real,
          typename Device,
          typename Index,
          int EntityDimension,
          typename Config >
__cuda_callable__ inline
Index
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, EntityDimension, Config >::
getIndex() const
{
   typedef Meshes::Grid< Dimension, Real, Device, Index > GridType;
   typedef typename GridType::template MeshEntity< EntityDimension > EntityType;
   TNL_ASSERT( this->entityIndex >= 0 &&
              this-> entityIndex < grid.template getEntitiesCount< EntityType >(),
              std::cerr << "this->entityIndex = " << this->entityIndex
                   << " grid.template getEntitiesCount< EntityDimension >() = " << grid.template getEntitiesCount< EntityType >() );
   TNL_ASSERT( this->entityIndex == grid.getEntityIndex( *this ),
              std::cerr << "this->entityIndex = " << this->entityIndex
                   << " grid.getEntityIndex( *this ) = " << grid.getEntityIndex( *this ) );
   return this->entityIndex;
}

template< int Dimension,
          typename Real,
          typename Device,
          typename Index,
          int EntityDimension,
          typename Config >
__cuda_callable__ inline
const typename GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, EntityDimension, Config >::EntityOrientationType&
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, EntityDimension, Config >::
getOrientation() const
{
   return this->orientation;
}

template< int Dimension,
          typename Real,
          typename Device,
          typename Index,
          int EntityDimension,
          typename Config >
__cuda_callable__ inline
void
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, EntityDimension, Config >::
setOrientation( const EntityOrientationType& orientation )
{
   this->orientation = orientation;
   this->basis = EntityBasisType( 1 ) - abs( orientation );
}

template< int Dimension,
          typename Real,
          typename Device,
          typename Index,
          int EntityDimension,
          typename Config >
__cuda_callable__ inline
const typename GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, EntityDimension, Config >::EntityBasisType&
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, EntityDimension, Config >::
getBasis() const
{
   return this->basis;
}

template< int Dimension,
          typename Real,
          typename Device,
          typename Index,
          int EntityDimension,
          typename Config >
__cuda_callable__ inline
void
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, EntityDimension, Config >::
setBasis( const EntityBasisType& basis )
{
   this->basis = basis;
   this->orientation = EntityOrientationType( 1 ) - abs( basis );
}

template< int Dimension,
          typename Real,
          typename Device,
          typename Index,
          int EntityDimension,
          typename Config >
   template< int NeighbourEntityDimension >
__cuda_callable__ inline
const typename GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, EntityDimension, Config >::template NeighbourEntities< NeighbourEntityDimension >&
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, EntityDimension, Config >::
getNeighbourEntities() const
{
   return neighbourEntitiesStorage.template getNeighbourEntities< NeighbourEntityDimension >();
}

template< int Dimension,
          typename Real,
          typename Device,
          typename Index,
          int EntityDimension,
          typename Config >
__cuda_callable__ inline
bool
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, EntityDimension, Config >::
isBoundaryEntity() const
{
   return BoundaryGridEntityChecker< ThisType >::isBoundaryEntity( *this );
}

template< int Dimension,
          typename Real,
          typename Device,
          typename Index,
          int EntityDimension,
          typename Config >
__cuda_callable__ inline
typename Meshes::Grid< Dimension, Real, Device, Index >::VertexType
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, EntityDimension, Config >::
getCenter() const
{
   return GridEntityCenterGetter< ThisType >::getEntityCenter( *this );
}

template< int Dimension,
          typename Real,
          typename Device,
          typename Index,
          int EntityDimension,
          typename Config >
__cuda_callable__ inline
const typename GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, EntityDimension, Config >::RealType&
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, EntityDimension, Config >::
getMeasure() const
{
   return GridEntityMeasureGetter< GridType, EntityDimension >::getMeasure( this->getMesh(), *this );
}

template< int Dimension,
          typename Real,
          typename Device,
          typename Index,
          int EntityDimension,
          typename Config >
__cuda_callable__ inline
const typename GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, EntityDimension, Config >::GridType&
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, EntityDimension, Config >::
getMesh() const
{
   return this->grid;
}

/****
 * Specialization for cells
 */
/*template< int Dimension,
          typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, Dimension >::
GridEntity()
{
}*/

template< int Dimension,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, Dimension, Config >::
GridEntity( const GridType& grid )
: grid( grid ),
  entityIndex( -1 ),
  neighbourEntitiesStorage( *this )
{
   this->coordinates = CoordinatesType( ( Index ) 0 );
}

template< int Dimension,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, Dimension, Config >::
GridEntity( const GridType& grid,
               const CoordinatesType& coordinates,
               const EntityOrientationType& orientation,
               const EntityBasisType& basis )
: grid( grid ),
  entityIndex( -1 ),
  coordinates( coordinates ),
  neighbourEntitiesStorage( *this )
{
}

template< int Dimension,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
const typename GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, Dimension, Config >::CoordinatesType&
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, Dimension, Config >::
getCoordinates() const
{
   return this->coordinates;
}

template< int Dimension,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
typename GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, Dimension, Config >::CoordinatesType&
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, Dimension, Config >::
getCoordinates()
{
   return this->coordinates;
}

template< int Dimension,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
void
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, Dimension, Config >::
setCoordinates( const CoordinatesType& coordinates )
{
   this->coordinates = coordinates;
}

template< int Dimension,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
void
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, Dimension, Config >::
refresh()
{
   this->entityIndex = this->grid.getEntityIndex( *this );
   this->neighbourEntitiesStorage.refresh( this->grid, this->entityIndex );
}

template< int Dimension,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
Index
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, Dimension, Config >::
getIndex() const
{
   TNL_ASSERT( this->entityIndex >= 0 &&
              this-> entityIndex < grid.template getEntitiesCount< ThisType >(),
              std::cerr << "this->entityIndex = " << this->entityIndex
                   << " grid.template getEntitiesCount< Dimension >() = " << grid.template getEntitiesCount< ThisType >() );
   TNL_ASSERT( this->entityIndex == grid.getEntityIndex( *this ),
              std::cerr << "this->index = " << this->entityIndex
                   << " grid.getEntityIndex( *this ) = " << grid.getEntityIndex( *this ) );
   return this->entityIndex;
}

template< int Dimension,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
const typename GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, Dimension, Config >::EntityOrientationType
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, Dimension, Config >::
getOrientation() const
{
   return EntityOrientationType( ( IndexType ) 0 );
}

template< int Dimension,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
const typename GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, Dimension, Config >::EntityBasisType
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, Dimension, Config >::
getBasis() const
{
   return EntityBasisType( ( IndexType ) 1 );
}

template< int Dimension,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
   template< int NeighbourEntityDimension >
__cuda_callable__ inline
const typename GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, Dimension, Config >::template NeighbourEntities< NeighbourEntityDimension >&
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, Dimension, Config >::
getNeighbourEntities() const
{
   return neighbourEntitiesStorage.template getNeighbourEntities< NeighbourEntityDimension >();
}

template< int Dimension,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
bool
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, Dimension, Config >::
isBoundaryEntity() const
{
   return BoundaryGridEntityChecker< ThisType >::isBoundaryEntity( *this );
}

template< int Dimension,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
typename Meshes::Grid< Dimension, Real, Device, Index >::VertexType
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, Dimension, Config >::
getCenter() const
{
   return GridEntityCenterGetter< ThisType >::getEntityCenter( *this );
}

template< int Dimension,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
const typename GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, Dimension, Config >::RealType&
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, Dimension, Config >::
getMeasure() const
{
   return this->getMesh().getCellMeasure();
}


template< int Dimension,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
const typename GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, Dimension, Config >::VertexType&
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, Dimension, Config >::
getEntityProportions() const
{
   return grid.getSpaceSteps();
}

template< int Dimension,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
const typename GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, Dimension, Config >::GridType&
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, Dimension, Config >::
getMesh() const
{
   return this->grid;
}


/****
 * Specialization for vertices
 */
template< int Dimension,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, 0, Config >::
GridEntity( const GridType& grid )
 : grid( grid ),
   entityIndex( -1 ),
   coordinates( 0 ),
   neighbourEntitiesStorage( *this )
{
}

template< int Dimension,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, 0, Config >::
GridEntity( const GridType& grid,
               const CoordinatesType& coordinates,
               const EntityOrientationType& orientation,
               const EntityBasisType& basis )
: grid( grid ),
  entityIndex( -1 ),
  coordinates( coordinates ),
  neighbourEntitiesStorage( *this )
{
}

template< int Dimension,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
const typename GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, 0, Config >::CoordinatesType&
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, 0, Config >::
getCoordinates() const
{
   return this->coordinates;
}

template< int Dimension,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
typename GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, 0, Config >::CoordinatesType&
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, 0, Config >::
getCoordinates()
{
   return this->coordinates;
}

template< int Dimension,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
void
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, 0, Config >::
setCoordinates( const CoordinatesType& coordinates )
{
   this->coordinates = coordinates;
}

template< int Dimension,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
void
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, 0, Config >::
refresh()
{
   this->entityIndex = this->grid.getEntityIndex( *this );
   this->neighbourEntitiesStorage.refresh( this->grid, this->entityIndex );
}

template< int Dimension,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
Index
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, 0, Config >::
getIndex() const
{
   typedef Meshes::Grid< Dimension, Real, Device, Index > GridType;
   typedef typename GridType::Vertex Vertex;
   TNL_ASSERT( this->entityIndex >= 0 &&
              this-> entityIndex < grid.template getEntitiesCount< Vertex >(),
              std::cerr << "this->entityIndex = " << this->entityIndex
                   << " grid.template getEntitiesCount< 0 >() = " << grid.template getEntitiesCount< Vertex >() );
   TNL_ASSERT( this->entityIndex == grid.getEntityIndex( *this ),
              std::cerr << "this->entityIndex = " << this->entityIndex
                   << " grid.getEntityIndex( *this ) = " << grid.getEntityIndex( *this ) );
   return this->entityIndex;
}

template< int Dimension,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
const typename GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, 0, Config >::EntityOrientationType
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, 0, Config >::
getOrientation() const
{
   return EntityOrientationType( ( IndexType ) 0 );
}

template< int Dimension,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
const typename GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, 0, Config >::EntityBasisType
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, 0, Config >::
getBasis() const
{
   return EntityBasisType( ( IndexType ) 0 );
}

template< int Dimension,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
   template< int NeighbourEntityDimension >
__cuda_callable__ inline
const typename GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, 0, Config >::template NeighbourEntities< NeighbourEntityDimension >&
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, 0, Config >::
getNeighbourEntities() const
{
   return neighbourEntitiesStorage.template getNeighbourEntities< NeighbourEntityDimension >();
}

template< int Dimension,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
bool
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, 0, Config >::
isBoundaryEntity() const
{
   return BoundaryGridEntityChecker< ThisType >::isBoundaryEntity( *this );
}

template< int Dimension,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
typename Meshes::Grid< Dimension, Real, Device, Index >::VertexType
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, 0, Config >::
getCenter() const
{
   return GridEntityCenterGetter< ThisType >::getEntityCenter( *this );
}

template< int Dimension,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
const typename GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, 0, Config >::RealType
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, 0, Config >::
getMeasure() const
{
   return 0.0;
}


template< int Dimension,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
typename GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, 0, Config >::VertexType
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, 0, Config >::
getEntityProportions() const
{
   return VertexType( 0.0 );
}

template< int Dimension,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
const typename GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, 0, Config >::GridType&
GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, 0, Config >::
getMesh() const
{
   return this->grid;
}

} // namespace Meshes
} // namespace TNL

