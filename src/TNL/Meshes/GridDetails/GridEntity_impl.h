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

/*template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,          typename Config,
          int EntityDimensions >
__cuda_callable__ inline
GridEntity< Meshes::Grid< Dimensions, Real, Device, Index >, EntityDimensions >::
GridEntity()
{
}*/

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          int EntityDimensions,
          typename Config >
__cuda_callable__ inline
GridEntity< Meshes::Grid< Dimensions, Real, Device, Index >, EntityDimensions, Config >::
GridEntity( const Meshes::Grid< Dimensions, Real, Device, Index >& grid )
: grid( grid ),
  entityIndex( -1 ),
  coordinates( 0 ),
  orientation( 0 ),
  basis( 0 ),
  neighbourEntitiesStorage( *this )
{
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          int EntityDimensions,
          typename Config >
__cuda_callable__ inline
GridEntity< Meshes::Grid< Dimensions, Real, Device, Index >, EntityDimensions, Config >::
GridEntity( const Meshes::Grid< Dimensions, Real, Device, Index >& grid,
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

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          int EntityDimensions,
          typename Config >
__cuda_callable__ inline
const typename GridEntity< Meshes::Grid< Dimensions, Real, Device, Index >, EntityDimensions, Config >::CoordinatesType&
GridEntity< Meshes::Grid< Dimensions, Real, Device, Index >, EntityDimensions, Config >::
getCoordinates() const
{
   return this->coordinates;
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          int EntityDimensions,
          typename Config >
__cuda_callable__ inline
typename GridEntity< Meshes::Grid< Dimensions, Real, Device, Index >, EntityDimensions, Config >::CoordinatesType&
GridEntity< Meshes::Grid< Dimensions, Real, Device, Index >, EntityDimensions, Config >::
getCoordinates()
{
   return this->coordinates;
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          int EntityDimensions,
          typename Config >
__cuda_callable__ inline
void
GridEntity< Meshes::Grid< Dimensions, Real, Device, Index >, EntityDimensions, Config >::
setCoordinates( const CoordinatesType& coordinates )
{
   this->coordinates = coordinates;
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          int EntityDimensions,
          typename Config >
__cuda_callable__ inline
void
GridEntity< Meshes::Grid< Dimensions, Real, Device, Index >, EntityDimensions, Config >::
refresh()
{
   this->entityIndex = this->grid.getEntityIndex( *this );
   this->neighbourEntitiesStorage.refresh( this->grid, this->entityIndex );
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          int EntityDimensions,
          typename Config >
__cuda_callable__ inline
Index
GridEntity< Meshes::Grid< Dimensions, Real, Device, Index >, EntityDimensions, Config >::
getIndex() const
{
   typedef Meshes::Grid< Dimensions, Real, Device, Index > GridType;
   typedef typename GridType::template MeshEntity< EntityDimensions > EntityType;
   Assert( this->entityIndex >= 0 &&
              this-> entityIndex < grid.template getEntitiesCount< EntityType >(),
              std::cerr << "this->entityIndex = " << this->entityIndex
                   << " grid.template getEntitiesCount< EntityDimensions >() = " << grid.template getEntitiesCount< EntityType >() );
   Assert( this->entityIndex == grid.getEntityIndex( *this ),
              std::cerr << "this->entityIndex = " << this->entityIndex
                   << " grid.getEntityIndex( *this ) = " << grid.getEntityIndex( *this ) );
   return this->entityIndex;
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          int EntityDimensions,
          typename Config >
__cuda_callable__ inline
const typename GridEntity< Meshes::Grid< Dimensions, Real, Device, Index >, EntityDimensions, Config >::EntityOrientationType&
GridEntity< Meshes::Grid< Dimensions, Real, Device, Index >, EntityDimensions, Config >::
getOrientation() const
{
   return this->orientation;
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          int EntityDimensions,
          typename Config >
__cuda_callable__ inline
void
GridEntity< Meshes::Grid< Dimensions, Real, Device, Index >, EntityDimensions, Config >::
setOrientation( const EntityOrientationType& orientation )
{
   this->orientation = orientation;
   this->basis = EntityBasisType( 1 ) - abs( orientation );
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          int EntityDimensions,
          typename Config >
__cuda_callable__ inline
const typename GridEntity< Meshes::Grid< Dimensions, Real, Device, Index >, EntityDimensions, Config >::EntityBasisType&
GridEntity< Meshes::Grid< Dimensions, Real, Device, Index >, EntityDimensions, Config >::
getBasis() const
{
   return this->basis;
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          int EntityDimensions,
          typename Config >
__cuda_callable__ inline
void
GridEntity< Meshes::Grid< Dimensions, Real, Device, Index >, EntityDimensions, Config >::
setBasis( const EntityBasisType& basis )
{
   this->basis = basis;
   this->orientation = EntityOrientationType( 1 ) - abs( basis );
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          int EntityDimensions,
          typename Config >
   template< int NeighbourEntityDimensions >
__cuda_callable__ inline
const typename GridEntity< Meshes::Grid< Dimensions, Real, Device, Index >, EntityDimensions, Config >::template NeighbourEntities< NeighbourEntityDimensions >&
GridEntity< Meshes::Grid< Dimensions, Real, Device, Index >, EntityDimensions, Config >::
getNeighbourEntities() const
{
   return neighbourEntitiesStorage.template getNeighbourEntities< NeighbourEntityDimensions >();
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          int EntityDimensions,
          typename Config >
__cuda_callable__ inline
bool
GridEntity< Meshes::Grid< Dimensions, Real, Device, Index >, EntityDimensions, Config >::
isBoundaryEntity() const
{
   return BoundaryGridEntityChecker< ThisType >::isBoundaryEntity( *this );
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          int EntityDimensions,
          typename Config >
__cuda_callable__ inline
typename Meshes::Grid< Dimensions, Real, Device, Index >::VertexType
GridEntity< Meshes::Grid< Dimensions, Real, Device, Index >, EntityDimensions, Config >::
getCenter() const
{
   return GridEntityCenterGetter< ThisType >::getEntityCenter( *this );
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          int EntityDimensions,
          typename Config >
__cuda_callable__ inline
const typename GridEntity< Meshes::Grid< Dimensions, Real, Device, Index >, EntityDimensions, Config >::RealType&
GridEntity< Meshes::Grid< Dimensions, Real, Device, Index >, EntityDimensions, Config >::
getMeasure() const
{
   return GridEntityMeasureGetter< GridType, EntityDimensions >::getMeasure( this->getMesh(), *this );
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          int EntityDimensions,
          typename Config >
__cuda_callable__ inline
const typename GridEntity< Meshes::Grid< Dimensions, Real, Device, Index >, EntityDimensions, Config >::GridType&
GridEntity< Meshes::Grid< Dimensions, Real, Device, Index >, EntityDimensions, Config >::
getMesh() const
{
   return this->grid;
}

/****
 * Specialization for cells
 */
/*template< int Dimensions,
          typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline
GridEntity< Meshes::Grid< Dimensions, Real, Device, Index >, Dimensions >::
GridEntity()
{
}*/

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
GridEntity< Meshes::Grid< Dimensions, Real, Device, Index >, Dimensions, Config >::
GridEntity( const GridType& grid )
: grid( grid ),
  entityIndex( -1 ),
  neighbourEntitiesStorage( *this )
{
   this->coordinates = CoordinatesType( ( Index ) 0 );
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
GridEntity< Meshes::Grid< Dimensions, Real, Device, Index >, Dimensions, Config >::
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

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
const typename GridEntity< Meshes::Grid< Dimensions, Real, Device, Index >, Dimensions, Config >::CoordinatesType&
GridEntity< Meshes::Grid< Dimensions, Real, Device, Index >, Dimensions, Config >::
getCoordinates() const
{
   return this->coordinates;
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
typename GridEntity< Meshes::Grid< Dimensions, Real, Device, Index >, Dimensions, Config >::CoordinatesType&
GridEntity< Meshes::Grid< Dimensions, Real, Device, Index >, Dimensions, Config >::
getCoordinates()
{
   return this->coordinates;
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
void
GridEntity< Meshes::Grid< Dimensions, Real, Device, Index >, Dimensions, Config >::
setCoordinates( const CoordinatesType& coordinates )
{
   this->coordinates = coordinates;
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
void
GridEntity< Meshes::Grid< Dimensions, Real, Device, Index >, Dimensions, Config >::
refresh()
{
   this->entityIndex = this->grid.getEntityIndex( *this );
   this->neighbourEntitiesStorage.refresh( this->grid, this->entityIndex );
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
Index
GridEntity< Meshes::Grid< Dimensions, Real, Device, Index >, Dimensions, Config >::
getIndex() const
{
   Assert( this->entityIndex >= 0 &&
              this-> entityIndex < grid.template getEntitiesCount< ThisType >(),
              std::cerr << "this->entityIndex = " << this->entityIndex
                   << " grid.template getEntitiesCount< Dimensions >() = " << grid.template getEntitiesCount< ThisType >() );
   Assert( this->entityIndex == grid.getEntityIndex( *this ),
              std::cerr << "this->index = " << this->entityIndex
                   << " grid.getEntityIndex( *this ) = " << grid.getEntityIndex( *this ) );
   return this->entityIndex;
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
const typename GridEntity< Meshes::Grid< Dimensions, Real, Device, Index >, Dimensions, Config >::EntityOrientationType
GridEntity< Meshes::Grid< Dimensions, Real, Device, Index >, Dimensions, Config >::
getOrientation() const
{
   return EntityOrientationType( ( IndexType ) 0 );
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
const typename GridEntity< Meshes::Grid< Dimensions, Real, Device, Index >, Dimensions, Config >::EntityBasisType
GridEntity< Meshes::Grid< Dimensions, Real, Device, Index >, Dimensions, Config >::
getBasis() const
{
   return EntityBasisType( ( IndexType ) 1 );
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
   template< int NeighbourEntityDimensions >
__cuda_callable__ inline
const typename GridEntity< Meshes::Grid< Dimensions, Real, Device, Index >, Dimensions, Config >::template NeighbourEntities< NeighbourEntityDimensions >&
GridEntity< Meshes::Grid< Dimensions, Real, Device, Index >, Dimensions, Config >::
getNeighbourEntities() const
{
   return neighbourEntitiesStorage.template getNeighbourEntities< NeighbourEntityDimensions >();
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
bool
GridEntity< Meshes::Grid< Dimensions, Real, Device, Index >, Dimensions, Config >::
isBoundaryEntity() const
{
   return BoundaryGridEntityChecker< ThisType >::isBoundaryEntity( *this );
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
typename Meshes::Grid< Dimensions, Real, Device, Index >::VertexType
GridEntity< Meshes::Grid< Dimensions, Real, Device, Index >, Dimensions, Config >::
getCenter() const
{
   return GridEntityCenterGetter< ThisType >::getEntityCenter( *this );
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
const typename GridEntity< Meshes::Grid< Dimensions, Real, Device, Index >, Dimensions, Config >::RealType&
GridEntity< Meshes::Grid< Dimensions, Real, Device, Index >, Dimensions, Config >::
getMeasure() const
{
   return this->getMesh().getCellMeasure();
}


template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
const typename GridEntity< Meshes::Grid< Dimensions, Real, Device, Index >, Dimensions, Config >::VertexType&
GridEntity< Meshes::Grid< Dimensions, Real, Device, Index >, Dimensions, Config >::
getEntityProportions() const
{
   return grid.getSpaceSteps();
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
const typename GridEntity< Meshes::Grid< Dimensions, Real, Device, Index >, Dimensions, Config >::GridType&
GridEntity< Meshes::Grid< Dimensions, Real, Device, Index >, Dimensions, Config >::
getMesh() const
{
   return this->grid;
}


/****
 * Specialization for vertices
 */
template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
GridEntity< Meshes::Grid< Dimensions, Real, Device, Index >, 0, Config >::
GridEntity( const GridType& grid )
 : grid( grid ),
   entityIndex( -1 ),
   coordinates( 0 ),
   neighbourEntitiesStorage( *this )
{
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
GridEntity< Meshes::Grid< Dimensions, Real, Device, Index >, 0, Config >::
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

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
const typename GridEntity< Meshes::Grid< Dimensions, Real, Device, Index >, 0, Config >::CoordinatesType&
GridEntity< Meshes::Grid< Dimensions, Real, Device, Index >, 0, Config >::
getCoordinates() const
{
   return this->coordinates;
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
typename GridEntity< Meshes::Grid< Dimensions, Real, Device, Index >, 0, Config >::CoordinatesType&
GridEntity< Meshes::Grid< Dimensions, Real, Device, Index >, 0, Config >::
getCoordinates()
{
   return this->coordinates;
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
void
GridEntity< Meshes::Grid< Dimensions, Real, Device, Index >, 0, Config >::
setCoordinates( const CoordinatesType& coordinates )
{
   this->coordinates = coordinates;
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
void
GridEntity< Meshes::Grid< Dimensions, Real, Device, Index >, 0, Config >::
refresh()
{
   this->entityIndex = this->grid.getEntityIndex( *this );
   this->neighbourEntitiesStorage.refresh( this->grid, this->entityIndex );
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
Index
GridEntity< Meshes::Grid< Dimensions, Real, Device, Index >, 0, Config >::
getIndex() const
{
   typedef Meshes::Grid< Dimensions, Real, Device, Index > GridType;
   typedef typename GridType::Vertex Vertex;
   Assert( this->entityIndex >= 0 &&
              this-> entityIndex < grid.template getEntitiesCount< Vertex >(),
              std::cerr << "this->entityIndex = " << this->entityIndex
                   << " grid.template getEntitiesCount< 0 >() = " << grid.template getEntitiesCount< Vertex >() );
   Assert( this->entityIndex == grid.getEntityIndex( *this ),
              std::cerr << "this->entityIndex = " << this->entityIndex
                   << " grid.getEntityIndex( *this ) = " << grid.getEntityIndex( *this ) );
   return this->entityIndex;
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
const typename GridEntity< Meshes::Grid< Dimensions, Real, Device, Index >, 0, Config >::EntityOrientationType
GridEntity< Meshes::Grid< Dimensions, Real, Device, Index >, 0, Config >::
getOrientation() const
{
   return EntityOrientationType( ( IndexType ) 0 );
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
const typename GridEntity< Meshes::Grid< Dimensions, Real, Device, Index >, 0, Config >::EntityBasisType
GridEntity< Meshes::Grid< Dimensions, Real, Device, Index >, 0, Config >::
getBasis() const
{
   return EntityBasisType( ( IndexType ) 0 );
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
   template< int NeighbourEntityDimensions >
__cuda_callable__ inline
const typename GridEntity< Meshes::Grid< Dimensions, Real, Device, Index >, 0, Config >::template NeighbourEntities< NeighbourEntityDimensions >&
GridEntity< Meshes::Grid< Dimensions, Real, Device, Index >, 0, Config >::
getNeighbourEntities() const
{
   return neighbourEntitiesStorage.template getNeighbourEntities< NeighbourEntityDimensions >();
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
bool
GridEntity< Meshes::Grid< Dimensions, Real, Device, Index >, 0, Config >::
isBoundaryEntity() const
{
   return BoundaryGridEntityChecker< ThisType >::isBoundaryEntity( *this );
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
typename Meshes::Grid< Dimensions, Real, Device, Index >::VertexType
GridEntity< Meshes::Grid< Dimensions, Real, Device, Index >, 0, Config >::
getCenter() const
{
   return GridEntityCenterGetter< ThisType >::getEntityCenter( *this );
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
const typename GridEntity< Meshes::Grid< Dimensions, Real, Device, Index >, 0, Config >::RealType
GridEntity< Meshes::Grid< Dimensions, Real, Device, Index >, 0, Config >::
getMeasure() const
{
   return 0.0;
}


template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
typename GridEntity< Meshes::Grid< Dimensions, Real, Device, Index >, 0, Config >::VertexType
GridEntity< Meshes::Grid< Dimensions, Real, Device, Index >, 0, Config >::
getEntityProportions() const
{
   return VertexType( 0.0 );
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
const typename GridEntity< Meshes::Grid< Dimensions, Real, Device, Index >, 0, Config >::GridType&
GridEntity< Meshes::Grid< Dimensions, Real, Device, Index >, 0, Config >::
getMesh() const
{
   return this->grid;
}

} // namespace Meshes
} // namespace TNL

