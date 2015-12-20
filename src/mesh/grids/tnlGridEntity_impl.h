/***************************************************************************
                          tnlGridEntity_impl.h  -  description
                             -------------------
    begin                : Nov 20, 2015
    copyright            : (C) 2015 by Tomas Oberhuber
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


#ifndef TNLGRIDENTITY_IMPL_H
#define TNLGRIDENTITY_IMPL_H

#include <mesh/grids/tnlBoundaryGridEntityChecker.h>
#include <mesh/grids/tnlGridEntityCenterGetter.h>
#include <mesh/grids/tnlGridEntity.h>


/*template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,          typename Config,
          int EntityDimensions >
__cuda_callable__ inline
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, EntityDimensions >::
tnlGridEntity()
{
}*/

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          int EntityDimensions,
          typename Config >
__cuda_callable__ inline
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, EntityDimensions, Config >::
tnlGridEntity( const tnlGrid< Dimensions, Real, Device, Index >& grid )
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
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, EntityDimensions, Config >::
tnlGridEntity( const tnlGrid< Dimensions, Real, Device, Index >& grid,
               const CoordinatesType& coordinates,
               const EntityOrientationType& orientation,
               const EntityBasisType& basis )
: grid( grid ),
  entityIndex( -1 ),
  coordinates( coordinates ),
  orientation( orientation ),
  basis( basis )
{  
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          int EntityDimensions,
          typename Config >
__cuda_callable__ inline
const typename tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, EntityDimensions, Config >::CoordinatesType& 
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, EntityDimensions, Config >::
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
typename tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, EntityDimensions, Config >::CoordinatesType& 
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, EntityDimensions, Config >::
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
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, EntityDimensions, Config >::
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
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, EntityDimensions, Config >::
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
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, EntityDimensions, Config >::
getIndex() const
{
   tnlAssert( this->entityIndex >= 0 &&
              this-> entityIndex < grid.template getEntitiesCount< EntityDimensions >(),
              cerr << "this->entityIndex = " << this->entityIndex
                   << " grid.template getEntitiesCount< EntityDimensions >() = " << grid.template getEntitiesCount< EntityDimensions >() );
   tnlAssert( this->entityIndex == grid.getEntityIndex( *this ),
              cerr << "this->entityIndex = " << this->entityIndex
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
const typename tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, EntityDimensions, Config >::EntityOrientationType& 
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, EntityDimensions, Config >::
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
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, EntityDimensions, Config >::
setOrientation( const EntityOrientationType& orientation )
{
   this->orientation = orientation;
   this->basis = EntityBasisType( 1 ) - tnlAbs( orientation );
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          int EntityDimensions,
          typename Config >
__cuda_callable__ inline
const typename tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, EntityDimensions, Config >::EntityBasisType& 
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, EntityDimensions, Config >::
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
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, EntityDimensions, Config >::
setBasis( const EntityBasisType& basis )
{
   this->basis = basis;
   this->orientation = EntityOrientationType( 1 ) - tnlAbs( basis );
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          int EntityDimensions,
          typename Config >
   template< int NeighbourEntityDimensions >
__cuda_callable__ inline
tnlNeighbourGridEntityGetter< 
   tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, EntityDimensions, Config >, 
   NeighbourEntityDimensions >
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, EntityDimensions, Config >::
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
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, EntityDimensions, Config >::
isBoundaryEntity() const
{
   return tnlBoundaryGridEntityChecker< ThisType >::isBoundaryEntity( *this );
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          int EntityDimensions,
          typename Config >
__cuda_callable__ inline
typename tnlGrid< Dimensions, Real, Device, Index >::VertexType
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, EntityDimensions, Config >::
getCenter() const
{
   return tnlGridEntityCenterGetter< ThisType >::getEntityCenter( *this );
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          int EntityDimensions,
          typename Config >
__cuda_callable__ inline
const typename tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, EntityDimensions, Config >::GridType&
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, EntityDimensions, Config >::
getGrid() const
{
   return this->gridType;
}

/****
 * Specialization for cells
 */
/*template< int Dimensions,
          typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, Dimensions >::
tnlGridEntity()
{
}*/

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, Dimensions, Config >::
tnlGridEntity( const GridType& grid )
: grid( grid ),
  entityIndex( -1 ),
  neighbourEntitiesStorage( *this )
{
   this->coordinates = CoordinatesType( ( Index ) 0 );
   this->orientation = EntityOrientationType( ( Index ) 0 );
   this->basis = EntityBasisType( ( Index ) 1 );
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, Dimensions, Config >::
tnlGridEntity( const GridType& grid,
               const CoordinatesType& coordinates,
               const EntityOrientationType& orientation,
               const EntityBasisType& basis )
: grid( grid ),
  entityIndex( -1 ),
  coordinates( coordinates ),
  neighbourEntitiesStorage( *this )
{
   this->orientation = EntityOrientationType( ( Index ) 0 );
   this->basis = EntityBasisType( ( Index ) 1 );
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
const typename tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, Dimensions, Config >::CoordinatesType& 
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, Dimensions, Config >::
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
typename tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, Dimensions, Config >::CoordinatesType& 
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, Dimensions, Config >::
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
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, Dimensions, Config >::
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
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, Dimensions, Config >::
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
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, Dimensions, Config >::
getIndex() const
{
   tnlAssert( this->entityIndex >= 0 &&
              this-> entityIndex < grid.template getEntitiesCount< ThisType >(),
              cerr << "this->entityIndex = " << this->entityIndex
                   << " grid.template getEntitiesCount< Dimensions >() = " << grid.template getEntitiesCount< ThisType >() );
   tnlAssert( this->entityIndex == grid.getEntityIndex( *this ),
              cerr << "this->index = " << this->entityIndex
                   << " grid.getEntityIndex( *this ) = " << grid.getEntityIndex( *this ) );
   return this->entityIndex;
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
const typename tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, Dimensions, Config >::EntityOrientationType 
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, Dimensions, Config >::
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
const typename tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, Dimensions, Config >::EntityBasisType 
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, Dimensions, Config >::
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
tnlNeighbourGridEntityGetter< 
   tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, Dimensions, Config >, 
   NeighbourEntityDimensions >
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, Dimensions, Config >::
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
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, Dimensions, Config >::
isBoundaryEntity() const
{
   return tnlBoundaryGridEntityChecker< ThisType >::isBoundaryEntity( *this );
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
typename tnlGrid< Dimensions, Real, Device, Index >::VertexType
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, Dimensions, Config >::
getCenter() const
{
   return tnlGridEntityCenterGetter< ThisType >::getEntityCenter( *this );
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
const typename tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, Dimensions, Config >::VertexType&
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, Dimensions, Config >::
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
const typename tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, Dimensions, Config >::GridType&
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, Dimensions, Config >::
getGrid() const
{
   return this->grid;
}


/****
 * Specialization for vertices
 */
/*template< int Dimensions,
          typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, 0 >::
tnlGridEntity()
{
}*/

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, 0, Config >::
tnlGridEntity( const GridType& grid )
 : grid( grid ),
   entityIndex( -1 ),
   coordinates( 0 ),
   orientation( 1 ),
   basis( 0 ),
   neighbourEntitiesStorage( *this )
{
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, 0, Config >::
tnlGridEntity( const GridType& grid,
               const CoordinatesType& coordinates,
               const EntityOrientationType& orientation,
               const EntityBasisType& basis )
: grid( grid ),
  entityIndex( -1 ),
  coordinates( coordinates )
{  
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
const typename tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, 0, Config >::CoordinatesType& 
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, 0, Config >::
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
typename tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, 0, Config >::CoordinatesType& 
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, 0, Config >::
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
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, 0, Config >::
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
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, 0, Config >::
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
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, 0, Config >::
getIndex() const
{
   tnlAssert( this->entityIndex >= 0 &&
              this-> entityIndex < grid.template getEntitiesCount< 0 >(),
              cerr << "this->entityIndex = " << this->entityIndex
                   << " grid.template getEntitiesCount< 0 >() = " << grid.template getEntitiesCount< 0 >() );
   tnlAssert( this->entityIndex == grid.getEntityIndex( *this ),
              cerr << "this->entityIndex = " << this->entityIndex
                   << " grid.getEntityIndex( *this ) = " << grid.getEntityIndex( *this ) );
   return this->entityIndex;
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
const typename tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, 0, Config >::EntityOrientationType
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, 0, Config >::
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
const typename tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, 0, Config >::EntityBasisType
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, 0, Config >::
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
tnlNeighbourGridEntityGetter< 
   tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, 0, Config >, 
   NeighbourEntityDimensions >
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, 0, Config >::
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
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, 0, Config >::
isBoundaryEntity() const
{
   return tnlBoundaryGridEntityChecker< ThisType >::isBoundaryEntity( *this );
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
typename tnlGrid< Dimensions, Real, Device, Index >::VertexType
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, 0, Config >::
getCenter() const
{
   return tnlGridEntityCenterGetter< ThisType >::getEntityCenter( *this );
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
__cuda_callable__ inline
typename tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, 0, Config >::VertexType
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, 0, Config >::
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
const typename tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, 0, Config >::GridType&
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, 0, Config >::
getGrid() const
{
   return this->gridType;
}

#endif	/* TNLGRIDENTITY_IMPL_H */

