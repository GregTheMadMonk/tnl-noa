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

#include "tnlGridEntity.h"


template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          int EntityDimensions >
__cuda_callable__ inline
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, EntityDimensions >::
tnlGridEntity()
{
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          int EntityDimensions >
__cuda_callable__ inline
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, EntityDimensions >::
tnlGridEntity( const tnlGrid< Dimensions, Real, Device, Index >& grid )
: grid( grid ),
  entityIndex( -1 ),
  coordinates( 0 ),
  orientation( 0 ),
  basis( 0 ) 
{
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          int EntityDimensions >
__cuda_callable__ inline
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, EntityDimensions >::
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
          int EntityDimensions >
__cuda_callable__ inline
const typename tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, EntityDimensions >::CoordinatesType& 
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, EntityDimensions >::
getCoordinates() const
{
   return this->coordinates;
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          int EntityDimensions >
__cuda_callable__ inline
typename tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, EntityDimensions >::CoordinatesType& 
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, EntityDimensions >::
getCoordinates()
{
   return this->coordinates;
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          int EntityDimensions >
__cuda_callable__ inline
void
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, EntityDimensions >::
setCoordinates( const CoordinatesType& coordinates )
{
   this->coordinates = coordinates;
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          int EntityDimensions >
__cuda_callable__ inline
void
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, EntityDimensions >::
setIndex( IndexType entityIndex )
{
   tnlAssert( entityIndex >= 0 &&
              entityIndex < grid.template getEntitiesCount< EntityDimensions >(),
              cerr << "entityIndex = " << entityIndex
                   << " grid.template getEntitiesCount< EntityDimensions >() = " << grid.template getEntitiesCount< EntityDimensions >() );
   this->entityIndex = entityIndex;                   
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          int EntityDimensions >
__cuda_callable__ inline
Index
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, EntityDimensions >::
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
          int EntityDimensions >
__cuda_callable__ inline
const typename tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, EntityDimensions >::EntityOrientationType& 
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, EntityDimensions >::
getOrientation() const
{
   return this->orientation;
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          int EntityDimensions >
__cuda_callable__ inline
void 
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, EntityDimensions >::
setOrientation( const EntityOrientationType& orientation )
{
   this->orientation = orientation;
   this->basis = EntityBasisType( 1 ) - tnlAbs( orientation );
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          int EntityDimensions >
__cuda_callable__ inline
const typename tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, EntityDimensions >::EntityBasisType& 
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, EntityDimensions >::
getBasis() const
{
   return this->basis;
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          int EntityDimensions >
__cuda_callable__ inline
void 
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, EntityDimensions >::
setBasis( const EntityBasisType& basis )
{
   this->basis = basis;
   this->orientation = EntityOrientationType( 1 ) - tnlAbs( basis );
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          int EntityDimensions >
   template< int NeighbourEntityDimensions >
__cuda_callable__ inline
tnlNeighbourGridEntityGetter< 
   tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, EntityDimensions >, 
   NeighbourEntityDimensions >
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, EntityDimensions >::
getNeighbourEntities() const
{
   return tnlNeighbourGridEntityGetter< ThisType, NeighbourEntityDimensions >( this->grid, *this );
}


/****
 * Specialization for cells
 */
template< int Dimensions,
          typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, Dimensions >::
tnlGridEntity()
{
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, Dimensions >::
tnlGridEntity( const GridType& grid )
: grid( grid ),
  entityIndex( -1 )
{
   this->coordinates = CoordinatesType( ( Index ) 0 );
   this->orientation = EntityOrientationType( ( Index ) 0 );
   this->basis = EntityBasisType( ( Index ) 1 );
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, Dimensions >::
tnlGridEntity( const GridType& grid,
               const CoordinatesType& coordinates,
               const EntityOrientationType& orientation,
               const EntityBasisType& basis )
: grid( grid ),
  entityIndex( -1 ),
  coordinates( coordinates )
{
   this->orientation = EntityOrientationType( ( Index ) 0 );
   this->basis = EntityBasisType( ( Index ) 1 );
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline
const typename tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, Dimensions >::CoordinatesType& 
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, Dimensions >::
getCoordinates() const
{
   return this->coordinates;
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline
typename tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, Dimensions >::CoordinatesType& 
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, Dimensions >::
getCoordinates()
{
   return this->coordinates;
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline
void
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, Dimensions >::
setCoordinates( const CoordinatesType& coordinates )
{
   this->coordinates = coordinates;
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline
void
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, Dimensions >::
setIndex( IndexType entityIndex )
{
   tnlAssert( entityIndex >= 0 &&
              entityIndex < grid.template getEntitiesCount< Dimensions >(),
              cerr << "entityIndex = " << entityIndex
                   << " grid.template getEntitiesCount< Dimensions >() = " << grid.template getEntitiesCount< Dimensions >() );
   this->entityIndex = entityIndex;                   
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline
Index
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, Dimensions >::
getIndex() const
{
   tnlAssert( this->entityIndex >= 0 &&
              this-> entityIndex < grid.template getEntitiesCount< Dimensions >(),
              cerr << "this->entityIndex = " << this->entityIndex
                   << " grid.template getEntitiesCount< Dimensions >() = " << grid.template getEntitiesCount< Dimensions >() );
   tnlAssert( this->entityIndex == grid.getEntityIndex( *this ),
              cerr << "this->index = " << this->entityIndex
                   << " grid.getEntityIndex( *this ) = " << grid.getEntityIndex( *this ) );
   return this->entityIndex;
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline
const typename tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, Dimensions >::EntityOrientationType 
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, Dimensions >::
getOrientation() const
{
   return EntityOrientationType( ( IndexType ) 0 );
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline
const typename tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, Dimensions >::EntityBasisType 
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, Dimensions >::
getBasis() const
{
   return EntityBasisType( ( IndexType ) 1 );
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index >
   template< int NeighbourEntityDimensions >
__cuda_callable__ inline
tnlNeighbourGridEntityGetter< 
   tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, Dimensions >, 
   NeighbourEntityDimensions >
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, Dimensions >::
getNeighbourEntities() const
{
   return tnlNeighbourGridEntityGetter< ThisType, NeighbourEntityDimensions >( this->grid, *this );
}

/****
 * Specialization for vertices
 */
template< int Dimensions,
          typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, 0 >::
tnlGridEntity()
{
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, 0 >::
tnlGridEntity( const GridType& grid )
 : grid( grid ),
   entityIndex( -1 ),
   coordinates( 0 ),
   orientation( 1 ),
   basis( 0 )
{
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, 0 >::
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
          typename Index >
__cuda_callable__ inline
const typename tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, 0 >::CoordinatesType& 
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, 0 >::
getCoordinates() const
{
   return this->coordinates;
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline
typename tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, 0 >::CoordinatesType& 
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, 0 >::
getCoordinates()
{
   return this->coordinates;
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline
void
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, 0 >::
setCoordinates( const CoordinatesType& coordinates )
{
   this->coordinates = coordinates;
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline
void
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, 0 >::
setIndex( IndexType entityIndex ) const
{
   tnlAssert( entityIndex >= 0 &&
              entityIndex < grid.template getEntitiesCount< 0 >(),
              cerr << "entityIndex = " << entityIndex
                   << " grid.template getEntitiesCount< 0 >() = " << grid.template getEntitiesCount< 0 >() );
   this->entityIndex = entityIndex;                   
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline
Index
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, 0 >::
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
          typename Index >
__cuda_callable__ inline
const typename tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, 0 >::EntityOrientationType
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, 0 >::
getOrientation() const
{
   return EntityOrientationType( ( IndexType ) 0 );
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline
const typename tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, 0 >::EntityBasisType
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, 0 >::
getBasis() const
{
   return EntityBasisType( ( IndexType ) 0 );
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index >
   template< int NeighbourEntityDimensions >
__cuda_callable__ inline
tnlNeighbourGridEntityGetter< 
   tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, 0 >, 
   NeighbourEntityDimensions >
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, 0 >::
getNeighbourEntities() const
{
   return tnlNeighbourGridEntityGetter< ThisType, NeighbourEntityDimensions >( this->grid, *this );
}

#endif	/* TNLGRIDENTITY_IMPL_H */

