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


template< typename Grid,
          int EntityDimensions >
tnlGridEntity< Grid, EntityDimensions >::
tnlGridEntity()
: coordinates( 0 ),
  orientation( 0 ),
  basis( 0 ) 
{
}

template< typename Grid,
          int EntityDimensions >
tnlGridEntity< Grid, EntityDimensions >::
tnlGridEntity( const CoordinatesType& coordinates,
               const EntityOrientationType& orientation,
               const EntityBasisType& basis )
: coordinates( coordinates ),
  orientation( orientation ),
  basis( basis )
{  
}

template< typename Grid,
          int EntityDimensions >
const typename tnlGridEntity< Grid, EntityDimensions >::CoordinatesType& 
tnlGridEntity< Grid, EntityDimensions >::
getCoordinates() const
{
   return this->coordinates;
}

template< typename Grid,
          int EntityDimensions >
typename tnlGridEntity< Grid, EntityDimensions >::CoordinatesType& 
tnlGridEntity< Grid, EntityDimensions >::
getCoordinates()
{
   return this->coordinates;
}

template< typename Grid,
          int EntityDimensions >
const typename tnlGridEntity< Grid, EntityDimensions >::EntityOrientationType& 
tnlGridEntity< Grid, EntityDimensions >::
getOrientation() const
{
   return this->orientation;
}

template< typename Grid,
          int EntityDimensions >
void 
tnlGridEntity< Grid, EntityDimensions >::
setOrientation( const EntityOrientationType& orientation )
{
   this->orientation = orientation;
   this->basis = EntityBasisType( 1 ) - tnlAbs( orientation );
}

template< typename Grid,
          int EntityDimensions >
const typename tnlGridEntity< Grid, EntityDimensions >::EntityBasisType& 
tnlGridEntity< Grid, EntityDimensions >::
getBasis() const
{
   return this->basis;
   this->orientation = EntityOrientationType( 1 ) - tnlAbs( basis );
}

template< typename Grid,
          int EntityDimensions >
void 
tnlGridEntity< Grid, EntityDimensions >::
setBasis( const EntityBasisType& basis )
{
   this->basis = basis;
}

/****
 * Specialization for cells
 */
template< int Dimensions,
          typename Real,
          typename Device,
          typename Index >
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, Dimensions >::
tnlGridEntity()
 : coordinates( 0 ),
   orientation( 0 ),
   basis( 1 )
{
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index >
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, Dimensions >::
tnlGridEntity( const CoordinatesType& coordinates )
: coordinates( coordinates )
{  
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index >
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
const typename tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, Dimensions >::EntityOrientationType& 
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, Dimensions >::
getOrientation() const
{
   return this->orientation;
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index >
const typename tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, Dimensions >::EntityBasisType& 
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, Dimensions >::
getBasis() const
{
   return this->basis;
}

/****
 * Specialization for vertices
 */
template< int Dimensions,
          typename Real,
          typename Device,
          typename Index >
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, 0 >::
tnlGridEntity()
 : coordinates( 0 ),
   orientation( 1 ),
   basis( 0 )
{
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index >
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, 0 >::
tnlGridEntity( const CoordinatesType& coordinates )
: coordinates( coordinates )
{  
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index >
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
const typename tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, 0 >::EntityOrientationType& 
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, 0 >::
getOrientation() const
{
   return this->orientation;
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index >
const typename tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, 0 >::EntityBasisType& 
tnlGridEntity< tnlGrid< Dimensions, Real, Device, Index >, 0 >::
getBasis() const
{
   return this->basis;
}


#endif	/* TNLGRIDENTITY_IMPL_H */

