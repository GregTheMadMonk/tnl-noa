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
}

template< typename Grid,
          int EntityDimensions >
const typename tnlGridEntity< Grid, EntityDimensions >::EntityBasisType& 
tnlGridEntity< Grid, EntityDimensions >::
getBasis() const
{
   return this->basis;
}

template< typename Grid,
          int EntityDimensions >
typename tnlGridEntity< Grid, EntityDimensions >::EntityBasisType& 
tnlGridEntity< Grid, EntityDimensions >::
getBasis()
{
   return this->basis;
}

template< typename Grid,
          int EntityDimensions >
void 
tnlGridEntity< Grid, EntityDimensions >::
setBasis( const EntityBasisType& basis )
{
   this->basis = basis;
}

#endif	/* TNLGRIDENTITY_IMPL_H */

