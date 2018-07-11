/***************************************************************************
                          DistributedGrid_Base.hpp  -  description
                             -------------------
    begin                : July 07, 2018
    copyright            : (C) 2018 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Meshes/DistributedMeshes/DistributedGrid_1D.h>

namespace TNL {
   namespace Meshes {
      namespace DistributedMeshes {


template<int dim, typename RealType, typename Device, typename Index >
DistributedGrid_Base< dim, RealType, Device, Index >::
DistributedGrid_Base()
 : domainDecomposition( 0 ), isSet( false ) {}

template< int dim, typename RealType, typename Device, typename Index >     
void
DistributedGrid_Base< dim, RealType, Device, Index >::
setDomainDecomposition( const CoordinatesType& domainDecomposition )
{
   this->domainDecomposition = domainDecomposition;
}

template< int dim, typename RealType, typename Device, typename Index >     
const typename DistributedGrid_Base< dim, RealType, Device, Index >::CoordinatesType&
DistributedGrid_Base< dim, RealType, Device, Index >::
getDomainDecomposition() const
{
   return this->domainDecomposition;
}
      
template< int dim, typename RealType, typename Device, typename Index >     
bool
DistributedGrid_Base< dim, RealType, Device, Index >::
isDistributed() const
{
   return this->distributed;
};

template< int dim, typename RealType, typename Device, typename Index >     
const typename DistributedGrid_Base< dim, RealType, Device, Index >::CoordinatesType&
DistributedGrid_Base< dim, RealType, Device, Index >::
getOverlap() const
{
   return this->overlap;
};

template< int dim, typename RealType, typename Device, typename Index >     
const typename DistributedGrid_Base< dim, RealType, Device, Index >::CoordinatesType&
DistributedGrid_Base< dim, RealType, Device, Index >::
getLocalSize() const
{
   return this->localSize;
}

template< int dim, typename RealType, typename Device, typename Index >     
const typename DistributedGrid_Base< dim, RealType, Device, Index >::CoordinatesType&
DistributedGrid_Base< dim, RealType, Device, Index >::
getGlobalSize() const
{
   return this->globalGrid.getDimensions();
}

template< int dim, typename RealType, typename Device, typename Index >     
const typename DistributedGrid_Base< dim, RealType, Device, Index >::CoordinatesType&
DistributedGrid_Base< dim, RealType, Device, Index >::
getGlobalBegin() const
{
   return this->globalBegin;
}

template< int dim, typename RealType, typename Device, typename Index >     
const typename DistributedGrid_Base< dim, RealType, Device, Index >::CoordinatesType&
DistributedGrid_Base< dim, RealType, Device, Index >::
getLocalGridSize() const
{
   return this->localGridSize;
}

template< int dim, typename RealType, typename Device, typename Index >     
const typename DistributedGrid_Base< dim, RealType, Device, Index >::CoordinatesType&
DistributedGrid_Base< dim, RealType, Device, Index >::
getLocalBegin() const
{
   return this->localBegin;
}

template< int dim, typename RealType, typename Device, typename Index >      
   template< int EntityDimension >
Index
DistributedGrid_Base< dim, RealType, Device, Index >::
getEntitiesCount() const
{
   return this->globalGrid. template getEntitiesCount< EntityDimension >();
}

template< int dim, typename RealType, typename Device, typename Index >       
   template< typename Entity >
Index
DistributedGrid_Base< dim, RealType, Device, Index >::
getEntitiesCount() const
{
   return this->globalGrid. template getEntitiesCount< Entity >();
}

template< int dim, typename RealType, typename Device, typename Index >    
void 
DistributedGrid_Base< dim, RealType, Device, Index >::
setCommunicationGroup(void * group)
{
    this->communciatonGroup=group;
}

template< int dim, typename RealType, typename Device, typename Index >    
void *
DistributedGrid_Base< dim, RealType, Device, Index >::
getCommunicationGroup() const
{
    return this->communicationGroup;
}

      } //namespace DistributedMeshes
   } // namespace Meshes
} // namespace TNL
