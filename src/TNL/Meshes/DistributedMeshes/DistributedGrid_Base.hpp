/***************************************************************************
                          DistributedGrid_Base.hpp  -  description
                             -------------------
    begin                : July 07, 2018
    copyright            : (C) 2018 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/StaticVectorFor.h>
#include <cstdlib>

#include <iostream>

#include "DistributedGrid_Base.h"

namespace TNL {
   namespace Meshes {
      namespace DistributedMeshes {


template<int dim, typename RealType, typename Device, typename Index >
DistributedGrid_Base< dim, RealType, Device, Index >::
DistributedGrid_Base()
 : domainDecomposition( 0 ), isSet( false ) {}

template<int dim, typename RealType, typename Device, typename Index >
DistributedGrid_Base< dim, RealType, Device, Index >::
~DistributedGrid_Base()
{
    if(isSet && this->communicationGroup!=nullptr)
        std::free(this->communicationGroup);
}

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
const typename DistributedGrid_Base< dim, RealType, Device, Index >::CoordinatesType&
DistributedGrid_Base< dim, RealType, Device, Index >::
getSubdomainCoordinates() const
{
   return this->subdomainCoordinates;
}

template< int dim, typename RealType, typename Device, typename Index >     
const typename DistributedGrid_Base< dim, RealType, Device, Index >::PointType&
DistributedGrid_Base< dim, RealType, Device, Index >::
getLocalOrigin() const
{
   return this->localOrigin;
}

template< int dim, typename RealType, typename Device, typename Index >     
const typename DistributedGrid_Base< dim, RealType, Device, Index >::PointType&
DistributedGrid_Base< dim, RealType, Device, Index >::
getSpaceSteps() const
{
   return this->spaceSteps;
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
const typename DistributedGrid_Base< dim, RealType, Device, Index >::GridType&
DistributedGrid_Base< dim, RealType, Device, Index >::
getGlobalGrid() const
{
    return this->globalGrid;
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

template< int dim, typename RealType, typename Device, typename Index >    
int
DistributedGrid_Base< dim, RealType, Device, Index >::
getRankOfProcCoord(const CoordinatesType &nodeCoordinates) const
{
    int dimOffset=1;
    int ret=0;
    for(int i=0;i<dim;i++)
    {
        ret += dimOffset*nodeCoordinates[i];
        dimOffset *= this->domainDecomposition[i];
    }
    return ret;
}

template< int dim, typename RealType, typename Device, typename Index >    
bool
DistributedGrid_Base< dim, RealType, Device, Index >::
isThereNeighbor(const CoordinatesType &direction) const
{
    bool res=true;
    for(int i=0;i<dim;i++)
    {
        if(direction[i]==-1)
            res&= this->subdomainCoordinates[i]>0;

        if(direction[i]==1)
            res&= this->subdomainCoordinates[i]<this->domainDecomposition[i]-1;
    }
    return res;

}

template< int dim, typename RealType, typename Device, typename Index >    
void
DistributedGrid_Base< dim, RealType, Device, Index >::
setupNeighbors()
{
   int *neighbors = this->neighbors;

   for( int i = 0; i < getNeighborsCount(); i++ )
   {
      auto direction = Directions::template getXYZ< dim >( i );
      auto coordinates = this->subdomainCoordinates+direction;
      if( this->isThereNeighbor( direction ) )
         this->neighbors[ i ] = this->getRankOfProcCoord( coordinates );
      else
         this->neighbors[ i ] =- 1;
      
      // Handling periodic neighbors
      for( int d = 0; d < dim; d++ )
      {
         if( coordinates[ d ] == -1 )
            coordinates[ d ] = this->domainDecomposition[ d ] - 1;
         if( coordinates[ d ] == this->domainDecomposition[ d ] )
            coordinates[ d ] = 0;
         this->periodicNeighbors[ i ] = this->getRankOfProcCoord( coordinates );
      }
      
      std::cout << "Setting i-th neigbour to " << neighbors[ i ] << " and " << periodicNeighbors[ i ] << std::endl;
   }
}

template< int dim, typename RealType, typename Device, typename Index >   
const int*
DistributedGrid_Base< dim, RealType, Device, Index >::
getNeighbors() const
{
    TNL_ASSERT_TRUE(this->isSet,"DistributedGrid is not set, but used by getNeighbors");
    return this->neighbors;
}

template< int dim, typename RealType, typename Device, typename Index >    
    template<typename CommunicatorType, typename DistributedGridType >
bool 
DistributedGrid_Base< dim, RealType, Device, Index >::
SetupByCut(DistributedGridType &inputDistributedGrid, 
         Containers::StaticVector<dim, int> savedDimensions, 
         Containers::StaticVector<DistributedGridType::getMeshDimension()-dim,int> reducedDimensions, 
         Containers::StaticVector<DistributedGridType::getMeshDimension()-dim,IndexType> fixedIndexs)
{

      int codimension=DistributedGridType::getMeshDimension()-dim;

      bool isInCut=true;
      for(int i=0;i<codimension; i++)
      {
            auto begin=inputDistributedGrid.getGlobalBegin();
            auto size= inputDistributedGrid.getLocalSize();
            isInCut &= fixedIndexs[i]>begin[reducedDimensions[i]] && fixedIndexs[i]< (begin[reducedDimensions[i]]+size[reducedDimensions[i]]);
      }

      //create new group with used nodes
      typename CommunicatorType::CommunicationGroup *oldGroup=(typename CommunicatorType::CommunicationGroup *)(inputDistributedGrid.getCommunicationGroup());
      if(this->isSet && this->communicationGroup != nullptr)
            free(this->communicationGroup);
      this->communicationGroup = std::malloc(sizeof(typename CommunicatorType::CommunicationGroup));

      if(isInCut)
      {
           this->isSet=true;
            
            auto fromGlobalMesh=inputDistributedGrid.getGlobalGrid();
            //set global grid
            typename GridType::PointType outOrigin;
            typename GridType::PointType outProportions;
            typename GridType::CoordinatesType outDimensions;
            
            for(int i=0; i<dim;i++)
            {
                outOrigin[i]=fromGlobalMesh.getOrigin()[savedDimensions[i]];
                outProportions[i]=fromGlobalMesh.getProportions()[savedDimensions[i]];
                outDimensions[i]=fromGlobalMesh.getDimensions()[savedDimensions[i]];

                this->domainDecomposition[i]=inputDistributedGrid.getDomainDecomposition()[savedDimensions[i]];
                this->subdomainCoordinates[i]=inputDistributedGrid.getSubdomainCoordinates()[savedDimensions[i]];

                this->overlap[i]=inputDistributedGrid.getOverlap()[savedDimensions[i]];
                this->localSize[i]=inputDistributedGrid.getLocalSize()[savedDimensions[i]];
                this->globalBegin[i]=inputDistributedGrid.getGlobalBegin()[savedDimensions[i]];
                this->localGridSize[i]=inputDistributedGrid.getLocalGridSize()[savedDimensions[i]];
                this->localBegin[i]=inputDistributedGrid.getLocalBegin()[savedDimensions[i]];

                this->localOrigin[i]=inputDistributedGrid.getLocalOrigin()[savedDimensions[i]];
                this->spaceSteps[i]=inputDistributedGrid.getSpaceSteps()[savedDimensions[i]];
            }

            int newRank= getRankOfProcCoord(this->subdomainCoordinates);

            CommunicatorType::CreateNewGroup(isInCut,newRank,*oldGroup ,*((typename CommunicatorType::CommunicationGroup*) this->communicationGroup));

            setupNeighbors();


            
            bool isDistributed=false;
            for(int i=0;i<dim; i++)
            {
                isDistributed|=(domainDecomposition[i]>1);
            }

            this->distributed=isDistributed;
            
            this->globalGrid.setDimensions(outDimensions);
            this->globalGrid.setDomain(outOrigin,outProportions);

            return true;
      }
      else
      {
         CommunicatorType::CreateNewGroup(isInCut,0,*oldGroup ,*((typename CommunicatorType::CommunicationGroup*) this->communicationGroup));
      }

      return false;
}

      } //namespace DistributedMeshes
   } // namespace Meshes
} // namespace TNL
