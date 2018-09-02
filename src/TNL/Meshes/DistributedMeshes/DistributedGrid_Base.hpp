/***************************************************************************
                          DistributedGrid_Base.hpp  -  description
                             -------------------
    begin                : July 07, 2018
    copyright            : (C) 2018 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <cstdlib>
#include <TNL/StaticVectorFor.h>
#include <TNL/Communicators/MpiCommunicator.h>
#include <TNL/Exceptions/UnsupportedDimension.h>

#include <iostream>

#include "DistributedGrid_Base.h"

namespace TNL {
   namespace Meshes {
      namespace DistributedMeshes {


template<int Dimension, typename RealType, typename Device, typename Index >
DistributedGrid_Base< Dimension, RealType, Device, Index >::
DistributedGrid_Base()
 : domainDecomposition( 0 ), isSet( false ) {}

template<int Dimension, typename RealType, typename Device, typename Index >
DistributedGrid_Base< Dimension, RealType, Device, Index >::
~DistributedGrid_Base()
{
    if(isSet && this->communicationGroup!=nullptr)
        std::free(this->communicationGroup);
}

template< int Dimension, typename RealType, typename Device, typename Index >     
void
DistributedGrid_Base< Dimension, RealType, Device, Index >::
setDomainDecomposition( const CoordinatesType& domainDecomposition )
{
   this->domainDecomposition = domainDecomposition;
}

template< int Dimension, typename RealType, typename Device, typename Index >     
const typename DistributedGrid_Base< Dimension, RealType, Device, Index >::CoordinatesType&
DistributedGrid_Base< Dimension, RealType, Device, Index >::
getDomainDecomposition() const
{
   return this->domainDecomposition;
}


template< int Dimension, typename RealType, typename Device, typename Index >     
template< typename CommunicatorType >
void
DistributedGrid_Base< Dimension, RealType, Device, Index >::
setGlobalGrid( const GridType &globalGrid )
{
   if(this->isSet && this->communicationGroup != nullptr)
        std::free(this->communicationGroup);
   this->communicationGroup= std::malloc(sizeof(typename CommunicatorType::CommunicationGroup));

   *((typename CommunicatorType::CommunicationGroup *)this->communicationGroup) = CommunicatorType::AllGroup;
   auto group=*((typename CommunicatorType::CommunicationGroup *)this->communicationGroup);

   this->globalGrid = globalGrid;
   this->isSet=true;
   this->overlap.setValue( 1 ); // TODO: Remove this - its only for compatibility with old code
   this->lowerOverlap.setValue( 0 );
   this->upperOverlap.setValue( 0 );

   for( int i = 0; i < getNeighborsCount(); i++ )
      this->neighbors[ i ] = -1;

   this->Dimensions= GridType::getMeshDimension();
   this->spaceSteps=globalGrid.getSpaceSteps();
   this->distributed=false;

   if( CommunicatorType::IsInitialized() )
   {
      this->rank=CommunicatorType::GetRank(group);
      this->nproc=CommunicatorType::GetSize(group);
      //use MPI only if have more than one process
      if(this->nproc>1)
      {
         this->distributed=true;
      }
   }

   if( !this->distributed )
   {
      this->subdomainCoordinates.setValue( 0 );
      this->domainDecomposition.setValue( 0 );
      this->localOrigin=globalGrid.getOrigin();
      this->localGridSize=globalGrid.getDimensions();
      this->localSize=globalGrid.getDimensions();
      this->globalBegin=CoordinatesType(0);
      this->localBegin.setValue( 0 );
      return;
   }
   else
   {
      CoordinatesType numberOfLarger;
      //compute node distribution
      int dims[ Dimension ];
      for( int i = 0; i < Dimension; i++ )
         dims[ i ]= this->domainDecomposition[ i ];


      CommunicatorType::DimsCreate( this->nproc, Dimension, dims );
      for( int i = 0; i < Dimension; i++ )
         this->domainDecomposition[ i ] = dims[ i ];

      // TODO: Make one formula for arbitraty dimension
      switch( Dimension )
      {
         case 1:
            this->subdomainCoordinates[ 0 ] = this->rank;
            break;
         case 2:
            this->subdomainCoordinates[ 0 ] = this->rank % this->domainDecomposition[ 0 ];
            this->subdomainCoordinates[ 1 ] = this->rank / this->domainDecomposition[ 0 ];        
            break;
         case 3:
            this->subdomainCoordinates[ 2 ] =   this->rank / ( this->domainDecomposition[0] * this->domainDecomposition[1] );
            this->subdomainCoordinates[ 1 ] = ( this->rank % ( this->domainDecomposition[0] * this->domainDecomposition[1] ) ) / this->domainDecomposition[0];
            this->subdomainCoordinates[ 0 ] = ( this->rank % ( this->domainDecomposition[0] * this->domainDecomposition[1] ) ) % this->domainDecomposition[0];
            break;
         default:
            throw Exceptions::UnsupportedDimension( Dimension );
      }

      for( int i = 0; i < Dimension; i++ )
      {
         numberOfLarger[ i ] = globalGrid.getDimensions().x() % this->domainDecomposition[ i ];
         
         this->localSize[ i ] = globalGrid.getDimensions()[ i ] / this->domainDecomposition[ i ];
         
         if( numberOfLarger[ i ] > this->subdomainCoordinates[ i ] )
            this->localSize[ i ] += 1;
         
         if( numberOfLarger[ i ] > this->subdomainCoordinates[ i ] )
             this->globalBegin[ i ] = this->subdomainCoordinates[ i ] * this->localSize[ i ];
         else
             this->globalBegin[ i ] = numberOfLarger[ i ] * ( this->localSize[ i ] + 1 ) + 
                                     ( this->subdomainCoordinates[ i ] - numberOfLarger[ i ] ) * this->localSize[ i ];
      }

      this->localGridSize = this->localSize;
      this->setupNeighbors();
  }
}

template< int Dimension, typename RealType, typename Device, typename Index >     
void
DistributedGrid_Base< Dimension, RealType, Device, Index >::
setOverlaps( const SubdomainOverlapsType& lower,
             const SubdomainOverlapsType& upper )
{
   this->lowerOverlap = lower;
   this->upperOverlap = upper;
   
   for( int i = 0; i < Dimension; i++ )
   {
      this->localOrigin[ i ] = this->globalGrid.getOrigin()[ i ] +
         this->globalGrid.getSpaceSteps()[ i ] * 
            ( this->globalBegin[ i ] - this->lowerOverlap[ i ] );         

   }

   this->localBegin = this->lowerOverlap;
   this->localGridSize = this->localSize + this->lowerOverlap + this->upperOverlap;
   //this->print( std::cerr );   
}


template< int Dimension, typename RealType, typename Device, typename Index >     
const typename DistributedGrid_Base< Dimension, RealType, Device, Index >::CoordinatesType&
DistributedGrid_Base< Dimension, RealType, Device, Index >::
getSubdomainCoordinates() const
{
   return this->subdomainCoordinates;
}

template< int Dimension, typename RealType, typename Device, typename Index >     
const typename DistributedGrid_Base< Dimension, RealType, Device, Index >::PointType&
DistributedGrid_Base< Dimension, RealType, Device, Index >::
getLocalOrigin() const
{
   return this->localOrigin;
}

template< int Dimension, typename RealType, typename Device, typename Index >     
const typename DistributedGrid_Base< Dimension, RealType, Device, Index >::PointType&
DistributedGrid_Base< Dimension, RealType, Device, Index >::
getSpaceSteps() const
{
   return this->spaceSteps;
}
   
template< int Dimension, typename RealType, typename Device, typename Index >     
bool
DistributedGrid_Base< Dimension, RealType, Device, Index >::
isDistributed() const
{
   return this->distributed;
};

template< int Dimension, typename RealType, typename Device, typename Index >     
const typename DistributedGrid_Base< Dimension, RealType, Device, Index >::CoordinatesType&
DistributedGrid_Base< Dimension, RealType, Device, Index >::
getOverlap() const
{
   return this->overlap;
};

template< int Dimension, typename RealType, typename Device, typename Index >     
const typename DistributedGrid_Base< Dimension, RealType, Device, Index >::CoordinatesType&
DistributedGrid_Base< Dimension, RealType, Device, Index >::
getLocalSize() const
{
   return this->localSize;
}

template< int Dimension, typename RealType, typename Device, typename Index >     
const typename DistributedGrid_Base< Dimension, RealType, Device, Index >::CoordinatesType&
DistributedGrid_Base< Dimension, RealType, Device, Index >::
getGlobalSize() const
{
   return this->globalGrid.getDimensions();
}

template< int Dimension, typename RealType, typename Device, typename Index >     
const typename DistributedGrid_Base< Dimension, RealType, Device, Index >::GridType&
DistributedGrid_Base< Dimension, RealType, Device, Index >::
getGlobalGrid() const
{
    return this->globalGrid;
}

template< int Dimension, typename RealType, typename Device, typename Index >     
const typename DistributedGrid_Base< Dimension, RealType, Device, Index >::CoordinatesType&
DistributedGrid_Base< Dimension, RealType, Device, Index >::
getGlobalBegin() const
{
   return this->globalBegin;
}

template< int Dimension, typename RealType, typename Device, typename Index >     
const typename DistributedGrid_Base< Dimension, RealType, Device, Index >::CoordinatesType&
DistributedGrid_Base< Dimension, RealType, Device, Index >::
getLocalGridSize() const
{
   return this->localGridSize;
}

template< int Dimension, typename RealType, typename Device, typename Index >     
const typename DistributedGrid_Base< Dimension, RealType, Device, Index >::CoordinatesType&
DistributedGrid_Base< Dimension, RealType, Device, Index >::
getLocalBegin() const
{
   return this->localBegin;
}

template< int Dimension, typename RealType, typename Device, typename Index >      
   template< int EntityDimension >
Index
DistributedGrid_Base< Dimension, RealType, Device, Index >::
getEntitiesCount() const
{
   return this->globalGrid. template getEntitiesCount< EntityDimension >();
}

template< int Dimension, typename RealType, typename Device, typename Index >       
   template< typename Entity >
Index
DistributedGrid_Base< Dimension, RealType, Device, Index >::
getEntitiesCount() const
{
   return this->globalGrid. template getEntitiesCount< Entity >();
}

template< int Dimension, typename RealType, typename Device, typename Index >    
void 
DistributedGrid_Base< Dimension, RealType, Device, Index >::
setCommunicationGroup(void * group)
{
    this->communicationGroup=group;
}

template< int Dimension, typename RealType, typename Device, typename Index >    
void *
DistributedGrid_Base< Dimension, RealType, Device, Index >::
getCommunicationGroup() const
{
    return this->communicationGroup;
}

template< int Dimension, typename RealType, typename Device, typename Index >    
int
DistributedGrid_Base< Dimension, RealType, Device, Index >::
getRankOfProcCoord(const CoordinatesType &nodeCoordinates) const
{
    int DimensionOffset=1;
    int ret=0;
    for(int i=0;i<Dimension;i++)
    {
        ret += DimensionOffset*nodeCoordinates[i];
        DimensionOffset *= this->domainDecomposition[i];
    }
    return ret;
}

template< int Dimension, typename RealType, typename Device, typename Index >    
bool
DistributedGrid_Base< Dimension, RealType, Device, Index >::
isThereNeighbor(const CoordinatesType &direction) const
{
    bool res=true;
    for(int i=0;i<Dimension;i++)
    {
        if(direction[i]==-1)
            res&= this->subdomainCoordinates[i]>0;

        if(direction[i]==1)
            res&= this->subdomainCoordinates[i]<this->domainDecomposition[i]-1;
    }
    return res;

}

template< int Dimension, typename RealType, typename Device, typename Index >    
void
DistributedGrid_Base< Dimension, RealType, Device, Index >::
setupNeighbors()
{
   int *neighbors = this->neighbors;

   for( int i = 0; i < getNeighborsCount(); i++ )
   {
      auto direction = Directions::template getXYZ< Dimension >( i );
      auto coordinates = this->subdomainCoordinates+direction;
      if( this->isThereNeighbor( direction ) )
         this->neighbors[ i ] = this->getRankOfProcCoord( coordinates );
      else
         this->neighbors[ i ] =- 1;
      
      // Handling periodic neighbors
      for( int d = 0; d < Dimension; d++ )
      {
         if( coordinates[ d ] == -1 )
            coordinates[ d ] = this->domainDecomposition[ d ] - 1;
         if( coordinates[ d ] == this->domainDecomposition[ d ] )
            coordinates[ d ] = 0;
         this->periodicNeighbors[ i ] = this->getRankOfProcCoord( coordinates );
      }
      
      //std::cout << "Setting i-th neighbour to " << neighbors[ i ] << " and " << periodicNeighbors[ i ] << std::endl;
   }
}

template< int Dimension, typename RealType, typename Device, typename Index >   
const int*
DistributedGrid_Base< Dimension, RealType, Device, Index >::
getNeighbors() const
{
    TNL_ASSERT_TRUE(this->isSet,"DistributedGrid is not set, but used by getNeighbors");
    return this->neighbors;
}

template< int Dimension, typename RealType, typename Device, typename Index >   
const int*
DistributedGrid_Base< Dimension, RealType, Device, Index >::
getPeriodicNeighbors() const
{
    TNL_ASSERT_TRUE(this->isSet,"DistributedGrid is not set, but used by getNeighbors");
    return this->periodicNeighbors;
}

template< int Dimension, typename RealType, typename Device, typename Index >
    template<typename CommunicatorType, typename DistributedGridType >
bool 
DistributedGrid_Base< Dimension, RealType, Device, Index >::
SetupByCut(DistributedGridType &inputDistributedGrid, 
         Containers::StaticVector<Dimension, int> savedDimensions, 
         Containers::StaticVector<DistributedGridType::getMeshDimension()-Dimension,int> reducedDimensions, 
         Containers::StaticVector<DistributedGridType::getMeshDimension()-Dimension,IndexType> fixedIndexs)
{

      int coDimensionension=DistributedGridType::getMeshDimension()-Dimension;

      bool isInCut=true;
      for(int i=0;i<coDimensionension; i++)
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
            
            for(int i=0; i<Dimension;i++)
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
            for(int i=0;i<Dimension; i++)
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

template< int Dimension, typename RealType, typename Device, typename Index >    
void
DistributedGrid_Base< Dimension, RealType, Device, Index >::
print( ostream& str ) const
{
   using Communicator = Communicators::MpiCommunicator;
   for( int j = 0; j < Communicator::GetSize( Communicator::AllGroup ); j++ )
   {
      if( j == Communicator::GetRank( Communicator::AllGroup ) )
      {
         str << "Node : " << Communicator::GetRank( Communicator::AllGroup ) << std::endl
             << " localOrigin : " << localOrigin << std::endl
             << " localBegin : " << localBegin << std::endl
             << " localSize : " << localSize  << std::endl
             << " localGridSize : " << localGridSize << std::endl
             << " overlap : " << overlap << std::endl
             << " globalBegin : " << globalBegin << std::endl
             << " spaceSteps : " << spaceSteps << std::endl
             << " lowerOverlap : " << lowerOverlap << std::endl
             << " upperOverlap : " << upperOverlap << std::endl
             << " domainDecomposition : " << domainDecomposition << std::endl
             << " subdomainCoordinates : " << subdomainCoordinates << std::endl
             << " neighbors : ";
         for( int i = 0; i < getNeighborsCount(); i++ )
            str << " " << neighbors[ i ];
         str << std::endl;
         str << " periodicNeighbours : ";
         for( int i = 0; i < getNeighborsCount(); i++ )
            str << " " << periodicNeighbors[ i ];
         str << std::endl;
      }
      Communicator::Barrier( Communicator::AllGroup );
   }
}

      } //namespace DistributedMeshes
   } // namespace Meshes
} // namespace TNL
