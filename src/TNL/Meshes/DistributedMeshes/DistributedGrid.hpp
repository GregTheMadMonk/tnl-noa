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

#include "DistributedGrid.h"
#include <TNL/MPI/Wrappers.h>

namespace TNL {
   namespace Meshes {
      namespace DistributedMeshes {

template<int Dimension, typename Real, typename Device, typename Index >
DistributedMesh< Grid< Dimension, Real, Device, Index > >::
DistributedMesh()
 : domainDecomposition( 0 ), isSet( false ) {}

template<int Dimension, typename Real, typename Device, typename Index >
DistributedMesh< Grid< Dimension, Real, Device, Index > >::
~DistributedMesh()
{
}


template<int Dimension, typename Real, typename Device, typename Index >
void
DistributedMesh< Grid< Dimension, Real, Device, Index > >::
configSetup( Config::ConfigDescription& config )
{
   config.addEntry< int >( "grid-domain-decomposition-x", "Number of grid subdomains along x-axis.", 0 );
   config.addEntry< int >( "grid-domain-decomposition-y", "Number of grid subdomains along y-axis.", 0 );
   config.addEntry< int >( "grid-domain-decomposition-z", "Number of grid subdomains along z-axis.", 0 );
}

template<int Dimension, typename Real, typename Device, typename Index >
bool
DistributedMesh< Grid< Dimension, Real, Device, Index > >::
setup( const Config::ParameterContainer& parameters,
       const String& prefix )
{
   this->domainDecomposition[ 0 ] = parameters.getParameter< int >( "grid-domain-decomposition-x" );
   if( Dimension > 1 )
      this->domainDecomposition[ 1 ] = parameters.getParameter< int >( "grid-domain-decomposition-y" );
   if( Dimension > 2 )
      this->domainDecomposition[ 2 ] = parameters.getParameter< int >( "grid-domain-decomposition-z" );
   return true;
}

template< int Dimension, typename Real, typename Device, typename Index >
void
DistributedMesh< Grid< Dimension, Real, Device, Index > >::
setDomainDecomposition( const CoordinatesType& domainDecomposition )
{
   this->domainDecomposition = domainDecomposition;
}

template< int Dimension, typename Real, typename Device, typename Index >
const typename DistributedMesh< Grid< Dimension, Real, Device, Index > >::CoordinatesType&
DistributedMesh< Grid< Dimension, Real, Device, Index > >::
getDomainDecomposition() const
{
   return this->domainDecomposition;
}

template< int Dimension, typename Real, typename Device, typename Index >
void
DistributedMesh< Grid< Dimension, Real, Device, Index > >::
setGlobalGrid( const GridType &globalGrid )
{
   this->group = MPI::AllGroup();

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

   this->rank=MPI::GetRank(group);
   this->nproc=MPI::GetSize(group);
   //use MPI only if have more than one process
   if(this->nproc>1)
   {
      this->distributed=true;
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
         dims[ i ] = this->domainDecomposition[ i ];
      MPI::Compute_dims( this->nproc, Dimension, dims );
      for( int i = 0; i < Dimension; i++ )
         this->domainDecomposition[ i ] = dims[ i ];

      int size = this->nproc;
      int tmp = this->rank;
      for( int i = Dimension - 1; i >= 0; i-- )
      {
         size = size / this->domainDecomposition[ i ];
         this->subdomainCoordinates[ i ] = tmp / size;
         tmp = tmp % size;
      }

      for( int i = 0; i < Dimension; i++ )
      {
         numberOfLarger[ i ] = globalGrid.getDimensions()[ i ] % this->domainDecomposition[ i ];

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

template< int Dimension, typename Real, typename Device, typename Index >
void
DistributedMesh< Grid< Dimension, Real, Device, Index > >::
setOverlaps( const SubdomainOverlapsType& lower,
             const SubdomainOverlapsType& upper)
{
   this->lowerOverlap = lower;
   this->upperOverlap = upper;
   this->localOrigin = this->globalGrid.getOrigin() + this->globalGrid.getSpaceSteps() * (this->globalBegin - this->lowerOverlap);
   this->localBegin = this->lowerOverlap;
   this->localGridSize = this->localSize + this->lowerOverlap + this->upperOverlap;
}


template< int Dimension, typename Real, typename Device, typename Index >
void
DistributedMesh< Grid< Dimension, Real, Device, Index > >::
setupGrid( GridType& grid)
{
   TNL_ASSERT_TRUE(this->isSet,"DistributedGrid is not set, but used by SetupGrid");
   grid.setOrigin(this->localOrigin);
   grid.setDimensions(this->localGridSize);
   //compute local proportions by side efect
   grid.setSpaceSteps(this->spaceSteps);
   grid.setDistMesh(this);
};

template< int Dimension, typename Real, typename Device, typename Index >
const typename DistributedMesh< Grid< Dimension, Real, Device, Index > >::CoordinatesType&
DistributedMesh< Grid< Dimension, Real, Device, Index > >::
getSubdomainCoordinates() const
{
   return this->subdomainCoordinates;
}

template< int Dimension, typename Real, typename Device, typename Index >
const typename DistributedMesh< Grid< Dimension, Real, Device, Index > >::PointType&
DistributedMesh< Grid< Dimension, Real, Device, Index > >::
getLocalOrigin() const
{
   return this->localOrigin;
}

template< int Dimension, typename Real, typename Device, typename Index >
const typename DistributedMesh< Grid< Dimension, Real, Device, Index > >::PointType&
DistributedMesh< Grid< Dimension, Real, Device, Index > >::
getSpaceSteps() const
{
   return this->spaceSteps;
}

template< int Dimension, typename Real, typename Device, typename Index >
bool
DistributedMesh< Grid< Dimension, Real, Device, Index > >::
isDistributed() const
{
   return this->distributed;
};

template< int Dimension, typename Real, typename Device, typename Index >
bool
DistributedMesh< Grid< Dimension, Real, Device, Index > >::
isBoundarySubdomain() const
{
   for( int i = 0; i < getNeighborsCount(); i++ )
      if( this->neighbors[ i ] == -1 )
         return true;
   return false;
}

template< int Dimension, typename Real, typename Device, typename Index >
const typename DistributedMesh< Grid< Dimension, Real, Device, Index > >::CoordinatesType&
DistributedMesh< Grid< Dimension, Real, Device, Index > >::
getLowerOverlap() const
{
   return this->lowerOverlap;
};

template< int Dimension, typename Real, typename Device, typename Index >
const typename DistributedMesh< Grid< Dimension, Real, Device, Index > >::CoordinatesType&
DistributedMesh< Grid< Dimension, Real, Device, Index > >::
getUpperOverlap() const
{
   return this->upperOverlap;
};

template< int Dimension, typename Real, typename Device, typename Index >
const typename DistributedMesh< Grid< Dimension, Real, Device, Index > >::CoordinatesType&
DistributedMesh< Grid< Dimension, Real, Device, Index > >::
getLocalSize() const
{
   return this->localSize;
}

template< int Dimension, typename Real, typename Device, typename Index >
const typename DistributedMesh< Grid< Dimension, Real, Device, Index > >::CoordinatesType&
DistributedMesh< Grid< Dimension, Real, Device, Index > >::
getGlobalSize() const
{
   return this->globalGrid.getDimensions();
}

template< int Dimension, typename Real, typename Device, typename Index >
const typename DistributedMesh< Grid< Dimension, Real, Device, Index > >::GridType&
DistributedMesh< Grid< Dimension, Real, Device, Index > >::
getGlobalGrid() const
{
    return this->globalGrid;
}

template< int Dimension, typename Real, typename Device, typename Index >
const typename DistributedMesh< Grid< Dimension, Real, Device, Index > >::CoordinatesType&
DistributedMesh< Grid< Dimension, Real, Device, Index > >::
getGlobalBegin() const
{
   return this->globalBegin;
}

template< int Dimension, typename Real, typename Device, typename Index >
const typename DistributedMesh< Grid< Dimension, Real, Device, Index > >::CoordinatesType&
DistributedMesh< Grid< Dimension, Real, Device, Index > >::
getLocalGridSize() const
{
   return this->localGridSize;
}

template< int Dimension, typename Real, typename Device, typename Index >
const typename DistributedMesh< Grid< Dimension, Real, Device, Index > >::CoordinatesType&
DistributedMesh< Grid< Dimension, Real, Device, Index > >::
getLocalBegin() const
{
   return this->localBegin;
}

template< int Dimension, typename Real, typename Device, typename Index >
   template< int EntityDimension >
Index
DistributedMesh< Grid< Dimension, Real, Device, Index > >::
getEntitiesCount() const
{
   return this->globalGrid. template getEntitiesCount< EntityDimension >();
}

template< int Dimension, typename Real, typename Device, typename Index >
   template< typename Entity >
Index
DistributedMesh< Grid< Dimension, Real, Device, Index > >::
getEntitiesCount() const
{
   return this->globalGrid. template getEntitiesCount< Entity >();
}

template< int Dimension, typename Real, typename Device, typename Index >
void
DistributedMesh< Grid< Dimension, Real, Device, Index > >::
setCommunicationGroup(MPI_Comm group)
{
    this->group=group;
}

template< int Dimension, typename Real, typename Device, typename Index >
MPI_Comm
DistributedMesh< Grid< Dimension, Real, Device, Index > >::
getCommunicationGroup() const
{
    return this->group;
}

template< int Dimension, typename Real, typename Device, typename Index >
int
DistributedMesh< Grid< Dimension, Real, Device, Index > >::
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

template< int Dimension, typename Real, typename Device, typename Index >
bool
DistributedMesh< Grid< Dimension, Real, Device, Index > >::
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

template< int Dimension, typename Real, typename Device, typename Index >
void
DistributedMesh< Grid< Dimension, Real, Device, Index > >::
setupNeighbors()
{
   for( int i = 0; i < getNeighborsCount(); i++ )
   {
      auto direction = Directions::template getXYZ< Dimension >( i );
      CoordinatesType coordinates = this->subdomainCoordinates+direction;
      if( this->isThereNeighbor( direction ) )
         this->neighbors[ i ] = this->getRankOfProcCoord( coordinates );
      else
         this->neighbors[ i ] = -1;

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

template< int Dimension, typename Real, typename Device, typename Index >
const int*
DistributedMesh< Grid< Dimension, Real, Device, Index > >::
getNeighbors() const
{
    TNL_ASSERT_TRUE(this->isSet,"DistributedGrid is not set, but used by getNeighbors");
    return this->neighbors;
}

template< int Dimension, typename Real, typename Device, typename Index >
const int*
DistributedMesh< Grid< Dimension, Real, Device, Index > >::
getPeriodicNeighbors() const
{
    TNL_ASSERT_TRUE(this->isSet,"DistributedGrid is not set, but used by getNeighbors");
    return this->periodicNeighbors;
}

template< int Dimension, typename Real, typename Device, typename Index >
    template<typename DistributedGridType >
bool
DistributedMesh< Grid< Dimension, Real, Device, Index > >::
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
      const MPI_Comm oldGroup=inputDistributedGrid.getCommunicationGroup();
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

                this->overlap[i]=inputDistributedGrid.getOverlap()[savedDimensions[i]];//TODO: RomoveThis
                this->lowerOverlap[i]=inputDistributedGrid.getLowerOverlap()[savedDimensions[i]];
                this->upperOverlap[i]=inputDistributedGrid.getUpperOverlap()[savedDimensions[i]];
                this->localSize[i]=inputDistributedGrid.getLocalSize()[savedDimensions[i]];
                this->globalBegin[i]=inputDistributedGrid.getGlobalBegin()[savedDimensions[i]];
                this->localGridSize[i]=inputDistributedGrid.getLocalGridSize()[savedDimensions[i]];
                this->localBegin[i]=inputDistributedGrid.getLocalBegin()[savedDimensions[i]];

                this->localOrigin[i]=inputDistributedGrid.getLocalOrigin()[savedDimensions[i]];
                this->spaceSteps[i]=inputDistributedGrid.getSpaceSteps()[savedDimensions[i]];
            }

            int newRank = getRankOfProcCoord(this->subdomainCoordinates);
            this->group = MPI::Comm_split( oldGroup, 1, newRank );

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
         this->group = MPI::Comm_split( oldGroup, MPI_UNDEFINED, 0 );
      }

      return false;
}

template< int Dimension, typename Real, typename Device, typename Index >
String
DistributedMesh< Grid< Dimension, Real, Device, Index > >::
printProcessCoords() const
{
   String res = convertToString(this->subdomainCoordinates[0]);
   for(int i=1; i<Dimension; i++)
        res=res+String("-")+convertToString(this->subdomainCoordinates[i]);
   return res;
};

template< int Dimension, typename Real, typename Device, typename Index >
String
DistributedMesh< Grid< Dimension, Real, Device, Index > >::
printProcessDistr() const
{
   String res = convertToString(this->domainDecomposition[0]);
   for(int i=1; i<Dimension; i++)
        res=res+String("-")+convertToString(this->domainDecomposition[i]);
   return res;
};

template< int Dimension, typename Real, typename Device, typename Index >
void
DistributedMesh< Grid< Dimension, Real, Device, Index > >::
writeProlog( Logger& logger )
{
   logger.writeParameter( "Domain decomposition:", this->getDomainDecomposition() );
}

template< int Dimension, typename Real, typename Device, typename Index >
void
DistributedMesh< Grid< Dimension, Real, Device, Index > >::
print( std::ostream& str ) const
{
   for( int j = 0; j < MPI::GetSize(); j++ )
   {
      if( j == MPI::GetRank() )
      {
         str << "Node : " << MPI::GetRank() << std::endl
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
      MPI::Barrier();
   }
}

      } //namespace DistributedMeshes
   } // namespace Meshes
} // namespace TNL
