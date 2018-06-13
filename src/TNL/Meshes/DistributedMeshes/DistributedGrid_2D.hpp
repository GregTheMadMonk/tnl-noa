/***************************************************************************
                          DistributedGrid_2D.hpp  -  description
                             -------------------
    begin                : January 09, 2018
    copyright            : (C) 2018 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Meshes/DistributedMeshes/DistributedGrid_2D.h>

namespace TNL {
   namespace Meshes {
      namespace DistributedMeshes {

template< typename RealType, typename Device, typename Index >
DistributedMesh< Grid< 2, RealType, Device, Index > >::
DistributedMesh()
: domainDecomposition( 0 ), isSet( false ) {}

template< typename RealType, typename Device, typename Index >
void
DistributedMesh< Grid< 2, RealType, Device, Index > >::
setDomainDecomposition( const CoordinatesType& domainDecomposition )
{
   this->domainDecomposition = domainDecomposition;
}

template< typename RealType, typename Device, typename Index >
const typename DistributedMesh< Grid< 2, RealType, Device, Index > >::CoordinatesType&
DistributedMesh< Grid< 2, RealType, Device, Index > >::
getDomainDecomposition() const
{
   return this->domainDecomposition;
}

template< typename RealType, typename Device, typename Index >     
   template< int EntityDimension >
Index
DistributedMesh< Grid< 2, RealType, Device, Index > >::
getEntitiesCount() const
{
   return this->globalGrid. template getEntitiesCount< EntityDimension >();
}

template< typename RealType, typename Device, typename Index >     
   template< typename Entity >
Index
DistributedMesh< Grid< 2, RealType, Device, Index > >::
getEntitiesCount() const
{
   return this->globalGrid. template getEntitiesCount< Entity >();
}

template< typename RealType, typename Device, typename Index >
bool
DistributedMesh< Grid< 2, RealType, Device, Index > >::
setup( const Config::ParameterContainer& parameters,
       const String& prefix )
{
   this->domainDecomposition.x() = parameters.getParameter< int >( "grid-domain-decomposition-x" );
   this->domainDecomposition.y() = parameters.getParameter< int >( "grid-domain-decomposition-y" );
   return true;
}      

template< typename RealType, typename Device, typename Index >
template< typename CommunicatorType >
void
DistributedMesh< Grid< 2, RealType, Device, Index > >::
setGlobalGrid( const GridType &globalGrid,
               const CoordinatesType& overlap )
{
   this->globalGrid = globalGrid;
   isSet=true;
   this->overlap=overlap;

   for( int i=0; i<8; i++ )
      neighbors[i]=-1;

   Dimensions= GridType::getMeshDimension();
   spaceSteps=globalGrid.getSpaceSteps();
   distributed=false;

   if( CommunicatorType::IsInitialized() )
   {
      rank=CommunicatorType::GetRank();
      this->nproc=CommunicatorType::GetSize();
      //use MPI only if have more than one process
      if(this->nproc>1)
      {
         distributed=true;
      }
   }

   if( !distributed )
   {
      subdomainCoordinates[0]=0;
      subdomainCoordinates[1]=0;
      domainDecomposition[0]=1;
      domainDecomposition[1]=1;
      localOrigin=globalGrid.getOrigin();
      localGridSize=globalGrid.getDimensions();
      localSize=globalGrid.getDimensions();
      globalSize=globalGrid.getDimensions();
      globalBegin=CoordinatesType(0);
      localBegin.x()=0;
      localBegin.y()=0;

      return;
   }
   else
   {
      //compute node distribution
      int dims[ 2 ];
      dims[ 0 ] = domainDecomposition[ 0 ];
      dims[ 1 ] = domainDecomposition[ 1 ];

      CommunicatorType::DimsCreate( nproc, 2, dims );
      domainDecomposition[ 0 ] = dims[ 0 ];
      domainDecomposition[ 1 ] = dims[ 1 ];

      subdomainCoordinates[ 0 ] = rank % domainDecomposition[ 0 ];
      subdomainCoordinates[ 1 ] = rank / domainDecomposition[ 0 ];        

      //compute local mesh size
      globalSize=globalGrid.getDimensions();              
      numberOfLarger[0]=globalGrid.getDimensions().x()%domainDecomposition[0];
      numberOfLarger[1]=globalGrid.getDimensions().y()%domainDecomposition[1];

      localSize.x()=(globalGrid.getDimensions().x()/domainDecomposition[0]);
      localSize.y()=(globalGrid.getDimensions().y()/domainDecomposition[1]);

      if(numberOfLarger[0]>subdomainCoordinates[0])
           localSize.x()+=1;               
      if(numberOfLarger[1]>subdomainCoordinates[1])
          localSize.y()+=1;

      if(numberOfLarger[0]>subdomainCoordinates[0])
          globalBegin.x()=subdomainCoordinates[0]*localSize.x();
      else
          globalBegin.x()=numberOfLarger[0]*(localSize.x()+1)+(subdomainCoordinates[0]-numberOfLarger[0])*localSize.x();

      if(numberOfLarger[1]>subdomainCoordinates[1])
          globalBegin.y()=subdomainCoordinates[1]*localSize.y();

      else
          globalBegin.y()=numberOfLarger[1]*(localSize.y()+1)+(subdomainCoordinates[1]-numberOfLarger[1])*localSize.y();

      localOrigin=globalGrid.getOrigin()+TNL::Containers::tnlDotProduct(globalGrid.getSpaceSteps(),globalBegin-overlap);

      //nearnodes
      if(subdomainCoordinates[0]>0)
          neighbors[Left]=getRankOfProcCoord(subdomainCoordinates[0]-1,subdomainCoordinates[1]);
      if(subdomainCoordinates[0]<domainDecomposition[0]-1)
          neighbors[Right]=getRankOfProcCoord(subdomainCoordinates[0]+1,subdomainCoordinates[1]);
      if(subdomainCoordinates[1]>0)
          neighbors[Up]=getRankOfProcCoord(subdomainCoordinates[0],subdomainCoordinates[1]-1);
      if(subdomainCoordinates[1]<domainDecomposition[1]-1)
          neighbors[Down]=getRankOfProcCoord(subdomainCoordinates[0],subdomainCoordinates[1]+1);
      if(subdomainCoordinates[0]>0 && subdomainCoordinates[1]>0)
          neighbors[UpLeft]=getRankOfProcCoord(subdomainCoordinates[0]-1,subdomainCoordinates[1]-1);
      if(subdomainCoordinates[0]>0 && subdomainCoordinates[1]<domainDecomposition[1]-1)
          neighbors[DownLeft]=getRankOfProcCoord(subdomainCoordinates[0]-1,subdomainCoordinates[1]+1);
      if(subdomainCoordinates[0]<domainDecomposition[0]-1 && subdomainCoordinates[1]>0)
          neighbors[UpRight]=getRankOfProcCoord(subdomainCoordinates[0]+1,subdomainCoordinates[1]-1);
      if(subdomainCoordinates[0]<domainDecomposition[0]-1 && subdomainCoordinates[1]<domainDecomposition[1]-1)
          neighbors[DownRight]=getRankOfProcCoord(subdomainCoordinates[0]+1,subdomainCoordinates[1]+1);

      localBegin=overlap;

      if(neighbors[Left]==-1)
      {
           localOrigin.x()+=overlap.x()*globalGrid.getSpaceSteps().x();
           localBegin.x()=0;
      }

      if(neighbors[Up]==-1)
      {
          localOrigin.y()+=overlap.y()*globalGrid.getSpaceSteps().y();
          localBegin.y()=0;
      }

      localGridSize=localSize;
      //Add Overlaps
      if(neighbors[Left]!=-1)
          localGridSize.x()+=overlap.x();
      if(neighbors[Right]!=-1)
          localGridSize.x()+=overlap.x();

      if(neighbors[Up]!=-1)
          localGridSize.y()+=overlap.y();
      if(neighbors[Down]!=-1)
          localGridSize.y()+=overlap.y();
  }
}

template< typename RealType, typename Device, typename Index >
void
DistributedMesh< Grid< 2, RealType, Device, Index > >::
setupGrid( GridType& grid )
{
   TNL_ASSERT_TRUE(isSet,"DistributedGrid is not set, but used by SetupGrid");
   grid.setOrigin( localOrigin );
   grid.setDimensions( localGridSize );
   //compute local proporions by sideefect
   grid.setSpaceSteps( spaceSteps );
   grid.SetDistMesh(this);
};

template< typename RealType, typename Device, typename Index >
String
DistributedMesh< Grid< 2, RealType, Device, Index > >::
printProcessCoords() const
{
   return convertToString(subdomainCoordinates[0])+String("-")+convertToString(subdomainCoordinates[1]);
};

template< typename RealType, typename Device, typename Index >
String
DistributedMesh< Grid< 2, RealType, Device, Index > >::
printProcessDistr() const
{
   return convertToString(domainDecomposition[0])+String("-")+convertToString(domainDecomposition[1]);
};  

template< typename RealType, typename Device, typename Index >
bool
DistributedMesh< Grid< 2, RealType, Device, Index > >::
isDistributed() const
{
   return this->distributed;
};

template< typename RealType, typename Device, typename Index >
const typename DistributedMesh< Grid< 2, RealType, Device, Index > >::CoordinatesType&
DistributedMesh< Grid< 2, RealType, Device, Index > >::
getOverlap() const
{
   return this->overlap;
};

template< typename RealType, typename Device, typename Index >
const int*
DistributedMesh< Grid< 2, RealType, Device, Index > >::
getNeighbors() const
{
   TNL_ASSERT_TRUE(isSet,"DistributedGrid is not set, but used by getNeighbors");
   return this->neighbors;
}

template< typename RealType, typename Device, typename Index >
const typename DistributedMesh< Grid< 2, RealType, Device, Index > >::CoordinatesType&
DistributedMesh< Grid< 2, RealType, Device, Index > >::
getLocalSize() const
{
   return this->localSize;
}

template< typename RealType, typename Device, typename Index >
const typename DistributedMesh< Grid< 2, RealType, Device, Index > >::CoordinatesType&
DistributedMesh< Grid< 2, RealType, Device, Index > >::
getGlobalSize() const
{
   return this->globalSize;
}

template< typename RealType, typename Device, typename Index >
const typename DistributedMesh< Grid< 2, RealType, Device, Index > >::CoordinatesType&
DistributedMesh< Grid< 2, RealType, Device, Index > >::
getGlobalBegin() const
{
   return this->globalBegin;
}

template< typename RealType, typename Device, typename Index >
const typename DistributedMesh< Grid< 2, RealType, Device, Index > >::CoordinatesType&
DistributedMesh< Grid< 2, RealType, Device, Index > >::
getLocalGridSize() const
{
   return this->localGridSize;
}

template< typename RealType, typename Device, typename Index >
const typename DistributedMesh< Grid< 2, RealType, Device, Index > >::CoordinatesType&
DistributedMesh< Grid< 2, RealType, Device, Index > >::
getLocalBegin() const
{
   return this->localBegin;
}

template< typename RealType, typename Device, typename Index >
void
DistributedMesh< Grid< 2, RealType, Device, Index > >::
writeProlog( Logger& logger ) const
{
   logger.writeParameter( "Domain decomposition:", this->getDomainDecomposition() );
}

template< typename RealType, typename Device, typename Index >
int
DistributedMesh< Grid< 2, RealType, Device, Index > >::
getRankOfProcCoord(int x, int y) const
{
   return y*domainDecomposition[0]+x;
}
         
      } //namespace DistributedMeshes
   } // namespace Meshes
} // namespace TNL
