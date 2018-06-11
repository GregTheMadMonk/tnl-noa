/***************************************************************************
                          DistributedGrid_3D.hpp  -  description
                             -------------------
    begin                : January 09, 2018
    copyright            : (C) 2018 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Meshes/DistributedMeshes/DistributedGrid_3D.h>

namespace TNL {
   namespace Meshes {
      namespace DistributedMeshes {

template< typename RealType, typename Device, typename Index >
DistributedMesh< Grid< 3, RealType, Device, Index > >::
DistributedMesh()
: isSet( false ) {};

template< typename RealType, typename Device, typename Index >     
void
DistributedMesh< Grid< 3, RealType, Device, Index > >::
configSetup( Config::ConfigDescription& config )
{
   config.addEntry< int >( "grid-domain-decomposition-x", "Number of grid subdomains along x-axis.", 0 );
   config.addEntry< int >( "grid-domain-decomposition-y", "Number of grid subdomains along y-axis.", 0 );
   config.addEntry< int >( "grid-domain-decomposition-z", "Number of grid subdomains along z-axis.", 0 );
}

template< typename RealType, typename Device, typename Index >     
bool
DistributedMesh< Grid< 3, RealType, Device, Index > >::
setup( const Config::ParameterContainer& parameters,
       const String& prefix )
{
   this->domainDecomposition.x() = parameters.getParameter< int >( "grid-domain-decomposition-x" );
   this->domainDecomposition.y() = parameters.getParameter< int >( "grid-domain-decomposition-y" );
   this->domainDecomposition.z() = parameters.getParameter< int >( "grid-domain-decomposition-z" );
   return true;
}      

template< typename RealType, typename Device, typename Index >     
void
DistributedMesh< Grid< 3, RealType, Device, Index > >::
setDomainDecomposition( const CoordinatesType& domainDecomposition )
{
   this->domainDecomposition = domainDecomposition;
}      

template< typename RealType, typename Device, typename Index >     
const typename DistributedMesh< Grid< 3, RealType, Device, Index > >::CoordinatesType&
DistributedMesh< Grid< 3, RealType, Device, Index > >::
getDomainDecomposition() const
{
   return this->domainDecomposition;
}      

template< typename RealType, typename Device, typename Index >     
   template< int EntityDimension >
Index
DistributedMesh< Grid< 3, RealType, Device, Index > >::
getEntitiesCount() const
{
   return this->globalGrid. template getEntitiesCount< EntityDimension >();
}

template< typename RealType, typename Device, typename Index >     
   template< typename Entity >
Index
DistributedMesh< Grid< 3, RealType, Device, Index > >::
getEntitiesCount() const
{
   return this->globalGrid. template getEntitiesCount< Entity >();
}

template< typename RealType, typename Device, typename Index >     
   template< typename CommunicatorType >
void
DistributedMesh< Grid< 3, RealType, Device, Index > >::
setGlobalGrid( const GridType &globalGrid,
               const CoordinatesType& overlap )
{
   this->globalGrid = globalGrid;
   isSet=true;           
   this->overlap=overlap;

   for (int i=0;i<26;i++)
        neighbors[i]=-1;

   Dimensions= GridType::getMeshDimension();
   spaceSteps=globalGrid.getSpaceSteps();

   distributed=false;



   if(CommunicatorType::IsInitialized())
   {
      rank=CommunicatorType::GetRank();
      this->nproc=CommunicatorType::GetSize();
      //use MPI only if have more than one process
      if(this->nproc>1)
      {
         distributed=true;
      }
   }

   if(!distributed)
   {
      //Without MPI
      subdomainCoordinates[0]=0;
      subdomainCoordinates[1]=0;
      subdomainCoordinates[2]=0;

      domainDecomposition[0]=1;
      domainDecomposition[1]=1;
      domainDecomposition[2]=1;               

      localOrigin=globalGrid.getOrigin();
      localSize=globalGrid.getDimensions();
      globalSize=globalGrid.getDimensions();
      localGridSize=localSize;
      globalBegin=CoordinatesType(0);
      return;
   }
   else
   {
      //With MPI
      //compute node distribution
      int dims[ 3 ];
      dims[ 0 ] = domainDecomposition[ 0 ];
      dims[ 1 ] = domainDecomposition[ 1 ];
      dims[ 2 ] = domainDecomposition[ 2 ];

      CommunicatorType::DimsCreate( nproc, 3, dims );
      domainDecomposition[ 0 ] = dims[ 0 ];
      domainDecomposition[ 1 ] = dims[ 1 ];
      domainDecomposition[ 2 ] = dims[ 2 ];

      subdomainCoordinates[ 2 ] =   rank / ( domainDecomposition[0] * domainDecomposition[1] );
      subdomainCoordinates[ 1 ] = ( rank % ( domainDecomposition[0] * domainDecomposition[1] ) ) / domainDecomposition[0];
      subdomainCoordinates[ 0 ] = ( rank % ( domainDecomposition[0] * domainDecomposition[1] ) ) % domainDecomposition[0];

      //compute local mesh size 
      globalSize=globalGrid.getDimensions();                
      numberOfLarger[0]=globalGrid.getDimensions().x()%domainDecomposition[0];
      numberOfLarger[1]=globalGrid.getDimensions().y()%domainDecomposition[1];
      numberOfLarger[2]=globalGrid.getDimensions().z()%domainDecomposition[2];

      localSize.x()=(globalGrid.getDimensions().x()/domainDecomposition[0]);
      localSize.y()=(globalGrid.getDimensions().y()/domainDecomposition[1]);
      localSize.z()=(globalGrid.getDimensions().z()/domainDecomposition[2]);

      if(numberOfLarger[0]>subdomainCoordinates[0])
         localSize.x()+=1;               
      if(numberOfLarger[1]>subdomainCoordinates[1])
         localSize.y()+=1;
      if(numberOfLarger[2]>subdomainCoordinates[2])
         localSize.z()+=1;

      if(numberOfLarger[0]>subdomainCoordinates[0])
         globalBegin.x()=subdomainCoordinates[0]*localSize.x();
      else
         globalBegin.x()=numberOfLarger[0]*(localSize.x()+1)+(subdomainCoordinates[0]-numberOfLarger[0])*localSize.x();

      if(numberOfLarger[1]>subdomainCoordinates[1])
         globalBegin.y()=subdomainCoordinates[1]*localSize.y();
      else
         globalBegin.y()=numberOfLarger[1]*(localSize.y()+1)+(subdomainCoordinates[1]-numberOfLarger[1])*localSize.y();

      if(numberOfLarger[2]>subdomainCoordinates[2])
         globalBegin.z()=subdomainCoordinates[2]*localSize.z();
      else
         globalBegin.z()=numberOfLarger[2]*(localSize.z()+1)+(subdomainCoordinates[2]-numberOfLarger[2])*localSize.z();

      localOrigin=globalGrid.getOrigin()+TNL::Containers::tnlDotProduct(globalGrid.getSpaceSteps(),globalBegin-overlap);

      //nearnodes
      //X Y Z
      if(subdomainCoordinates[0]>0)
         neighbors[West]=getRankOfProcCoord(subdomainCoordinates[0]-1,subdomainCoordinates[1],subdomainCoordinates[2]);               
      if(subdomainCoordinates[0]<domainDecomposition[0]-1)
         neighbors[East]=getRankOfProcCoord(subdomainCoordinates[0]+1,subdomainCoordinates[1],subdomainCoordinates[2]);
      if(subdomainCoordinates[1]>0)
         neighbors[North]=getRankOfProcCoord(subdomainCoordinates[0],subdomainCoordinates[1]-1,subdomainCoordinates[2]);
      if(subdomainCoordinates[1]<domainDecomposition[1]-1)
         neighbors[South]=getRankOfProcCoord(subdomainCoordinates[0],subdomainCoordinates[1]+1,subdomainCoordinates[2]);
      if(subdomainCoordinates[2]>0)
         neighbors[Bottom]=getRankOfProcCoord(subdomainCoordinates[0],subdomainCoordinates[1],subdomainCoordinates[2]-1);
      if(subdomainCoordinates[2]<domainDecomposition[2]-1)
         neighbors[Top]=getRankOfProcCoord(subdomainCoordinates[0],subdomainCoordinates[1],subdomainCoordinates[2]+1);

      //XY
      if(subdomainCoordinates[0]>0 && subdomainCoordinates[1]>0)
         neighbors[NorthWest]=getRankOfProcCoord(subdomainCoordinates[0]-1,subdomainCoordinates[1]-1,subdomainCoordinates[2]);
      if(subdomainCoordinates[0]<domainDecomposition[0]-1 && subdomainCoordinates[1]>0)
         neighbors[NorthEast]=getRankOfProcCoord(subdomainCoordinates[0]+1,subdomainCoordinates[1]-1,subdomainCoordinates[2]);
      if(subdomainCoordinates[0]>0 && subdomainCoordinates[1]<domainDecomposition[1]-1)
         neighbors[SouthWest]=getRankOfProcCoord(subdomainCoordinates[0]-1,subdomainCoordinates[1]+1,subdomainCoordinates[2]);
      if(subdomainCoordinates[0]<domainDecomposition[0]-1 && subdomainCoordinates[1]<domainDecomposition[1]-1)
         neighbors[SouthEast]=getRankOfProcCoord(subdomainCoordinates[0]+1,subdomainCoordinates[1]+1,subdomainCoordinates[2]);             
      //XZ
      if(subdomainCoordinates[0]>0 && subdomainCoordinates[2]>0)
         neighbors[BottomWest]=getRankOfProcCoord(subdomainCoordinates[0]-1,subdomainCoordinates[1],subdomainCoordinates[2]-1);
      if(subdomainCoordinates[0]<domainDecomposition[0]-1 && subdomainCoordinates[2]>0)
         neighbors[BottomEast]=getRankOfProcCoord(subdomainCoordinates[0]+1,subdomainCoordinates[1],subdomainCoordinates[2]-1); 
      if(subdomainCoordinates[0]>0 && subdomainCoordinates[2]<domainDecomposition[2]-1)
         neighbors[TopWest]=getRankOfProcCoord(subdomainCoordinates[0]-1,subdomainCoordinates[1],subdomainCoordinates[2]+1);
      if(subdomainCoordinates[0]<domainDecomposition[0]-1 && subdomainCoordinates[2]<domainDecomposition[2]-1)
         neighbors[TopEast]=getRankOfProcCoord(subdomainCoordinates[0]+1,subdomainCoordinates[1],subdomainCoordinates[2]+1);
      //YZ
      if(subdomainCoordinates[1]>0 && subdomainCoordinates[2]>0)
         neighbors[BottomNorth]=getRankOfProcCoord(subdomainCoordinates[0],subdomainCoordinates[1]-1,subdomainCoordinates[2]-1);
      if(subdomainCoordinates[1]<domainDecomposition[1]-1 && subdomainCoordinates[2]>0)
         neighbors[BottomSouth]=getRankOfProcCoord(subdomainCoordinates[0],subdomainCoordinates[1]+1,subdomainCoordinates[2]-1);
      if(subdomainCoordinates[1]>0 && subdomainCoordinates[2]<domainDecomposition[2]-1)
         neighbors[TopNorth]=getRankOfProcCoord(subdomainCoordinates[0],subdomainCoordinates[1]-1,subdomainCoordinates[2]+1);               
      if(subdomainCoordinates[1]<domainDecomposition[1]-1 && subdomainCoordinates[2]<domainDecomposition[2]-1)
         neighbors[TopSouth]=getRankOfProcCoord(subdomainCoordinates[0],subdomainCoordinates[1]+1,subdomainCoordinates[2]+1);
      //XYZ
      if(subdomainCoordinates[0]>0 && subdomainCoordinates[1]>0 && subdomainCoordinates[2]>0 )
         neighbors[BottomNorthWest]=getRankOfProcCoord(subdomainCoordinates[0]-1,subdomainCoordinates[1]-1,subdomainCoordinates[2]-1);
      if(subdomainCoordinates[0]<domainDecomposition[0]-1 && subdomainCoordinates[1]>0 && subdomainCoordinates[2]>0 )
         neighbors[BottomNorthEast]=getRankOfProcCoord(subdomainCoordinates[0]+1,subdomainCoordinates[1]-1,subdomainCoordinates[2]-1);
      if(subdomainCoordinates[0]>0 && subdomainCoordinates[1]<domainDecomposition[1]-1 && subdomainCoordinates[2]>0 )
         neighbors[BottomSouthWest]=getRankOfProcCoord(subdomainCoordinates[0]-1,subdomainCoordinates[1]+1,subdomainCoordinates[2]-1);
      if(subdomainCoordinates[0]<domainDecomposition[0]-1 && subdomainCoordinates[1]<domainDecomposition[1]-1 && subdomainCoordinates[2]>0 )
         neighbors[BottomSouthEast]=getRankOfProcCoord(subdomainCoordinates[0]+1,subdomainCoordinates[1]+1,subdomainCoordinates[2]-1);
      if(subdomainCoordinates[0]>0 && subdomainCoordinates[1]>0 && subdomainCoordinates[2]<domainDecomposition[2]-1 )
         neighbors[TopNorthWest]=getRankOfProcCoord(subdomainCoordinates[0]-1,subdomainCoordinates[1]-1,subdomainCoordinates[2]+1);
      if(subdomainCoordinates[0]<domainDecomposition[0]-1 && subdomainCoordinates[1]>0 && subdomainCoordinates[2]<domainDecomposition[2]-1 )
         neighbors[TopNorthEast]=getRankOfProcCoord(subdomainCoordinates[0]+1,subdomainCoordinates[1]-1,subdomainCoordinates[2]+1);
      if(subdomainCoordinates[0]>0 && subdomainCoordinates[1]<domainDecomposition[1]-1 && subdomainCoordinates[2]<domainDecomposition[2]-1 )
         neighbors[TopSouthWest]=getRankOfProcCoord(subdomainCoordinates[0]-1,subdomainCoordinates[1]+1,subdomainCoordinates[2]+1);
      if(subdomainCoordinates[0]<domainDecomposition[0]-1 && subdomainCoordinates[1]<domainDecomposition[1]-1 && subdomainCoordinates[2]<domainDecomposition[2]-1 )
         neighbors[TopSouthEast]=getRankOfProcCoord(subdomainCoordinates[0]+1,subdomainCoordinates[1]+1,subdomainCoordinates[2]+1);   


      localBegin=overlap;

      if(neighbors[West]==-1)
      {
         localOrigin.x()+=overlap.x()*globalGrid.getSpaceSteps().x();
         localBegin.x()=0;
      }
      if(neighbors[North]==-1)
      {
         localOrigin.y()+=overlap.y()*globalGrid.getSpaceSteps().y();
         localBegin.y()=0;
      }
      if(neighbors[Bottom]==-1)
      {
         localOrigin.z()+=overlap.z()*globalGrid.getSpaceSteps().z();
         localBegin.z()=0;
      }

      localGridSize=localSize;

      if(neighbors[West]!=-1)
         localGridSize.x()+=overlap.x();
      if(neighbors[East]!=-1)
         localGridSize.x()+=overlap.x();

      if(neighbors[North]!=-1)
         localGridSize.y()+=overlap.y();
      if(neighbors[South]!=-1)
         localGridSize.y()+=overlap.y();

      if(neighbors[Bottom]!=-1)
         localGridSize.z()+=overlap.z();
      if(neighbors[Top]!=-1)
         localGridSize.z()+=overlap.z();
   }                     
}

template< typename RealType, typename Device, typename Index >     
void
DistributedMesh< Grid< 3, RealType, Device, Index > >::
setupGrid( GridType& grid)
{
   TNL_ASSERT_TRUE(isSet,"DistributedGrid is not set, but used by SetupGrid");
   grid.setOrigin(localOrigin);
   grid.setDimensions(localGridSize);
   //compute local proportions by side efect
   grid.setSpaceSteps(spaceSteps);
   grid.SetDistMesh(this);
};

template< typename RealType, typename Device, typename Index >     
String
DistributedMesh< Grid< 3, RealType, Device, Index > >::
printProcessCoords() const
{
   return convertToString(subdomainCoordinates[0])+String("-")+convertToString(subdomainCoordinates[1])+String("-")+convertToString(subdomainCoordinates[2]);
};

template< typename RealType, typename Device, typename Index >     
String
DistributedMesh< Grid< 3, RealType, Device, Index > >::
printProcessDistr() const
{
   return convertToString(domainDecomposition[0])+String("-")+convertToString(domainDecomposition[1])+String("-")+convertToString(domainDecomposition[2]);
};  

template< typename RealType, typename Device, typename Index >     
bool
DistributedMesh< Grid< 3, RealType, Device, Index > >::
isDistributed() const
{
   return this->distributed;
};

template< typename RealType, typename Device, typename Index >     
const typename DistributedMesh< Grid< 3, RealType, Device, Index > >::CoordinatesType&
DistributedMesh< Grid< 3, RealType, Device, Index > >::
getOverlap() const
{
   return this->overlap;
};

template< typename RealType, typename Device, typename Index >     
const int*
DistributedMesh< Grid< 3, RealType, Device, Index > >::
getNeighbors() const
{
   TNL_ASSERT_TRUE(isSet,"DistributedGrid is not set, but used by getNeighbors");
   return this->neighbors;
}

template< typename RealType, typename Device, typename Index >     
const typename DistributedMesh< Grid< 3, RealType, Device, Index > >::CoordinatesType&
DistributedMesh< Grid< 3, RealType, Device, Index > >::
getLocalSize() const
{
   return this->localSize;
}

template< typename RealType, typename Device, typename Index >     
const typename DistributedMesh< Grid< 3, RealType, Device, Index > >::CoordinatesType&
DistributedMesh< Grid< 3, RealType, Device, Index > >::
getLocalGridSize() const
{
   return this->localGridSize;
}

template< typename RealType, typename Device, typename Index >     
const typename DistributedMesh< Grid< 3, RealType, Device, Index > >::CoordinatesType&
DistributedMesh< Grid< 3, RealType, Device, Index > >::
getLocalBegin() const
{
   return this->localBegin;
}

template< typename RealType, typename Device, typename Index >     
const typename DistributedMesh< Grid< 3, RealType, Device, Index > >::CoordinatesType&
DistributedMesh< Grid< 3, RealType, Device, Index > >::
getGlobalSize() const
{
   return this->globalSize;
}

template< typename RealType, typename Device, typename Index >     
const typename DistributedMesh< Grid< 3, RealType, Device, Index > >::CoordinatesType&
DistributedMesh< Grid< 3, RealType, Device, Index > >::
getGlobalBegin() const
{
   return this->globalBegin;
}

template< typename RealType, typename Device, typename Index >     
void
DistributedMesh< Grid< 3, RealType, Device, Index > >::
writeProlog( Logger& logger )
{
   logger.writeParameter( "Domain decomposition:", this->getDomainDecomposition() );
}           

template< typename RealType, typename Device, typename Index >     
int
DistributedMesh< Grid< 3, RealType, Device, Index > >::
getRankOfProcCoord( int x, int y, int z ) const
{
   return z*domainDecomposition[0]*domainDecomposition[1]+y*domainDecomposition[0]+x;
}
         
         
      } //namespace DistributedMeshes
   } // namespace Meshes
} // namespace TNL
