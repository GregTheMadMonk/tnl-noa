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

/*template< typename RealType, typename Device, typename Index >     
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
}*/

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
   this->isSet=true;
   this->overlap=overlap;

   for( int i=0; i<8; i++ )
      neighbors[i]=-1;

   this->Dimensions= GridType::getMeshDimension();
   this->spaceSteps=globalGrid.getSpaceSteps();
   this->distributed=false;

   if( CommunicatorType::IsInitialized() )
   {
      this->rank=CommunicatorType::GetRank();
      this->nproc=CommunicatorType::GetSize();
      //use MPI only if have more than one process
      if(this->nproc>1)
      {
         this->distributed=true;
      }
   }

   if( !this->distributed )
   {
      this->subdomainCoordinates[0]=0;
      this->subdomainCoordinates[1]=0;
      this->domainDecomposition[0]=1;
      this->domainDecomposition[1]=1;
      this->localOrigin=globalGrid.getOrigin();
      this->localGridSize=globalGrid.getDimensions();
      this->localSize=globalGrid.getDimensions();
      this->globalBegin=CoordinatesType(0);
      this->localBegin.x()=0;
      this->localBegin.y()=0;

      return;
   }
   else
   {
      int numberOfLarger[2];
      //compute node distribution
      int dims[ 2 ];
      dims[ 0 ] = this->domainDecomposition[ 0 ];
      dims[ 1 ] = this->domainDecomposition[ 1 ];

      CommunicatorType::DimsCreate( this->nproc, 2, dims );
      this->domainDecomposition[ 0 ] = dims[ 0 ];
      this->domainDecomposition[ 1 ] = dims[ 1 ];

      this->subdomainCoordinates[ 0 ] = this->rank % this->domainDecomposition[ 0 ];
      this->subdomainCoordinates[ 1 ] = this->rank / this->domainDecomposition[ 0 ];        

      //compute local mesh size            
      numberOfLarger[0]=globalGrid.getDimensions().x()%this->domainDecomposition[0];
      numberOfLarger[1]=globalGrid.getDimensions().y()%this->domainDecomposition[1];

      this->localSize.x()=(globalGrid.getDimensions().x()/this->domainDecomposition[0]);
      this->localSize.y()=(globalGrid.getDimensions().y()/this->domainDecomposition[1]);

      if(numberOfLarger[0]>this->subdomainCoordinates[0])
           this->localSize.x()+=1;               
      if(numberOfLarger[1]>this->subdomainCoordinates[1])
          this->localSize.y()+=1;

      if(numberOfLarger[0]>this->subdomainCoordinates[0])
          this->globalBegin.x()=this->subdomainCoordinates[0]*this->localSize.x();
      else
          this->globalBegin.x()=numberOfLarger[0]*(this->localSize.x()+1)+(this->subdomainCoordinates[0]-numberOfLarger[0])*this->localSize.x();

      if(numberOfLarger[1]>this->subdomainCoordinates[1])
          this->globalBegin.y()=this->subdomainCoordinates[1]*this->localSize.y();

      else
          this->globalBegin.y()=numberOfLarger[1]*(this->localSize.y()+1)+(this->subdomainCoordinates[1]-numberOfLarger[1])*this->localSize.y();

      this->localOrigin=globalGrid.getOrigin()+TNL::Containers::tnlDotProduct(globalGrid.getSpaceSteps(),this->globalBegin-this->overlap);

      //nearnodes
      if(this->subdomainCoordinates[0]>0)
          neighbors[Left]=getRankOfProcCoord(this->subdomainCoordinates[0]-1,this->subdomainCoordinates[1]);
      if(this->subdomainCoordinates[0]<this->domainDecomposition[0]-1)
          neighbors[Right]=getRankOfProcCoord(this->subdomainCoordinates[0]+1,this->subdomainCoordinates[1]);
      if(this->subdomainCoordinates[1]>0)
          neighbors[Up]=getRankOfProcCoord(this->subdomainCoordinates[0],this->subdomainCoordinates[1]-1);
      if(this->subdomainCoordinates[1]<this->domainDecomposition[1]-1)
          neighbors[Down]=getRankOfProcCoord(this->subdomainCoordinates[0],this->subdomainCoordinates[1]+1);
      if(this->subdomainCoordinates[0]>0 && this->subdomainCoordinates[1]>0)
          neighbors[UpLeft]=getRankOfProcCoord(this->subdomainCoordinates[0]-1,this->subdomainCoordinates[1]-1);
      if(this->subdomainCoordinates[0]>0 && this->subdomainCoordinates[1]<this->domainDecomposition[1]-1)
          neighbors[DownLeft]=getRankOfProcCoord(this->subdomainCoordinates[0]-1,this->subdomainCoordinates[1]+1);
      if(this->subdomainCoordinates[0]<this->domainDecomposition[0]-1 && this->subdomainCoordinates[1]>0)
          neighbors[UpRight]=getRankOfProcCoord(this->subdomainCoordinates[0]+1,this->subdomainCoordinates[1]-1);
      if(this->subdomainCoordinates[0]<this->domainDecomposition[0]-1 && this->subdomainCoordinates[1]<this->domainDecomposition[1]-1)
          neighbors[DownRight]=getRankOfProcCoord(this->subdomainCoordinates[0]+1,this->subdomainCoordinates[1]+1);

      this->localBegin=this->overlap;

      if(neighbors[Left]==-1)
      {
           this->localOrigin.x()+=this->overlap.x()*globalGrid.getSpaceSteps().x();
           this->localBegin.x()=0;
      }

      if(neighbors[Up]==-1)
      {
          this->localOrigin.y()+=this->overlap.y()*globalGrid.getSpaceSteps().y();
          this->localBegin.y()=0;
      }

      this->localGridSize=this->localSize;
      //Add Overlaps
      if(neighbors[Left]!=-1)
          this->localGridSize.x()+=this->overlap.x();
      if(neighbors[Right]!=-1)
          this->localGridSize.x()+=this->overlap.x();

      if(neighbors[Up]!=-1)
          this->localGridSize.y()+=this->overlap.y();
      if(neighbors[Down]!=-1)
          this->localGridSize.y()+=this->overlap.y();
  }
}

template< typename RealType, typename Device, typename Index >
void
DistributedMesh< Grid< 2, RealType, Device, Index > >::
setupGrid( GridType& grid )
{
   TNL_ASSERT_TRUE(this->isSet,"DistributedGrid is not set, but used by SetupGrid");
   grid.setOrigin( this->localOrigin );
   grid.setDimensions( this->localGridSize );
   //compute local proporions by sideefect
   grid.setSpaceSteps( this->spaceSteps );
   grid.SetDistMesh(this);
};

template< typename RealType, typename Device, typename Index >
String
DistributedMesh< Grid< 2, RealType, Device, Index > >::
printProcessCoords() const
{
   return convertToString(this->subdomainCoordinates[0])+String("-")+convertToString(this->subdomainCoordinates[1]);
};

template< typename RealType, typename Device, typename Index >
String
DistributedMesh< Grid< 2, RealType, Device, Index > >::
printProcessDistr() const
{
   return convertToString(this->domainDecomposition[0])+String("-")+convertToString(this->domainDecomposition[1]);
};  

template< typename RealType, typename Device, typename Index >
const int*
DistributedMesh< Grid< 2, RealType, Device, Index > >::
getNeighbors() const
{
   TNL_ASSERT_TRUE(this->isSet,"DistributedGrid is not set, but used by getNeighbors");
   return this->neighbors;
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
   return y*this->domainDecomposition[0]+x;
}
         
      } //namespace DistributedMeshes
   } // namespace Meshes
} // namespace TNL
