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

/*template< typename RealType, typename Device, typename Index >     
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
}*/

template< typename RealType, typename Device, typename Index >     
   template< typename CommunicatorType >
void
DistributedMesh< Grid< 3, RealType, Device, Index > >::
setGlobalGrid( const GridType &globalGrid,
               const CoordinatesType& overlap )
{
   this->globalGrid = globalGrid;
   this->isSet=true;           
   this->overlap=overlap;

   for (int i=0;i<26;i++)
        neighbors[i]=-1;

   this->Dimensions= GridType::getMeshDimension();
   this->spaceSteps=globalGrid.getSpaceSteps();

   this->distributed=false;



   if(CommunicatorType::IsInitialized())
   {
      this->rank=CommunicatorType::GetRank();
      this->nproc=CommunicatorType::GetSize();
      //use MPI only if have more than one process
      if(this->nproc>1)
      {
         this->distributed=true;
      }
   }

   if(!this->distributed)
   {
      //Without MPI
      this->subdomainCoordinates[0]=0;
      this->subdomainCoordinates[1]=0;
      this->subdomainCoordinates[2]=0;

      this->domainDecomposition[0]=1;
      this->domainDecomposition[1]=1;
      this->domainDecomposition[2]=1;               

      this->localOrigin=globalGrid.getOrigin();
      this->localSize=globalGrid.getDimensions();
      this->localGridSize=this->localSize;
      this->globalBegin=CoordinatesType(0);
      return;
   }
   else
   {
      //With MPI
      //compute node distribution
      int dims[ 3 ];
      int numberOfLarger[3];
      dims[ 0 ] = this->domainDecomposition[ 0 ];
      dims[ 1 ] = this->domainDecomposition[ 1 ];
      dims[ 2 ] = this->domainDecomposition[ 2 ];

      CommunicatorType::DimsCreate( this->nproc, 3, dims );
      this->domainDecomposition[ 0 ] = dims[ 0 ];
      this->domainDecomposition[ 1 ] = dims[ 1 ];
      this->domainDecomposition[ 2 ] = dims[ 2 ];

      this->subdomainCoordinates[ 2 ] =   this->rank / ( this->domainDecomposition[0] * this->domainDecomposition[1] );
      this->subdomainCoordinates[ 1 ] = ( this->rank % ( this->domainDecomposition[0] * this->domainDecomposition[1] ) ) / this->domainDecomposition[0];
      this->subdomainCoordinates[ 0 ] = ( this->rank % ( this->domainDecomposition[0] * this->domainDecomposition[1] ) ) % this->domainDecomposition[0];

      //compute local mesh size               
      numberOfLarger[0]=globalGrid.getDimensions().x()%this->domainDecomposition[0];
      numberOfLarger[1]=globalGrid.getDimensions().y()%this->domainDecomposition[1];
      numberOfLarger[2]=globalGrid.getDimensions().z()%this->domainDecomposition[2];

      this->localSize.x()=(globalGrid.getDimensions().x()/this->domainDecomposition[0]);
      this->localSize.y()=(globalGrid.getDimensions().y()/this->domainDecomposition[1]);
      this->localSize.z()=(globalGrid.getDimensions().z()/this->domainDecomposition[2]);

      if(numberOfLarger[0]>this->subdomainCoordinates[0])
         this->localSize.x()+=1;               
      if(numberOfLarger[1]>this->subdomainCoordinates[1])
         this->localSize.y()+=1;
      if(numberOfLarger[2]>this->subdomainCoordinates[2])
         this->localSize.z()+=1;

      if(numberOfLarger[0]>this->subdomainCoordinates[0])
         this->globalBegin.x()=this->subdomainCoordinates[0]*this->localSize.x();
      else
         this->globalBegin.x()=numberOfLarger[0]*(this->localSize.x()+1)+(this->subdomainCoordinates[0]-numberOfLarger[0])*this->localSize.x();

      if(numberOfLarger[1]>this->subdomainCoordinates[1])
         this->globalBegin.y()=this->subdomainCoordinates[1]*this->localSize.y();
      else
         this->globalBegin.y()=numberOfLarger[1]*(this->localSize.y()+1)+(this->subdomainCoordinates[1]-numberOfLarger[1])*this->localSize.y();

      if(numberOfLarger[2]>this->subdomainCoordinates[2])
         this->globalBegin.z()=this->subdomainCoordinates[2]*this->localSize.z();
      else
         this->globalBegin.z()=numberOfLarger[2]*(this->localSize.z()+1)+(this->subdomainCoordinates[2]-numberOfLarger[2])*this->localSize.z();

      this->localOrigin=globalGrid.getOrigin()+TNL::Containers::tnlDotProduct(globalGrid.getSpaceSteps(),this->globalBegin-this->overlap);

      //nearnodes
      //X Y Z
      if(this->subdomainCoordinates[0]>0)
         neighbors[West]=getRankOfProcCoord(this->subdomainCoordinates[0]-1,this->subdomainCoordinates[1],this->subdomainCoordinates[2]);               
      if(this->subdomainCoordinates[0]<this->domainDecomposition[0]-1)
         neighbors[East]=getRankOfProcCoord(this->subdomainCoordinates[0]+1,this->subdomainCoordinates[1],this->subdomainCoordinates[2]);
      if(this->subdomainCoordinates[1]>0)
         neighbors[North]=getRankOfProcCoord(this->subdomainCoordinates[0],this->subdomainCoordinates[1]-1,this->subdomainCoordinates[2]);
      if(this->subdomainCoordinates[1]<this->domainDecomposition[1]-1)
         neighbors[South]=getRankOfProcCoord(this->subdomainCoordinates[0],this->subdomainCoordinates[1]+1,this->subdomainCoordinates[2]);
      if(this->subdomainCoordinates[2]>0)
         neighbors[Bottom]=getRankOfProcCoord(this->subdomainCoordinates[0],this->subdomainCoordinates[1],this->subdomainCoordinates[2]-1);
      if(this->subdomainCoordinates[2]<this->domainDecomposition[2]-1)
         neighbors[Top]=getRankOfProcCoord(this->subdomainCoordinates[0],this->subdomainCoordinates[1],this->subdomainCoordinates[2]+1);

      //XY
      if(this->subdomainCoordinates[0]>0 && this->subdomainCoordinates[1]>0)
         neighbors[NorthWest]=getRankOfProcCoord(this->subdomainCoordinates[0]-1,this->subdomainCoordinates[1]-1,this->subdomainCoordinates[2]);
      if(this->subdomainCoordinates[0]<this->domainDecomposition[0]-1 && this->subdomainCoordinates[1]>0)
         neighbors[NorthEast]=getRankOfProcCoord(this->subdomainCoordinates[0]+1,this->subdomainCoordinates[1]-1,this->subdomainCoordinates[2]);
      if(this->subdomainCoordinates[0]>0 && this->subdomainCoordinates[1]<this->domainDecomposition[1]-1)
         neighbors[SouthWest]=getRankOfProcCoord(this->subdomainCoordinates[0]-1,this->subdomainCoordinates[1]+1,this->subdomainCoordinates[2]);
      if(this->subdomainCoordinates[0]<this->domainDecomposition[0]-1 && this->subdomainCoordinates[1]<this->domainDecomposition[1]-1)
         neighbors[SouthEast]=getRankOfProcCoord(this->subdomainCoordinates[0]+1,this->subdomainCoordinates[1]+1,this->subdomainCoordinates[2]);             
      //XZ
      if(this->subdomainCoordinates[0]>0 && this->subdomainCoordinates[2]>0)
         neighbors[BottomWest]=getRankOfProcCoord(this->subdomainCoordinates[0]-1,this->subdomainCoordinates[1],this->subdomainCoordinates[2]-1);
      if(this->subdomainCoordinates[0]<this->domainDecomposition[0]-1 && this->subdomainCoordinates[2]>0)
         neighbors[BottomEast]=getRankOfProcCoord(this->subdomainCoordinates[0]+1,this->subdomainCoordinates[1],this->subdomainCoordinates[2]-1); 
      if(this->subdomainCoordinates[0]>0 && this->subdomainCoordinates[2]<this->domainDecomposition[2]-1)
         neighbors[TopWest]=getRankOfProcCoord(this->subdomainCoordinates[0]-1,this->subdomainCoordinates[1],this->subdomainCoordinates[2]+1);
      if(this->subdomainCoordinates[0]<this->domainDecomposition[0]-1 && this->subdomainCoordinates[2]<this->domainDecomposition[2]-1)
         neighbors[TopEast]=getRankOfProcCoord(this->subdomainCoordinates[0]+1,this->subdomainCoordinates[1],this->subdomainCoordinates[2]+1);
      //YZ
      if(this->subdomainCoordinates[1]>0 && this->subdomainCoordinates[2]>0)
         neighbors[BottomNorth]=getRankOfProcCoord(this->subdomainCoordinates[0],this->subdomainCoordinates[1]-1,this->subdomainCoordinates[2]-1);
      if(this->subdomainCoordinates[1]<this->domainDecomposition[1]-1 && this->subdomainCoordinates[2]>0)
         neighbors[BottomSouth]=getRankOfProcCoord(this->subdomainCoordinates[0],this->subdomainCoordinates[1]+1,this->subdomainCoordinates[2]-1);
      if(this->subdomainCoordinates[1]>0 && this->subdomainCoordinates[2]<this->domainDecomposition[2]-1)
         neighbors[TopNorth]=getRankOfProcCoord(this->subdomainCoordinates[0],this->subdomainCoordinates[1]-1,this->subdomainCoordinates[2]+1);               
      if(this->subdomainCoordinates[1]<this->domainDecomposition[1]-1 && this->subdomainCoordinates[2]<this->domainDecomposition[2]-1)
         neighbors[TopSouth]=getRankOfProcCoord(this->subdomainCoordinates[0],this->subdomainCoordinates[1]+1,this->subdomainCoordinates[2]+1);
      //XYZ
      if(this->subdomainCoordinates[0]>0 && this->subdomainCoordinates[1]>0 && this->subdomainCoordinates[2]>0 )
         neighbors[BottomNorthWest]=getRankOfProcCoord(this->subdomainCoordinates[0]-1,this->subdomainCoordinates[1]-1,this->subdomainCoordinates[2]-1);
      if(this->subdomainCoordinates[0]<this->domainDecomposition[0]-1 && this->subdomainCoordinates[1]>0 && this->subdomainCoordinates[2]>0 )
         neighbors[BottomNorthEast]=getRankOfProcCoord(this->subdomainCoordinates[0]+1,this->subdomainCoordinates[1]-1,this->subdomainCoordinates[2]-1);
      if(this->subdomainCoordinates[0]>0 && this->subdomainCoordinates[1]<this->domainDecomposition[1]-1 && this->subdomainCoordinates[2]>0 )
         neighbors[BottomSouthWest]=getRankOfProcCoord(this->subdomainCoordinates[0]-1,this->subdomainCoordinates[1]+1,this->subdomainCoordinates[2]-1);
      if(this->subdomainCoordinates[0]<this->domainDecomposition[0]-1 && this->subdomainCoordinates[1]<this->domainDecomposition[1]-1 && this->subdomainCoordinates[2]>0 )
         neighbors[BottomSouthEast]=getRankOfProcCoord(this->subdomainCoordinates[0]+1,this->subdomainCoordinates[1]+1,this->subdomainCoordinates[2]-1);
      if(this->subdomainCoordinates[0]>0 && this->subdomainCoordinates[1]>0 && this->subdomainCoordinates[2]<this->domainDecomposition[2]-1 )
         neighbors[TopNorthWest]=getRankOfProcCoord(this->subdomainCoordinates[0]-1,this->subdomainCoordinates[1]-1,this->subdomainCoordinates[2]+1);
      if(this->subdomainCoordinates[0]<this->domainDecomposition[0]-1 && this->subdomainCoordinates[1]>0 && this->subdomainCoordinates[2]<this->domainDecomposition[2]-1 )
         neighbors[TopNorthEast]=getRankOfProcCoord(this->subdomainCoordinates[0]+1,this->subdomainCoordinates[1]-1,this->subdomainCoordinates[2]+1);
      if(this->subdomainCoordinates[0]>0 && this->subdomainCoordinates[1]<this->domainDecomposition[1]-1 && this->subdomainCoordinates[2]<this->domainDecomposition[2]-1 )
         neighbors[TopSouthWest]=getRankOfProcCoord(this->subdomainCoordinates[0]-1,this->subdomainCoordinates[1]+1,this->subdomainCoordinates[2]+1);
      if(this->subdomainCoordinates[0]<this->domainDecomposition[0]-1 && this->subdomainCoordinates[1]<this->domainDecomposition[1]-1 && this->subdomainCoordinates[2]<this->domainDecomposition[2]-1 )
         neighbors[TopSouthEast]=getRankOfProcCoord(this->subdomainCoordinates[0]+1,this->subdomainCoordinates[1]+1,this->subdomainCoordinates[2]+1);   


      this->localBegin=this->overlap;

      if(neighbors[West]==-1)
      {
         this->localOrigin.x()+=this->overlap.x()*globalGrid.getSpaceSteps().x();
         this->localBegin.x()=0;
      }
      if(neighbors[North]==-1)
      {
         this->localOrigin.y()+=this->overlap.y()*globalGrid.getSpaceSteps().y();
         this->localBegin.y()=0;
      }
      if(neighbors[Bottom]==-1)
      {
         this->localOrigin.z()+=this->overlap.z()*globalGrid.getSpaceSteps().z();
         this->localBegin.z()=0;
      }

      this->localGridSize=this->localSize;

      if(neighbors[West]!=-1)
         this->localGridSize.x()+=this->overlap.x();
      if(neighbors[East]!=-1)
         this->localGridSize.x()+=this->overlap.x();

      if(neighbors[North]!=-1)
         this->localGridSize.y()+=this->overlap.y();
      if(neighbors[South]!=-1)
         this->localGridSize.y()+=this->overlap.y();

      if(neighbors[Bottom]!=-1)
         this->localGridSize.z()+=this->overlap.z();
      if(neighbors[Top]!=-1)
         this->localGridSize.z()+=this->overlap.z();
   }                     
}

template< typename RealType, typename Device, typename Index >     
void
DistributedMesh< Grid< 3, RealType, Device, Index > >::
setupGrid( GridType& grid)
{
   TNL_ASSERT_TRUE(this->isSet,"DistributedGrid is not set, but used by SetupGrid");
   grid.setOrigin(this->localOrigin);
   grid.setDimensions(this->localGridSize);
   //compute local proportions by side efect
   grid.setSpaceSteps(this->spaceSteps);
   grid.SetDistMesh(this);
};

template< typename RealType, typename Device, typename Index >     
String
DistributedMesh< Grid< 3, RealType, Device, Index > >::
printProcessCoords() const
{
   return convertToString(this->subdomainCoordinates[0])+String("-")+convertToString(this->subdomainCoordinates[1])+String("-")+convertToString(this->subdomainCoordinates[2]);
};

template< typename RealType, typename Device, typename Index >     
String
DistributedMesh< Grid< 3, RealType, Device, Index > >::
printProcessDistr() const
{
   return convertToString(this->domainDecomposition[0])+String("-")+convertToString(this->domainDecomposition[1])+String("-")+convertToString(this->domainDecomposition[2]);
};  

template< typename RealType, typename Device, typename Index >     
const int*
DistributedMesh< Grid< 3, RealType, Device, Index > >::
getNeighbors() const
{
   TNL_ASSERT_TRUE(this->isSet,"DistributedGrid is not set, but used by getNeighbors");
   return this->neighbors;
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
   return z*this->domainDecomposition[0]*this->domainDecomposition[1]+y*this->domainDecomposition[0]+x;
}
         
         
      } //namespace DistributedMeshes
   } // namespace Meshes
} // namespace TNL
