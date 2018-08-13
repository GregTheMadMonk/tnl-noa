/***************************************************************************
                          DistributedGrid_2D.hpp  -  description
                             -------------------
    begin                : January 09, 2018
    copyright            : (C) 2018 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <cstdlib>

#include <TNL/Meshes/DistributedMeshes/DistributedGrid_Base.h>
#include <TNL/Communicators/MpiCommunicator.h>

#pragma once

namespace TNL {
   namespace Meshes { 
      namespace DistributedMeshes {

template< typename Real, typename Device, typename Index >
bool
DistributedMesh< Grid< 2, Real, Device, Index > >::
setup( const Config::ParameterContainer& parameters,
       const String& prefix )
{
   this->domainDecomposition.x() = parameters.getParameter< int >( "grid-domain-decomposition-x" );
   this->domainDecomposition.y() = parameters.getParameter< int >( "grid-domain-decomposition-y" );
   return true;
}      


/*template< typename RealType, typename Device, typename Index >
template< typename CommunicatorType >
void
DistributedMesh< Grid< 2, RealType, Device, Index > >::
setGlobalGrid( const GridType &globalGrid,
               const CoordinatesType& overlap )
{
   if(this->isSet && this->communicationGroup != nullptr)
        std::free(this->communicationGroup);
   this->communicationGroup= std::malloc(sizeof(typename CommunicatorType::CommunicationGroup));

   *((typename CommunicatorType::CommunicationGroup *)this->communicationGroup) = CommunicatorType::AllGroup;
   auto group=*((typename CommunicatorType::CommunicationGroup *)this->communicationGroup);

   this->globalGrid = globalGrid;
   this->isSet=true;
   this->overlap=overlap;

   for( int i=0; i<8; i++ )
      this->neighbors[i]=-1;

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

      this->setupNeighbors();

      this->localBegin=this->overlap;

      if(this->neighbors[Left]==-1)
      {
           this->localOrigin.x()+=this->overlap.x()*globalGrid.getSpaceSteps().x();
           this->localBegin.x()=0;
      }

      if(this->neighbors[Up]==-1)
      {
          this->localOrigin.y()+=this->overlap.y()*globalGrid.getSpaceSteps().y();
          this->localBegin.y()=0;
      }

      this->localGridSize=this->localSize;
      //Add Overlaps
      if(this->neighbors[Left]!=-1)
          this->localGridSize.x()+=this->overlap.x();
      if(this->neighbors[Right]!=-1)
          this->localGridSize.x()+=this->overlap.x();

      if(this->neighbors[Up]!=-1)
          this->localGridSize.y()+=this->overlap.y();
      if(this->neighbors[Down]!=-1)
          this->localGridSize.y()+=this->overlap.y();
  }
   //this->print( std::cerr );
}*/

/*template< typename Real, typename Device, typename Index >
template< typename CommunicatorType >
void
DistributedMesh< Grid< 2, Real, Device, Index > >::
setGlobalGrid( const GridType &globalGrid )
{
   if(this->isSet && this->communicationGroup != nullptr)
        std::free(this->communicationGroup);
   this->communicationGroup= std::malloc(sizeof(typename CommunicatorType::CommunicationGroup));

   *((typename CommunicatorType::CommunicationGroup *)this->communicationGroup) = CommunicatorType::AllGroup;
   auto group=*((typename CommunicatorType::CommunicationGroup *)this->communicationGroup);

   this->globalGrid = globalGrid;
   this->isSet=true;
   this->overlap.setValue( 1 );
   this->lowerOverlap.setValue( 0 );
   this->upperOverlap.setValue( 0 );

   for( int i=0; i<8; i++ )
      this->neighbors[i]=-1;

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
      //this->subdomainCoordinates[1]=0;
      this->domainDecomposition.setValue( 0 );
      //this->domainDecomposition[1]=1;
      this->localOrigin=globalGrid.getOrigin();
      this->localGridSize=globalGrid.getDimensions();
      this->localSize=globalGrid.getDimensions();
      this->globalBegin=CoordinatesType(0);
      this->localBegin.setValue( 0 );
      //this->localBegin.y()=0;
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
      numberOfLarger[ 0 ] = globalGrid.getDimensions().x() % this->domainDecomposition[ 0 ];
      numberOfLarger[ 1 ] = globalGrid.getDimensions().y() % this->domainDecomposition[ 1 ];

      this->localSize.x() = globalGrid.getDimensions().x() / this->domainDecomposition[ 0 ];
      this->localSize.y() = globalGrid.getDimensions().y() / this->domainDecomposition[ 1 ];
            
      if( numberOfLarger[ 0 ] > this->subdomainCoordinates[ 0 ] )
           this->localSize.x() += 1;               
      if( numberOfLarger[ 1 ] > this->subdomainCoordinates[ 1 ] )
          this->localSize.y() += 1;

      if( numberOfLarger[ 0 ] > this->subdomainCoordinates[ 0 ] )
          this->globalBegin.x() = this->subdomainCoordinates[ 0 ] * this->localSize.x();
      else
          this->globalBegin.x() = numberOfLarger[ 0 ] * ( this->localSize.x() + 1 ) + 
                                  ( this->subdomainCoordinates[ 0 ] - numberOfLarger[ 0 ] ) * this->localSize.x();

      if( numberOfLarger[ 1 ] > this->subdomainCoordinates[ 1 ] )
          this->globalBegin.y() = this->subdomainCoordinates[ 1 ] * this->localSize.y();
      else
          this->globalBegin.y() = numberOfLarger[ 1 ] * ( this->localSize.y() + 1 ) + 
                                  ( this->subdomainCoordinates[ 1 ] - numberOfLarger[ 1 ] ) * this->localSize.y();

      this->localGridSize = this->localSize;
      this->setupNeighbors();

  }
}

template< typename Real, typename Device, typename Index >
void
DistributedMesh< Grid< 2, Real, Device, Index > >::
setOverlaps( const SubdomainOverlapsType& lower,
             const SubdomainOverlapsType& upper )
{
   this->lowerOverlap = lower;
   this->upperOverlap = upper;
   
   this->localOrigin = this->globalGrid.getOrigin() +
         Containers::tnlDotProduct( this->globalGrid.getSpaceSteps(),
                                    this->globalBegin - this->lowerOverlap );

   this->localBegin = this->lowerOverlap;
   this->localGridSize = this->localSize + this->lowerOverlap + this->upperOverlap;   
}
*/

template< typename Real, typename Device, typename Index >
void
DistributedMesh< Grid< 2, Real, Device, Index > >::
setupGrid( GridType& grid )
{
   TNL_ASSERT_TRUE(this->isSet,"DistributedGrid is not set, but used by SetupGrid");
   grid.setOrigin( this->localOrigin );
   grid.setDimensions( this->localGridSize );
   //compute local proporions by sideefect
   grid.setSpaceSteps( this->spaceSteps );
   grid.setDistMesh(this);
};

template< typename Real, typename Device, typename Index >
String
DistributedMesh< Grid< 2, Real, Device, Index > >::
printProcessCoords() const
{
   return convertToString(this->subdomainCoordinates[0])+String("-")+convertToString(this->subdomainCoordinates[1]);
};

template< typename Real, typename Device, typename Index >
String
DistributedMesh< Grid< 2, Real, Device, Index > >::
printProcessDistr() const
{
   return convertToString(this->domainDecomposition[0])+String("-")+convertToString(this->domainDecomposition[1]);
};  


template< typename Real, typename Device, typename Index >
void
DistributedMesh< Grid< 2, Real, Device, Index > >::
writeProlog( Logger& logger ) const
{
   logger.writeParameter( "Domain decomposition:", this->getDomainDecomposition() );
};
        
      } //namespace DistributedMeshes
   } // namespace Meshes
} // namespace TNL
