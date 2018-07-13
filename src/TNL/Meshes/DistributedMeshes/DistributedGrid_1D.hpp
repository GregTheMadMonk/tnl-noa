/***************************************************************************
                          DistributedGrid_1D.hpp  -  description
                             -------------------
    begin                : January 09, 2018
    copyright            : (C) 2018 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <cstdlib>

namespace TNL {
   namespace Meshes {
      namespace DistributedMeshes {

template< typename RealType, typename Device, typename Index >     
bool
DistributedMesh< Grid< 1, RealType, Device, Index > >::
setup( const Config::ParameterContainer& parameters,
       const String& prefix )
{
   this->domainDecomposition.x() = parameters.getParameter< int >( "grid-domain-decomposition-x" );
   return true;
}      

template< typename RealType, typename Device, typename Index >     
   template< typename CommunicatorType>
void
DistributedMesh< Grid< 1, RealType, Device, Index > >::
setGlobalGrid( const GridType& globalGrid,
               const CoordinatesType& overlap )
{

   if(this->isSet && this->communicationGroup != nullptr)
        std::free(this->communicationGroup);
   this->communicationGroup= std::malloc(sizeof(typename CommunicatorType::CommunicationGroup));

   *((typename CommunicatorType::CommunicationGroup *)this->communicationGroup) = CommunicatorType::AllGroup;
    auto group=*((typename CommunicatorType::CommunicationGroup *)this->communicationGroup);

   this->globalGrid = globalGrid;
   this->isSet = true;
   this->overlap = overlap;
   this->neighbors[Left]=-1;
   this->neighbors[Right]=-1;

   this->Dimensions = GridType::getMeshDimension();
   this->spaceSteps = globalGrid.getSpaceSteps();

   this->distributed = false;
   if( CommunicatorType::IsInitialized() )
   {
       this->rank = CommunicatorType::GetRank(group);
       this->nproc = CommunicatorType::GetSize(group);
       if( this->nproc>1 )
       {
           this->distributed = true;
       }
   }

   if( !this->distributed )
   {
       this->rank = 0;
       this->localOrigin = globalGrid.getOrigin();
       this->localSize = globalGrid.getDimensions();
       this->localGridSize = globalGrid.getDimensions();
       this->globalBegin = CoordinatesType(0);
       this->localBegin = CoordinatesType(0);
       this->domainDecomposition[ 0 ];
       return;
   }
   else
   {            
       this->domainDecomposition[ 0 ] = this->nproc;
       this->subdomainCoordinates[ 0 ]= this->rank;
       
       //compute local mesh size               
       int numberOfLarger = globalGrid.getDimensions().x() % this->nproc;

       this->localSize.x() = globalGrid.getDimensions().x() / this->nproc;
       if(numberOfLarger>this->rank) this->localSize.x() += 1;

       if(numberOfLarger>this->rank)
       {
           this->globalBegin.x()=this->rank*this->localSize.x();
           this->localOrigin.x()=globalGrid.getOrigin().x()
                        +(this->globalBegin.x()-this->overlap.x())*this->globalGrid.getSpaceSteps().x();
       }
       else
       {
           this->globalBegin.x()=numberOfLarger*(this->localSize.x()+1)+(this->rank-numberOfLarger)*this->localSize.x();
           this->localOrigin.x()=(this->globalGrid.getOrigin().x()-overlap.x())
                        +this->globalBegin.x()*this->globalGrid.getSpaceSteps().x();
       }

      this->setUpNeighbors();

      this->localBegin=overlap;

       //vlevo neni prekryv
       if(this->neighbors[Left]==-1)
       {
           this->localOrigin.x()+=this->overlap.x()*this->globalGrid.getSpaceSteps().x();
           this->localBegin.x()=0;
       }

       this->localGridSize = this->localSize;
       //add overlaps
       if( this->neighbors[Left] == -1 || this->neighbors[Right] == -1 )
           this->localGridSize.x() += this->overlap.x();
       else
           this->localGridSize.x() += 2*this->overlap.x();
   }  
} 

template< typename RealType, typename Device, typename Index >     
void
DistributedMesh< Grid< 1, RealType, Device, Index > >::
setupGrid( GridType& grid)
{
   TNL_ASSERT_TRUE(this->isSet,"DistributedGrid is not set, but used by SetupGrid");
   grid.setOrigin(this->localOrigin);
   grid.setDimensions(this->localGridSize);
   //compute local proportions by sideefect
   grid.setSpaceSteps(this->spaceSteps);
   grid.setDistMesh(this);
};

template< typename RealType, typename Device, typename Index >     
String
DistributedMesh< Grid< 1, RealType, Device, Index > >::
printProcessCoords() const
{
   return convertToString(this->rank);
};

template< typename RealType, typename Device, typename Index >     
String
DistributedMesh< Grid< 1, RealType, Device, Index > >::
printProcessDistr() const
{
   return convertToString(this->nproc);
};       

template< typename RealType, typename Device, typename Index >
void
DistributedMesh< Grid< 1, RealType, Device, Index > >::
writeProlog( Logger& logger ) const
{
   this->globalGrid.writeProlog( logger );
   logger.writeParameter( "Domain decomposition:", this->getDomainDecomposition() );
}

      } //namespace DistributedMeshes
   } // namespace Meshes
} // namespace TNL

