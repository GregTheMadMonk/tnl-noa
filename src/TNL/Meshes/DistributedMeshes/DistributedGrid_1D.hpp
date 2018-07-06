/***************************************************************************
                          DistributedGrid_1D.hpp  -  description
                             -------------------
    begin                : January 09, 2018
    copyright            : (C) 2018 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

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
   this->globalGrid = globalGrid;
   this->isSet = true;
   this->overlap = overlap;
   left=-1;
   right=-1;

   this->Dimensions = GridType::getMeshDimension();
   this->spaceSteps = globalGrid.getSpaceSteps();

   this->distributed = false;
   if( CommunicatorType::IsInitialized() )
   {
       this->rank = CommunicatorType::GetRank();
       this->nproc = CommunicatorType::GetSize();
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
       this->globalDimensions = globalGrid.getDimensions();
       this->globalBegin = CoordinatesType(0);
       this->localBegin = CoordinatesType(0);
       this->domainDecomposition[ 0 ];
       return;
   }
   else
   {            
       //nearnodes
       if( this->rank != 0 ) left=this->rank-1;
       if( this->rank != this->nproc-1 ) right=this->rank+1;

       this->domainDecomposition[ 0 ] = this->nproc;
       this->globalDimensions=globalGrid.getDimensions();                 

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

      this->localBegin=overlap;

       //vlevo neni prekryv
       if(left==-1)
       {
           this->localOrigin.x()+=this->overlap.x()*this->globalGrid.getSpaceSteps().x();
           this->localBegin.x()=0;
       }

       this->localGridSize = this->localSize;
       //add overlaps
       if( left == -1 || right == -1 )
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
   grid.SetDistMesh(this);
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
int
DistributedMesh< Grid< 1, RealType, Device, Index > >::
getLeft() const
{
   TNL_ASSERT_TRUE(this->isSet,"DistributedGrid is not set, but used by getLeft");
   return this->left;
};

template< typename RealType, typename Device, typename Index >     
int
DistributedMesh< Grid< 1, RealType, Device, Index > >::
getRight() const
{
   TNL_ASSERT_TRUE(this->isSet,"DistributedGrid is not set, but used by getRight");
   return this->right;
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

