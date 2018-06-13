/***************************************************************************
                          DistributedGrid_1D.hpp  -  description
                             -------------------
    begin                : January 09, 2018
    copyright            : (C) 2018 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Meshes/DistributedMeshes/DistributedGrid_1D.h>

namespace TNL {
   namespace Meshes {
      namespace DistributedMeshes {

template<typename RealType, typename Device, typename Index >
DistributedMesh< Grid< 1, RealType, Device, Index > >::
DistributedMesh()
: domainDecomposition( 0 ), isSet( false ) {}

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
void
DistributedMesh< Grid< 1, RealType, Device, Index > >::
setDomainDecomposition( const CoordinatesType& domainDecomposition )
{
   this->domainDecomposition = domainDecomposition;
}

template< typename RealType, typename Device, typename Index >     
const typename DistributedMesh< Grid< 1, RealType, Device, Index > >::CoordinatesType&
DistributedMesh< Grid< 1, RealType, Device, Index > >::
getDomainDecomposition() const
{
   return this->domainDecomposition;
}
      
template< typename RealType, typename Device, typename Index >     
   template< int EntityDimension >
Index
DistributedMesh< Grid< 1, RealType, Device, Index > >::
getEntitiesCount() const
{
   return this->globalGrid. template getEntitiesCount< EntityDimension >();
}

template< typename RealType, typename Device, typename Index >     
   template< typename Entity >
Index
DistributedMesh< Grid< 1, RealType, Device, Index > >::
getEntitiesCount() const
{
   return this->globalGrid. template getEntitiesCount< Entity >();
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

   Dimensions = GridType::getMeshDimension();
   spaceSteps = globalGrid.getSpaceSteps();

   distributed = false;
   if( CommunicatorType::IsInitialized() )
   {
       rank = CommunicatorType::GetRank();
       this->nproc = CommunicatorType::GetSize();
       if( this->nproc>1 )
       {
           distributed = true;
       }
   }

   if( !distributed )
   {
       rank = 0;
       localOrigin = globalGrid.getOrigin();
       localSize = globalGrid.getDimensions();
       localGridSize = globalGrid.getDimensions();
       globalDimensions = globalGrid.getDimensions();
       globalBegin = CoordinatesType(0);
       localBegin = CoordinatesType(0);
       this->domainDecomposition[ 0 ];
       return;
   }
   else
   {            
       //nearnodes
       if( rank != 0 ) left=rank-1;
       if( rank != nproc-1 ) right=rank+1;

       this->domainDecomposition[ 0 ] = nproc;
       globalDimensions=globalGrid.getDimensions();                 

       //compute local mesh size               
       numberOfLarger = globalGrid.getDimensions().x() % nproc;

       localSize.x() = globalGrid.getDimensions().x() / nproc;
       if(numberOfLarger>rank) localSize.x() += 1;

       if(numberOfLarger>rank)
       {
           globalBegin.x()=rank*localSize.x();
           localOrigin.x()=globalGrid.getOrigin().x()
                        +(globalBegin.x()-overlap.x())*globalGrid.getSpaceSteps().x();
       }
       else
       {
           globalBegin.x()=numberOfLarger*(localSize.x()+1)+(rank-numberOfLarger)*localSize.x();
           localOrigin.x()=(globalGrid.getOrigin().x()-overlap.x())
                        +globalBegin.x()*globalGrid.getSpaceSteps().x();
       }

      localBegin=overlap;

       //vlevo neni prekryv
       if(left==-1)
       {
           localOrigin.x()+=overlap.x()*globalGrid.getSpaceSteps().x();
           localBegin.x()=0;
       }

       localGridSize = localSize;
       //add overlaps
       if( left == -1 || right == -1 )
           localGridSize.x() += overlap.x();
       else
           localGridSize.x() += 2*overlap.x();
   }  
} 

template< typename RealType, typename Device, typename Index >     
void
DistributedMesh< Grid< 1, RealType, Device, Index > >::
setupGrid( GridType& grid)
{
   TNL_ASSERT_TRUE(isSet,"DistributedGrid is not set, but used by SetupGrid");
   grid.setOrigin(localOrigin);
   grid.setDimensions(localGridSize);
   //compute local proportions by sideefect
   grid.setSpaceSteps(spaceSteps);
   grid.SetDistMesh(this);
};

template< typename RealType, typename Device, typename Index >     
String
DistributedMesh< Grid< 1, RealType, Device, Index > >::
printProcessCoords() const
{
   return convertToString(rank);
};

template< typename RealType, typename Device, typename Index >     
String
DistributedMesh< Grid< 1, RealType, Device, Index > >::
printProcessDistr() const
{
   return convertToString(nproc);
};       

template< typename RealType, typename Device, typename Index >     
bool
DistributedMesh< Grid< 1, RealType, Device, Index > >::
isDistributed() const
{
   return this->distributed;
};

template< typename RealType, typename Device, typename Index >     
int
DistributedMesh< Grid< 1, RealType, Device, Index > >::
getLeft() const
{
   TNL_ASSERT_TRUE(isSet,"DistributedGrid is not set, but used by getLeft");
   return this->left;
};

template< typename RealType, typename Device, typename Index >     
int
DistributedMesh< Grid< 1, RealType, Device, Index > >::
getRight() const
{
   TNL_ASSERT_TRUE(isSet,"DistributedGrid is not set, but used by getRight");
   return this->right;
};

template< typename RealType, typename Device, typename Index >     
const typename DistributedMesh< Grid< 1, RealType, Device, Index > >::CoordinatesType&
DistributedMesh< Grid< 1, RealType, Device, Index > >::
getOverlap() const
{
   return this->overlap;
};

template< typename RealType, typename Device, typename Index >     
const typename DistributedMesh< Grid< 1, RealType, Device, Index > >::CoordinatesType&
DistributedMesh< Grid< 1, RealType, Device, Index > >::
getLocalSize() const
{
   return this->localSize;
}

template< typename RealType, typename Device, typename Index >     
const typename DistributedMesh< Grid< 1, RealType, Device, Index > >::CoordinatesType&
DistributedMesh< Grid< 1, RealType, Device, Index > >::
getGlobalSize() const
{
   return this->globalGrid.getDimensions();
}

template< typename RealType, typename Device, typename Index >     
const typename DistributedMesh< Grid< 1, RealType, Device, Index > >::CoordinatesType&
DistributedMesh< Grid< 1, RealType, Device, Index > >::
getGlobalBegin() const
{
   return this->globalBegin;
}

template< typename RealType, typename Device, typename Index >     
const typename DistributedMesh< Grid< 1, RealType, Device, Index > >::CoordinatesType&
DistributedMesh< Grid< 1, RealType, Device, Index > >::
getLocalGridSize() const
{
   return this->localGridSize;
}

template< typename RealType, typename Device, typename Index >     
const typename DistributedMesh< Grid< 1, RealType, Device, Index > >::CoordinatesType&
DistributedMesh< Grid< 1, RealType, Device, Index > >::
getLocalBegin() const
{
   return this->localBegin;
}

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

