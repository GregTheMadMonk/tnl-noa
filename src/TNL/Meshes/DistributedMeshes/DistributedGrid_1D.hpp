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

