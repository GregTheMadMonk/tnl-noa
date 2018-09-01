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
