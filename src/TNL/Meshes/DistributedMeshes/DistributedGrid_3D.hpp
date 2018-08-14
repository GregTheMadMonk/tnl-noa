/***************************************************************************
                          DistributedGrid_3D.hpp  -  description
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
setupGrid( GridType& grid)
{
   TNL_ASSERT_TRUE(this->isSet,"DistributedGrid is not set, but used by SetupGrid");
   grid.setOrigin(this->localOrigin);
   grid.setDimensions(this->localGridSize);
   //compute local proportions by side efect
   grid.setSpaceSteps(this->spaceSteps);
   grid.setDistMesh(this);
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
void
DistributedMesh< Grid< 3, RealType, Device, Index > >::
writeProlog( Logger& logger )
{
   logger.writeParameter( "Domain decomposition:", this->getDomainDecomposition() );
}           
   
      } //namespace DistributedMeshes
   } // namespace Meshes
} // namespace TNL
