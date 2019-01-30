/***************************************************************************
                          SubdomainOverlapsGetter.hpp  -  description
                             -------------------
    begin                : Aug 13, 2018
    copyright            : (C) 2018 by oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Assert.h>
#include <TNL/Meshes/Grid.h>

namespace TNL {
   namespace Meshes {
      namespace DistributedMeshes {

template< int Dimension,
          typename Real,
          typename Device,
          typename Index,
          typename Communicator >
void
SubdomainOverlapsGetter< Grid< Dimension, Real, Device, Index >, Communicator >::
getOverlaps( const DistributedMeshType* distributedMesh,
             SubdomainOverlapsType& lower,
             SubdomainOverlapsType& upper,
             IndexType subdomainOverlapSize,
             const SubdomainOverlapsType& periodicBoundariesOverlapSize )
{
   if( ! CommunicatorType::isDistributed() )
      return;
   TNL_ASSERT_TRUE( distributedMesh != NULL, "" );
   
   const CoordinatesType& subdomainCoordinates = distributedMesh->getSubdomainCoordinates();
   int rank = CommunicatorType::GetRank( CommunicatorType::AllGroup );
   
   for( int i = 0; i < Dimension; i++ )
   {
      CoordinatesType neighborDirection( 0 );
      neighborDirection[ i ] = -1;
      if( subdomainCoordinates[ i ] > 0 )
         lower[ i ] = subdomainOverlapSize;
      else if( distributedMesh->getPeriodicNeighbors()[ Directions::getDirection( neighborDirection ) ] != rank )
         lower[ i ] = periodicBoundariesOverlapSize[ i ];
      
      neighborDirection[ i ] = 1;
      if( subdomainCoordinates[ i ] < distributedMesh->getDomainDecomposition()[ i ] - 1 )
         upper[ i ] = subdomainOverlapSize;
      else if( distributedMesh->getPeriodicNeighbors()[ Directions::getDirection( neighborDirection ) ] != rank )
         upper[ i ] = periodicBoundariesOverlapSize[ i ];
   }
}

      } // namespace DistributedMeshes
   } // namespace Meshes
} // namespace TNL
