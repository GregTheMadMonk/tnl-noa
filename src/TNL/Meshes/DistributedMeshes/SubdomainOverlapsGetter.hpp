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
   
   for( int i = 0; i < Dimension; i++ )
   {
      if( subdomainCoordinates[ i ] > 0 )
         lower[ i ] = subdomainOverlapSize;
      else
         lower[ i ] = periodicBoundariesOverlapSize[ i ];
      
      if( subdomainCoordinates[ i ] < distributedMesh->getDomainDecomposition()[ i ] - 1 )
         upper[ i ] = subdomainOverlapSize;
      else
         upper[ i ] = periodicBoundariesOverlapSize[ i ];
   }
}

      } // namespace DistributedMeshes
   } // namespace Meshes
} // namespace TNL
