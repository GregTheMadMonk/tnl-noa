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

/*
 * TODO: This could work when the MPI directions are rewritten
         
template< typename Real,
          typename Device,
          typename Index,
          typename Communicator >
void
SubdomainOverlapsGetter< Grid< 1, Real, Device, Index >, Communicator >::
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
 
*/

template< typename Real,
          typename Device,
          typename Index,
          typename Communicator >
void
SubdomainOverlapsGetter< Grid< 1, Real, Device, Index >, Communicator >::
getOverlaps( const DistributedMeshType* distributedMesh,
             SubdomainOverlapsType& lower,
             SubdomainOverlapsType& upper,
             IndexType subdomainOverlapSize,
             const SubdomainOverlapsType& lowerPeriodicBoundariesOverlapSize,
             const SubdomainOverlapsType& upperPeriodicBoundariesOverlapSize )
{
   if( ! CommunicatorType::isDistributed() )
      return;
   TNL_ASSERT_TRUE( distributedMesh != NULL, "" );

   const CoordinatesType& subdomainCoordinates = distributedMesh->getSubdomainCoordinates();
   int rank = CommunicatorType::GetRank( CommunicatorType::AllGroup );
   
   if( subdomainCoordinates[ 0 ] > 0 )
      lower[ 0 ] = subdomainOverlapSize;
   else if( distributedMesh->getPeriodicNeighbors()[ ZzYzXm ] != rank )
      lower[ 0 ] = lowerPeriodicBoundariesOverlapSize[ 0 ];

   if( subdomainCoordinates[ 0 ] < distributedMesh->getDomainDecomposition()[ 0 ] - 1 )
      upper[ 0 ] = subdomainOverlapSize;
   else if( distributedMesh->getPeriodicNeighbors()[ ZzYzXp ] != rank )
      upper[ 0 ] = upperPeriodicBoundariesOverlapSize[ 0 ];
}


template< typename Real,
          typename Device,
          typename Index,
          typename Communicator >
void
SubdomainOverlapsGetter< Grid< 2, Real, Device, Index >, Communicator >::
getOverlaps( const DistributedMeshType* distributedMesh,
             SubdomainOverlapsType& lower,
             SubdomainOverlapsType& upper,
             IndexType subdomainOverlapSize,
             const SubdomainOverlapsType& lowerPeriodicBoundariesOverlapSize,
             const SubdomainOverlapsType& upperPeriodicBoundariesOverlapSize )
{
   if( ! CommunicatorType::isDistributed() )
      return;
   TNL_ASSERT_TRUE( distributedMesh != NULL, "" );

   const CoordinatesType& subdomainCoordinates = distributedMesh->getSubdomainCoordinates();
   int rank = CommunicatorType::GetRank( CommunicatorType::AllGroup );
   lower = 0;
   upper = 0;
   
   if( subdomainCoordinates[ 0 ] > 0 )
      lower[ 0 ] = subdomainOverlapSize;
   else if( distributedMesh->getPeriodicNeighbors()[ ZzYzXm ] != rank )
      lower[ 0 ] = lowerPeriodicBoundariesOverlapSize[ 0 ];

   if( subdomainCoordinates[ 0 ] < distributedMesh->getDomainDecomposition()[ 0 ] - 1 )
      upper[ 0 ] = subdomainOverlapSize;
   else if( distributedMesh->getPeriodicNeighbors()[ ZzYzXp ] != rank )
      upper[ 0 ] = upperPeriodicBoundariesOverlapSize[ 0 ];
   
   if( subdomainCoordinates[ 1 ] > 0 )
      lower[ 1 ] = subdomainOverlapSize;
   else if( distributedMesh->getPeriodicNeighbors()[ ZzYmXz ] != rank )
      lower[ 1 ] = lowerPeriodicBoundariesOverlapSize[ 1 ];

   if( subdomainCoordinates[ 1 ] < distributedMesh->getDomainDecomposition()[ 1 ] - 1 )
      upper[ 1 ] = subdomainOverlapSize;
   else if( distributedMesh->getPeriodicNeighbors()[ ZzYpXz ] != rank )
      upper[ 1 ] = upperPeriodicBoundariesOverlapSize[ 1 ];
}

template< typename Real,
          typename Device,
          typename Index,
          typename Communicator >
void
SubdomainOverlapsGetter< Grid< 3, Real, Device, Index >, Communicator >::
getOverlaps( const DistributedMeshType* distributedMesh,
             SubdomainOverlapsType& lower,
             SubdomainOverlapsType& upper,
             IndexType subdomainOverlapSize,
             const SubdomainOverlapsType& lowerPeriodicBoundariesOverlapSize,
             const SubdomainOverlapsType& upperPeriodicBoundariesOverlapSize )
{
   if( ! CommunicatorType::isDistributed() )
      return;
   TNL_ASSERT_TRUE( distributedMesh != NULL, "" );

   const CoordinatesType& subdomainCoordinates = distributedMesh->getSubdomainCoordinates();
   int rank = CommunicatorType::GetRank( CommunicatorType::AllGroup );
   
   if( subdomainCoordinates[ 0 ] > 0 )
      lower[ 0 ] = subdomainOverlapSize;
   else if( distributedMesh->getPeriodicNeighbors()[ ZzYzXm ] != rank )
      lower[ 0 ] = lowerPeriodicBoundariesOverlapSize[ 0 ];

   if( subdomainCoordinates[ 0 ] < distributedMesh->getDomainDecomposition()[ 0 ] - 1 )
      upper[ 0 ] = subdomainOverlapSize;
   else if( distributedMesh->getPeriodicNeighbors()[ ZzYzXp ] != rank )
      upper[ 0 ] = upperPeriodicBoundariesOverlapSize[ 0 ];
   
   if( subdomainCoordinates[ 1 ] > 0 )
      lower[ 1 ] = subdomainOverlapSize;
   else if( distributedMesh->getPeriodicNeighbors()[ ZzYmXz ] != rank )
      lower[ 1 ] = lowerPeriodicBoundariesOverlapSize[ 1 ];

   if( subdomainCoordinates[ 1 ] < distributedMesh->getDomainDecomposition()[ 1 ] - 1 )
      upper[ 1 ] = subdomainOverlapSize;
   else if( distributedMesh->getPeriodicNeighbors()[ ZzYpXz ] != rank )
      upper[ 1 ] = upperPeriodicBoundariesOverlapSize[ 1 ];
   
   if( subdomainCoordinates[ 2 ] > 0 )
      lower[ 2 ] = subdomainOverlapSize;
   else if( distributedMesh->getPeriodicNeighbors()[ ZmYzXz ] != rank )
      lower[ 2 ] = lowerPeriodicBoundariesOverlapSize[ 2 ];

   if( subdomainCoordinates[ 2 ] < distributedMesh->getDomainDecomposition()[ 2 ] - 1 )
      upper[ 2 ] = subdomainOverlapSize;
   else if( distributedMesh->getPeriodicNeighbors()[ ZpYzXz ] != rank )
      upper[ 2 ] = upperPeriodicBoundariesOverlapSize[ 2 ];
}

      } // namespace DistributedMeshes
   } // namespace Meshes
} // namespace TNL
