/***************************************************************************
                          DistributedGridSynchronizer_1D.h  -  description
                             -------------------
    begin                : Aug 15, 2018
    copyright            : (C) 2018 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Meshes/Grid.h>
#include <TNL/Containers/Array.h>
#include <TNL/Meshes/DistributedMeshes/BufferEntitiesHelper.h>


namespace TNL {
namespace Functions{
template< typename Mesh,
          int MeshEntityDimension,
          typename Real  >
class MeshFunction;
}//Functions
}//TNL

namespace TNL {
namespace Meshes { 
namespace DistributedMeshes { 

template <typename Real,
          int EntityDimension,
          typename Index,
          typename Device,
          typename GridReal>  
class DistributedMeshSynchronizer< Functions::MeshFunction< Grid< 1, GridReal, Device, Index >, EntityDimension, Real > >
{

   public:
      using RealType = Real;
      typedef typename Grid< 1, GridReal, Device, Index >::Cell Cell;
      // FIXME: clang does not like this (incomplete type error)
//      typedef typename Functions::MeshFunction< Grid< 1, GridReal, Device, Index >,EntityDimension, RealType> MeshFunctionType;
      typedef typename Grid< 1, GridReal, Device, Index >::DistributedMeshType DistributedGridType;
      typedef typename DistributedGridType::CoordinatesType CoordinatesType;
      using SubdomainOverlapsType = typename DistributedGridType::SubdomainOverlapsType;

      DistributedMeshSynchronizer()
      {
          isSet = false;
      };

      DistributedMeshSynchronizer( DistributedGridType *distributedGrid )
      {
          isSet = false;
          setDistributedGrid( distributedGrid );
      };

      void setDistributedGrid( DistributedGridType *distributedGrid)
      {
         isSet=true;

         this->distributedGrid=distributedGrid;

         const SubdomainOverlapsType& lowerOverlap = distributedGrid->getLowerOverlap();
         const SubdomainOverlapsType& upperOverlap = distributedGrid->getUpperOverlap();
         const CoordinatesType& localSize = distributedGrid->getLocalSize(); 
         const CoordinatesType& localGridSize = this->distributedGrid->getLocalGridSize();
         

         sendBuffers[ Left ].setSize( lowerOverlap.x() );
         sendBuffers[ Right ].setSize( upperOverlap.x() );
         receiveBuffers[ Left ].setSize( lowerOverlap.x() );
         receiveBuffers[ Right ].setSize( upperOverlap.x() );         

      };

      template< typename CommunicatorType,
                typename MeshFunctionType,
                typename PeriodicBoundariesMaskPointer = Pointers::SharedPointer< MeshFunctionType > >
      void synchronize( MeshFunctionType &meshFunction,
                        bool periodicBoundaries = false,
                        const PeriodicBoundariesMaskPointer& mask = PeriodicBoundariesMaskPointer( nullptr ) )
      {
         TNL_ASSERT_TRUE( isSet, "Synchronizer is not set, but used to synchronize" );
         
         if( !distributedGrid->isDistributed() )
            return;

         int totalSize = meshFunction.getMesh().getDimensions().x();
         const SubdomainOverlapsType& lowerOverlap = distributedGrid->getLowerOverlap();
         const SubdomainOverlapsType& upperOverlap = distributedGrid->getUpperOverlap();
         const CoordinatesType& localGridSize = this->distributedGrid->getLocalGridSize();

         leftSource  = lowerOverlap.x();
         rightSource = localGridSize.x() - 2 * upperOverlap.x();
         leftDestination  = 0;
         rightDestination = localGridSize.x() - upperOverlap.x();
         
         const int *neighbors = distributedGrid->getNeighbors();
         const int *periodicNeighbors = distributedGrid->getPeriodicNeighbors();         
         
         if( periodicBoundaries )
         {
            if( neighbors[ Left ] == -1 )
               swap( leftSource, leftDestination );
            if( neighbors[ Right ] == -1 )
               swap( rightSource, rightDestination );
         }

         copyBuffers( meshFunction, sendBuffers, true,
                      leftSource, rightSource,
                      lowerOverlap, upperOverlap,
                      neighbors,
                      periodicBoundaries,
                      PeriodicBoundariesMaskPointer( nullptr ) ); // the mask is used only when receiving data 

         //async send
         typename CommunicatorType::Request requests[ 4 ];
         typename CommunicatorType::CommunicationGroup group;
         group=*((typename CommunicatorType::CommunicationGroup *)(distributedGrid->getCommunicationGroup()));
         int requestsCount( 0 );
         
         //send everything, recieve everything 
         if( neighbors[ Left ] != -1 )
         {
            TNL_ASSERT_GE( sendBuffers[ Left ].getSize(), lowerOverlap.x(), "" );
            TNL_ASSERT_GE( receiveBuffers[ Left ].getSize(), lowerOverlap.x(), "" );
            requests[ requestsCount++ ] = CommunicatorType::ISend( sendBuffers[ Left ].getData(), lowerOverlap.x(), neighbors[ Left ], 0, group );
            requests[ requestsCount++ ] = CommunicatorType::IRecv( receiveBuffers[ Left ].getData(), lowerOverlap.x(), neighbors[ Left ], 0, group );
         }
         else if( periodicBoundaries )
         {
            TNL_ASSERT_GE( sendBuffers[ Left ].getSize(), lowerOverlap.x(), "" );
            TNL_ASSERT_GE( receiveBuffers[ Left ].getSize(), lowerOverlap.x(), "" );            
            requests[ requestsCount++ ] = CommunicatorType::ISend( sendBuffers[ Left ].getData(), lowerOverlap.x(), periodicNeighbors[ Left ], 0, group );
            requests[ requestsCount++ ] = CommunicatorType::IRecv( receiveBuffers[ Left ].getData(), lowerOverlap.x(), periodicNeighbors[ Left ], 0, group );
         }        

         if( neighbors[ Right ] != -1 )
         {
            TNL_ASSERT_GE( sendBuffers[ Right ].getSize(), upperOverlap.x(), "" );
            TNL_ASSERT_GE( receiveBuffers[ Right ].getSize(), upperOverlap.x(), "" );
            requests[ requestsCount++ ] = CommunicatorType::ISend( sendBuffers[ Right ].getData(), upperOverlap.x(), neighbors[ Right ], 0, group );
            requests[ requestsCount++ ] = CommunicatorType::IRecv( receiveBuffers[ Right ].getData(), upperOverlap.x(), neighbors[ Right ], 0, group );
         }
         else if( periodicBoundaries )
         {
            TNL_ASSERT_GE( sendBuffers[ Right ].getSize(), upperOverlap.x(), "" );
            TNL_ASSERT_GE( receiveBuffers[ Right ].getSize(), upperOverlap.x(), "" );
            requests[ requestsCount++ ] = CommunicatorType::ISend( sendBuffers[ Right ].getData(), upperOverlap.x(), periodicNeighbors[ Right ], 0, group );
            requests[ requestsCount++ ] = CommunicatorType::IRecv( receiveBuffers[ Right ].getData(), upperOverlap.x(), periodicNeighbors[ Right ], 0, group );
         }
         
         //wait until send and receive is done
         CommunicatorType::WaitAll( requests, requestsCount );

         copyBuffers( meshFunction, receiveBuffers, false,
            leftDestination, rightDestination,
            lowerOverlap,
            upperOverlap,
            neighbors,
            periodicBoundaries,
            mask );
      }
      
   private:
      template< typename Real_,
                typename MeshFunctionType,
                typename PeriodicBoundariesMaskPointer >
      void copyBuffers( 
         MeshFunctionType& meshFunction,
         TNL::Containers::Array<Real_,Device>* buffers,
         bool toBuffer,
         int left, int right,
         const SubdomainOverlapsType& lowerOverlap,
         const SubdomainOverlapsType& upperOverlap,
         const int* neighbors,
         bool periodicBoundaries,
         const PeriodicBoundariesMaskPointer& mask )
      
      {
         typedef BufferEntitiesHelper< MeshFunctionType, PeriodicBoundariesMaskPointer, 1, Real_, Device > Helper;
         bool leftIsBoundary = ( neighbors[ Left ] == -1 );
         bool rightIsBoundary = ( neighbors[ Right ] == -1 );
         if( ! leftIsBoundary || periodicBoundaries )
            Helper::BufferEntities( meshFunction, mask, buffers[ Left ].getData(), leftIsBoundary, left, lowerOverlap.x(), toBuffer );
         if( ! rightIsBoundary || periodicBoundaries )
            Helper::BufferEntities( meshFunction, mask, buffers[ Right ].getData(), rightIsBoundary, right, upperOverlap.x(), toBuffer );
      }

      Containers::Array<RealType, Device> sendBuffers[ 2 ], receiveBuffers[ 2 ];

      DistributedGridType *distributedGrid;

      int leftSource;
      int rightSource;
      int leftDestination;
      int rightDestination;
      
      bool isSet;
};

} // namespace DistributedMeshes
} // namespace Meshes
} // namespace TNL
