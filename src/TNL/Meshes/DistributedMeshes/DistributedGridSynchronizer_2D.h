/***************************************************************************
                          DistributedGridSynchronizer_2D.h  -  description
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

template <typename RealType,
          int EntityDimension,
          typename Index,
          typename Device,
          typename GridReal>  
class DistributedMeshSynchronizer< Functions::MeshFunction< Grid< 2, GridReal, Device, Index >,EntityDimension, RealType>>
{

    public:
      
      typedef typename Grid< 2, GridReal, Device, Index >::Cell Cell;
      // FIXME: clang does not like this (incomplete type error)
//      typedef typename Functions::MeshFunction< Grid< 2, GridReal, Device, Index >,EntityDimension, RealType> MeshFunctionType;
      typedef typename Grid< 2, GridReal, Device, Index >::DistributedMeshType DistributedGridType; 
      typedef typename DistributedGridType::CoordinatesType CoordinatesType;
      using SubdomainOverlapsType = typename DistributedGridType::SubdomainOverlapsType;


      DistributedMeshSynchronizer()
      {
         isSet=false;
      };

      DistributedMeshSynchronizer(DistributedGridType *distributedGrid)
      {
         isSet=false;
         setDistributedGrid( distributedGrid );
      };

      void setDistributedGrid( DistributedGridType *distributedGrid )
      {
         isSet=true;

         this->distributedGrid = distributedGrid;
          
         const SubdomainOverlapsType& lowerOverlap = distributedGrid->getLowerOverlap();
         const SubdomainOverlapsType& upperOverlap = distributedGrid->getUpperOverlap();
         const CoordinatesType& localSize = distributedGrid->getLocalSize(); 
         const CoordinatesType& localGridSize = this->distributedGrid->getLocalGridSize();
         
         // TODO: SWAP up and down
         sizes[ Left ]      = localSize.y()    * lowerOverlap.x();
         sizes[ Right ]     = localSize.y()    * upperOverlap.x();
         sizes[ Up ]        = localSize.x()    * lowerOverlap.y();
         sizes[ Down ]      = localSize.x()    * upperOverlap.y();
         sizes[ UpLeft ]    = lowerOverlap.x() * lowerOverlap.y();
         sizes[ DownLeft ]  = lowerOverlap.x() * upperOverlap.y();
         sizes[ UpRight ]   = upperOverlap.x() * lowerOverlap.y();
         sizes[ DownRight ] = upperOverlap.x() * upperOverlap.y();
          
         for(int i=0;i<8;i++)
         {
            sendBuffers[ i ].setSize( sizes[ i ] );
            receiveBuffers[ i ].setSize( sizes[ i ] );
         }

      }

      template<typename CommunicatorType, typename MeshFunctionType>
      void synchronize( MeshFunctionType &meshFunction,
                        bool periodicBoundaries = false )
      {

         TNL_ASSERT_TRUE( isSet, "Synchronizer is not set, but used to synchronize" );

         if( !distributedGrid->isDistributed() )
            return;
         
         const SubdomainOverlapsType& lowerOverlap = distributedGrid->getLowerOverlap();
         const SubdomainOverlapsType& upperOverlap = distributedGrid->getUpperOverlap();
         const CoordinatesType& localSize = distributedGrid->getLocalSize();
         const CoordinatesType& localGridSize = this->distributedGrid->getLocalGridSize();
         
         leftSource  = lowerOverlap.x();
         rightSource = localGridSize.x() - 2 * upperOverlap.x();
         upSource    = lowerOverlap.y();                             // TODO: SWAP up and down
         downSource  = localGridSize.y() - 2 * upperOverlap.y();     // TODO: SWAP up and down

         xCenter  = lowerOverlap.x();
         yCenter  = lowerOverlap.y();

         leftDestination  = 0;
         rightDestination = localGridSize.x() - upperOverlap.x();
         upDestination    = 0;
         downDestination  = localGridSize.y() - upperOverlap.y();                       
         
         const int *neighbors = distributedGrid->getNeighbors();
         const int *periodicNeighbors = distributedGrid->getPeriodicNeighbors();
         
         if( periodicBoundaries )
         {
            if( neighbors[ Left ] == -1 )
               swap( leftSource, leftDestination );
            if( neighbors[ Right ] == -1 )
               swap( rightSource, rightDestination );
            if( neighbors[ Up ] == -1 )
               swap( upSource, upDestination );
            if( neighbors[ Down ] == -1 )
               swap( downSource, downDestination );
         }
         
         copyBuffers(meshFunction, sendBuffers, true,
            leftSource, rightSource, upSource, downSource,
            xCenter, yCenter,
            lowerOverlap, upperOverlap, localSize,
            neighbors,
            periodicBoundaries );

         //async send and receive
         typename CommunicatorType::Request requests[ 16 ];
         typename CommunicatorType::CommunicationGroup group;
         group=*((typename CommunicatorType::CommunicationGroup *)(distributedGrid->getCommunicationGroup()));
         int requestsCount( 0 );
         
         //send everything, receive everything 
         for( int i = 0; i < 8; i++ )
            if( neighbors[ i ] != -1 )
            {
               requests[ requestsCount++ ] = CommunicatorType::ISend( sendBuffers[ i ].getData(), sizes[ i ], neighbors[ i ], group );
               requests[ requestsCount++ ] = CommunicatorType::IRecv( receiveBuffers[ i ].getData(), sizes[ i ], neighbors[ i ], group );
            }
            else if( periodicBoundaries )
            {
               requests[ requestsCount++ ] = CommunicatorType::ISend( sendBuffers[ i ].getData(), sizes[ i ], periodicNeighbors[ i ], group );
               requests[ requestsCount++ ] = CommunicatorType::IRecv( receiveBuffers[ i ].getData(), sizes[ i ], periodicNeighbors[ i ], group );
            }

         //wait until send is done
         CommunicatorType::WaitAll( requests, requestsCount );

         //copy data from receive buffers
         copyBuffers(meshFunction, receiveBuffers, false,
              leftDestination, rightDestination, upDestination, downDestination,
              xCenter, yCenter,
              lowerOverlap, upperOverlap, localSize,
              neighbors,
              periodicBoundaries );
      }
    
   private:
      
      template< typename Real_, typename MeshFunctionType >
      void copyBuffers(MeshFunctionType& meshFunction, Containers::Array<Real_, Device, Index> * buffers, bool toBuffer,
                       int left, int right, int up, int down,
                       int xcenter, int ycenter,
                       const CoordinatesType& lowerOverlap,
                       const CoordinatesType& upperOverlap,
                       const CoordinatesType& localSize,
                       const int *neighbors,
                       bool periodicBoundaries )
      {
         // TODO: SWAP up and down
         using Helper = BufferEntitiesHelper< MeshFunctionType, 2, Real_, Device >;
         if( neighbors[ Left ] != -1 || periodicBoundaries )
            Helper::BufferEntities( meshFunction, buffers[ Left ].getData(), left, ycenter, lowerOverlap.x(), localSize.y(), toBuffer );
         if( neighbors[ Right ] != -1 || periodicBoundaries )
            Helper::BufferEntities( meshFunction, buffers[ Right ].getData(), right, ycenter, upperOverlap.x(), localSize.y(), toBuffer );
         if( neighbors[ Up ] != -1 || periodicBoundaries )
            Helper::BufferEntities( meshFunction, buffers[ Up ].getData(), xcenter, up, localSize.x(), lowerOverlap.y(), toBuffer );
         if( neighbors[ Down ] != -1 || periodicBoundaries )
            Helper::BufferEntities( meshFunction, buffers[ Down ].getData(), xcenter, down, localSize.x(), upperOverlap.y(), toBuffer );
         if( neighbors[ UpLeft ] != -1 || periodicBoundaries )
            Helper::BufferEntities( meshFunction, buffers[ UpLeft ].getData(), left, up, lowerOverlap.x(), lowerOverlap.y(), toBuffer );
         if( neighbors[ UpRight ] != -1 || periodicBoundaries )
            Helper::BufferEntities( meshFunction, buffers[ UpRight ].getData(), right, up, upperOverlap.x(), lowerOverlap.y(), toBuffer );
         if( neighbors[ DownLeft ] != -1 || periodicBoundaries )        
            Helper::BufferEntities( meshFunction, buffers[ DownLeft ].getData(), left, down, lowerOverlap.x(), upperOverlap.y(), toBuffer );
         if( neighbors[ DownRight ] != -1 || periodicBoundaries )
            Helper::BufferEntities( meshFunction, buffers[ DownRight ].getData(), right, down, upperOverlap.x(), upperOverlap.y(), toBuffer );
      }
      
      DistributedGridType *distributedGrid;

      Containers::Array<RealType, Device, Index> sendBuffers[8];
      Containers::Array<RealType, Device, Index> receiveBuffers[8];
      Containers::StaticArray< 8, int > sizes;

      int leftSource;
      int rightSource;
      int upSource;
      int downSource;
      int xCenter;
      int yCenter;
      int leftDestination;
      int rightDestination;
      int upDestination;
      int downDestination;

      bool isSet;      
};

} // namespace DistributedMeshes
} // namespace Meshes
} // namespace TNL


