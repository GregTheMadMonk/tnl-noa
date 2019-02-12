/***************************************************************************
                          DistributedGridSynchronizer.h  -  description
                             -------------------
    begin                : March 17, 2017
    copyright            : (C) 2017 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Meshes/Grid.h>
#include <TNL/Containers/Array.h>
#include <TNL/Meshes/DistributedMeshes/BufferEntitiesHelper.h>
#include <TNL/Meshes/DistributedMeshes/Directions.h>
#include <TNL/Communicators/MPIPrint.h>

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
          int MeshDimension,
          typename Index,
          typename Device,
          typename GridReal>  
class DistributedMeshSynchronizer< Functions::MeshFunction< Grid< MeshDimension, GridReal, Device, Index >,EntityDimension, RealType>>
{

   public:
      static constexpr int getMeshDimension() { return MeshDimension; };
      static constexpr int getNeighborCount() {return DirectionCount<MeshDimension>::get();};

      typedef typename Grid< MeshDimension, GridReal, Device, Index >::Cell Cell;
      // FIXME: clang does not like this (incomplete type error)
//      typedef typename Functions::MeshFunction< Grid< 3, GridReal, Device, Index >,EntityDimension, RealType> MeshFunctionType;
      typedef typename Grid< MeshDimension, GridReal, Device, Index >::DistributedMeshType DistributedGridType; 
      typedef typename DistributedGridType::CoordinatesType CoordinatesType;
      using SubdomainOverlapsType = typename DistributedGridType::SubdomainOverlapsType;

      enum PeriodicBoundariesCopyDirection
      {
         BoundaryToOverlap,
         OverlapToBoundary
      };

      DistributedMeshSynchronizer()
      {
         isSet = false;
      };

      DistributedMeshSynchronizer( DistributedGridType *distributedGrid )
      {
         isSet = false;
         setDistributedGrid( distributedGrid );
      };

      void setPeriodicBoundariesCopyDirection( const PeriodicBoundariesCopyDirection dir )
      {
         this->periodicBoundariesCopyDirection = dir;
      }

      void setDistributedGrid( DistributedGridType *distributedGrid )
      {
         isSet = true;

         this->distributedGrid = distributedGrid;
         
         const SubdomainOverlapsType& lowerOverlap = this->distributedGrid->getLowerOverlap();
         const SubdomainOverlapsType& upperOverlap = this->distributedGrid->getUpperOverlap();
       
         const CoordinatesType& localBegin = this->distributedGrid->getLocalBegin(); 
         const CoordinatesType& localSize = this->distributedGrid->getLocalSize(); 
         const CoordinatesType& localGridSize = this->distributedGrid->getLocalGridSize();

         const int *neighbors = distributedGrid->getNeighbors();

         for( int i=0; i<this->getNeighborCount(); i++ )
         {
            Index sendSize=1;//sended and recieve areas has same size

           // bool isBoundary=( neighbor[ i ] == -1 );
            auto directions=Directions::template getXYZ<getMeshDimension()>(i);

            sendDimensions[i]=localSize;//send and recieve areas has same dimensions
            sendBegin[i]=localBegin;
            recieveBegin[i]=localBegin;

            for(int j=0;j<this->getMeshDimension();j++)
            {
               if(directions[j]==-1)
               {
                  sendDimensions[i][j]=lowerOverlap[j];
                  recieveBegin[i][j]=0;
               }

               if(directions[j]==1)
               {
                  sendDimensions[i][j]=upperOverlap[j];
                  sendBegin[i][j]=localBegin[j]+localSize[j]-upperOverlap[j];
                  recieveBegin[i][j]=localBegin[j]+localSize[j];
               }

               sendSize*=sendDimensions[i][j];
            }

            sendSizes[ i ] = sendSize;
            sendBuffers[ i ].setSize( sendSize );
            recieveBuffers[ i ].setSize( sendSize);

            if( this->periodicBoundariesCopyDirection == OverlapToBoundary &&
               neighbors[ i ] == -1 )
                  swap( sendBegin[i], recieveBegin[i] );
         }
     }

      template< typename CommunicatorType,
                typename MeshFunctionType,
                typename PeriodicBoundariesMaskPointer = Pointers::SharedPointer< MeshFunctionType > >
      void synchronize( MeshFunctionType &meshFunction,
                        bool periodicBoundaries = false,
                        const PeriodicBoundariesMaskPointer& mask = PeriodicBoundariesMaskPointer( nullptr ) )
      {

         TNL_ASSERT_TRUE( isSet, "Synchronizer is not set, but used to synchronize" );

    	   if( !distributedGrid->isDistributed() ) return;
         
         const int *neighbors = distributedGrid->getNeighbors();
         const int *periodicNeighbors = distributedGrid->getPeriodicNeighbors();
        
         //fill send buffers
         copyBuffers( meshFunction, 
            sendBuffers, sendBegin,sendDimensions,
            true,
            neighbors,
            periodicBoundaries,
            PeriodicBoundariesMaskPointer( nullptr ) ); // the mask is used only when receiving data );

         //async send and receive
         typename CommunicatorType::Request requests[2*this->getNeighborCount()];
         typename CommunicatorType::CommunicationGroup group;
         group=*((typename CommunicatorType::CommunicationGroup *)(distributedGrid->getCommunicationGroup()));
         int requestsCount( 0 );

         //send everything, recieve everything 
         for( int i=0; i<this->getNeighborCount(); i++ )
            if( neighbors[ i ] != -1 )
            {
               requests[ requestsCount++ ] = CommunicatorType::ISend( sendBuffers[ i ].getData(),  sendSizes[ i ], neighbors[ i ], 0, group );
               requests[ requestsCount++ ] = CommunicatorType::IRecv( recieveBuffers[ i ].getData(),  sendSizes[ i ], neighbors[ i ], 0, group );
            }
            else if( periodicBoundaries && sendSizes[ i ] !=0 )
      	   {
               requests[ requestsCount++ ] = CommunicatorType::ISend( sendBuffers[ i ].getData(),  sendSizes[ i ], periodicNeighbors[ i ], 1, group );
               requests[ requestsCount++ ] = CommunicatorType::IRecv( recieveBuffers[ i ].getData(),  sendSizes[ i ], periodicNeighbors[ i ], 1, group );
            }

        //wait until send is done
        CommunicatorType::WaitAll( requests, requestsCount );

        //copy data from receive buffers
        copyBuffers(meshFunction,
            recieveBuffers,recieveBegin,sendDimensions  ,
            false,
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
         Containers::Array<Real_, Device, Index>* buffers,
         CoordinatesType* begins,
         CoordinatesType* sizes,
         bool toBuffer,
         const int* neighbor,
         bool periodicBoundaries,
         const PeriodicBoundariesMaskPointer& mask )
      {
         using Helper = BufferEntitiesHelper< MeshFunctionType, PeriodicBoundariesMaskPointer, getMeshDimension(), Real_, Device >;
       
         for(int i=0;i<this->getNeighborCount();i++)
         {
            bool isBoundary=( neighbor[ i ] == -1 );
            if( ! isBoundary || periodicBoundaries )
            {
               Helper::BufferEntities( meshFunction, mask, buffers[ i ].getData(), isBoundary, begins[i], sizes[i], toBuffer );
            }
         }
      }

   private:

      Containers::Array<RealType, Device, Index> sendBuffers[getNeighborCount()];
      Containers::Array<RealType, Device, Index> recieveBuffers[getNeighborCount()];
      Containers::StaticArray< getNeighborCount(), int > sendSizes;

      PeriodicBoundariesCopyDirection periodicBoundariesCopyDirection = BoundaryToOverlap;

      CoordinatesType sendDimensions[getNeighborCount()];
      CoordinatesType recieveDimensions[getNeighborCount()];
      CoordinatesType sendBegin[getNeighborCount()];
      CoordinatesType recieveBegin[getNeighborCount()];

      DistributedGridType *distributedGrid;

      bool isSet;
};


} // namespace DistributedMeshes
} // namespace Meshes
} // namespace TNL

