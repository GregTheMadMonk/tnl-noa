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
#include <TNL/Meshes/DistributedMeshes/DistributedGrid.h>
#include <TNL/Meshes/DistributedMeshes/DistributedMeshSynchronizer.h>
#include <TNL/Meshes/DistributedMeshes/BufferEntitiesHelper.h>
#include <TNL/Meshes/DistributedMeshes/Directions.h>
#include <TNL/Pointers/SharedPointer.h>

namespace TNL {
namespace Meshes {
namespace DistributedMeshes {

// NOTE: this specialization works only for synchronizations on cells
template< int MeshDimension,
          typename Index,
          typename Device,
          typename GridReal >
class DistributedMeshSynchronizer< DistributedMesh< Grid< MeshDimension, GridReal, Device, Index > >, MeshDimension >
{
   public:
      static constexpr int getMeshDimension() { return MeshDimension; };
      static constexpr int getNeighborCount() {return DirectionCount<MeshDimension>::get();};

      typedef typename Grid< MeshDimension, GridReal, Device, Index >::Cell Cell;
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

         const int *neighbors = distributedGrid->getNeighbors();

         for( int i=0; i<this->getNeighborCount(); i++ )
         {
            Index sendSize=1;//send and receive  areas have the same size

           // bool isBoundary=( neighbor[ i ] == -1 );
            auto directions=Directions::template getXYZ<getMeshDimension()>(i);

            sendDimensions[i]=localSize; //send and receive areas have the same dimensions
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

            if( this->periodicBoundariesCopyDirection == OverlapToBoundary &&
               neighbors[ i ] == -1 )
                  swap( sendBegin[i], recieveBegin[i] );
         }
     }

      template< typename MeshFunctionType,
                typename PeriodicBoundariesMaskPointer = Pointers::SharedPointer< MeshFunctionType > >
      void synchronize( MeshFunctionType &meshFunction,
                        bool periodicBoundaries = false,
                        const PeriodicBoundariesMaskPointer& mask = PeriodicBoundariesMaskPointer( nullptr ) )
      {
         using RealType = typename MeshFunctionType::RealType;

         static_assert( MeshFunctionType::getEntitiesDimension() == MeshFunctionType::getMeshDimension(),
                        "this specialization works only for synchronizations on cells" );
         TNL_ASSERT_TRUE( isSet, "Synchronizer is not set, but used to synchronize" );

         if( !distributedGrid->isDistributed() ) return;

         // allocate buffers (setSize does nothing if the array size is already correct)
         for( int i=0; i<this->getNeighborCount(); i++ ) {
            sendBuffers[ i ].setSize( sendSizes[ i ] * sizeof(RealType) );
            recieveBuffers[ i ].setSize( sendSizes[ i ] * sizeof(RealType));
         }

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
         MPI_Request requests[2*this->getNeighborCount()];
         MPI_Comm group = distributedGrid->getCommunicationGroup();
         int requestsCount( 0 );

         //send everything, recieve everything
         for( int i=0; i<this->getNeighborCount(); i++ )
         {
            /*TNL_MPI_PRINT( "Sending data... " << i << " sizes -> "
               << sendSizes[ i ] << "sendDimensions -> " <<  sendDimensions[ i ]
               << " upperOverlap -> " << this->distributedGrid->getUpperOverlap() );*/
            if( neighbors[ i ] != -1 )
            {
               //TNL_MPI_PRINT( "Sending data to node " << neighbors[ i ] );
               requests[ requestsCount++ ] = MPI::Isend( reinterpret_cast<RealType*>( sendBuffers[ i ].getData() ),  sendSizes[ i ], neighbors[ i ], 0, group );
               //TNL_MPI_PRINT( "Receiving data from node " << neighbors[ i ] );
               requests[ requestsCount++ ] = MPI::Irecv( reinterpret_cast<RealType*>( recieveBuffers[ i ].getData() ),  sendSizes[ i ], neighbors[ i ], 0, group );
            }
            else if( periodicBoundaries && sendSizes[ i ] !=0 )
      	   {
               //TNL_MPI_PRINT( "Sending data to node " << periodicNeighbors[ i ] );
               requests[ requestsCount++ ] = MPI::Isend( reinterpret_cast<RealType*>( sendBuffers[ i ].getData() ),  sendSizes[ i ], periodicNeighbors[ i ], 1, group );
               //TNL_MPI_PRINT( "Receiving data to node " << periodicNeighbors[ i ] );
               requests[ requestsCount++ ] = MPI::Irecv( reinterpret_cast<RealType*>( recieveBuffers[ i ].getData() ),  sendSizes[ i ], periodicNeighbors[ i ], 1, group );
            }
         }

        //wait until send is done
        //TNL_MPI_PRINT( "Waiting for data ..." )
        MPI::Waitall( requests, requestsCount );

        //copy data from receive buffers
        //TNL_MPI_PRINT( "Copying data ..." )
        copyBuffers(meshFunction,
            recieveBuffers,recieveBegin,sendDimensions  ,
            false,
            neighbors,
            periodicBoundaries,
            mask );
    }

   private:
      template< typename MeshFunctionType,
                typename PeriodicBoundariesMaskPointer >
      void copyBuffers(
         MeshFunctionType& meshFunction,
         Containers::Array<std::uint8_t, Device, Index>* buffers,
         CoordinatesType* begins,
         CoordinatesType* sizes,
         bool toBuffer,
         const int* neighbor,
         bool periodicBoundaries,
         const PeriodicBoundariesMaskPointer& mask )
      {
         using RealType = typename MeshFunctionType::RealType;
         using Helper = BufferEntitiesHelper< MeshFunctionType, PeriodicBoundariesMaskPointer, getMeshDimension(), RealType, Device >;

         for(int i=0;i<this->getNeighborCount();i++)
         {
            bool isBoundary=( neighbor[ i ] == -1 );
            if( ! isBoundary || periodicBoundaries )
            {
               Helper::BufferEntities( meshFunction, mask, reinterpret_cast<RealType*>( buffers[ i ].getData() ), isBoundary, begins[i], sizes[i], toBuffer );
            }
         }
      }

   private:

      Containers::StaticArray< getNeighborCount(), int > sendSizes;
      Containers::Array< std::uint8_t, Device, Index > sendBuffers[getNeighborCount()];
      Containers::Array< std::uint8_t, Device, Index > recieveBuffers[getNeighborCount()];

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

