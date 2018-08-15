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
      typedef typename Functions::MeshFunction< Grid< 2, GridReal, Device, Index >,EntityDimension, RealType> MeshFunctionType;
      typedef typename Grid< 2, GridReal, Device, Index >::DistributedMeshType DistributedGridType; 
      typedef typename MeshFunctionType::RealType Real;
      typedef typename DistributedGridType::CoordinatesType CoordinatesType;
      template< typename Real_ >
      using BufferEntitiesHelperType = BufferEntitiesHelper< MeshFunctionType, 2, Real_, Device >;
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
            sendbuffs[ i ].setSize( sizes[ i ] );
            rcvbuffs[ i ].setSize( sizes[ i ] );
         }

         leftSrc  = lowerOverlap.x();
         rightSrc = localGridSize.x() - 2 * upperOverlap.x();
         upSrc    = lowerOverlap.y();                             // TODO: SWAP up and down
         downSrc  = localGridSize.y() - 2 * upperOverlap.y();     // TODO: SWAP up and down

         xcenter  = lowerOverlap.x();
         ycenter  = lowerOverlap.y();

         leftDst  = 0;
         rightDst = localGridSize.x() - upperOverlap.x();
         upDst    = 0;
         downDst  = localGridSize.y() - upperOverlap.y();                       
      }

      template<typename CommunicatorType>
      void synchronize( MeshFunctionType &meshFunction,
                        bool periodicBoundaries = false )
      {

         TNL_ASSERT_TRUE( isSet, "Synchronizer is not set, but used to synchronize" );

         if( !distributedGrid->isDistributed() )
            return;
         
         const SubdomainOverlapsType& lowerOverlap = distributedGrid->getLowerOverlap();
         const SubdomainOverlapsType& upperOverlap = distributedGrid->getUpperOverlap();
         const CoordinatesType& localSize = distributedGrid->getLocalSize();
         
         const int *neighbors = distributedGrid->getNeighbors();
         const int *periodicNeighbors = distributedGrid->getPeriodicNeighbors();
         
         copyBuffers(meshFunction, sendbuffs, true,
            leftSrc, rightSrc, upSrc, downSrc,
            xcenter, ycenter,
            lowerOverlap, upperOverlap, localSize,
            neighbors);

         //async send and rcv
         typename CommunicatorType::Request req[16];
         typename CommunicatorType::CommunicationGroup group;
         group=*((typename CommunicatorType::CommunicationGroup *)(distributedGrid->getCommunicationGroup()));
         int requestsCount( 0 );
         
         //send everything, recieve everything 
         for( int i = 0; i < 8; i++ )
            if( neighbors[ i ] != -1 )
            {
               req[ requestsCount++ ] = CommunicatorType::ISend( sendbuffs[ i ].getData(), sizes[ i ], neighbors[ i ], group );
               req[ requestsCount++ ] = CommunicatorType::IRecv( rcvbuffs[ i ].getData(), sizes[ i ], neighbors[ i ], group );
            }
            else if( periodicBoundaries )
            {
               req[ requestsCount++ ] = CommunicatorType::ISend( sendbuffs[ i ].getData(), sizes[ i ], periodicNeighbors[ i ], group );
               req[ requestsCount++ ] = CommunicatorType::IRecv( rcvbuffs[ i ].getData(), sizes[ i ], periodicNeighbors[ i ], group );
            }

         //wait until send is done
         CommunicatorType::WaitAll( req, requestsCount );

         //copy data form rcv buffers
         copyBuffers(meshFunction, rcvbuffs, false,
              leftDst, rightDst, upDst, downDst,
              xcenter, ycenter,
              lowerOverlap, upperOverlap, localSize,
              neighbors);
      }
    
   private:
      
      template< typename Real_ >
      void copyBuffers(MeshFunctionType meshFunction, Containers::Array<Real_, Device, Index> * buffers, bool toBuffer,
                       int left, int right, int up, int down,
                       int xcenter, int ycenter,
                       const CoordinatesType& lowerOverlap,
                       const CoordinatesType& upperOverlap,
                       const CoordinatesType& localSize,
                       const int *neighbors )
      {
         // TODO: SWAP up and down
         using Helper = BufferEntitiesHelper< MeshFunctionType, 2, Real_, Device >;
         if( neighbors[ Left ] != -1 )
            Helper::BufferEntities( meshFunction, buffers[ Left ].getData(), left, ycenter, lowerOverlap.x(), localSize.y(), toBuffer );
         if( neighbors[ Right ] != -1)
            Helper::BufferEntities( meshFunction, buffers[ Right ].getData(), right, ycenter, upperOverlap.x(), localSize.y(), toBuffer );
         if( neighbors[ Up ] != -1 )
            Helper::BufferEntities( meshFunction, buffers[ Up ].getData(), xcenter, up, localSize.x(), lowerOverlap.y(), toBuffer );
         if( neighbors[ Down ] != -1 )
            Helper::BufferEntities( meshFunction, buffers[ Down ].getData(), xcenter, down, localSize.x(), upperOverlap.y(), toBuffer );
         if( neighbors[ UpLeft ] != -1 )
            Helper::BufferEntities( meshFunction, buffers[ UpLeft ].getData(), left, up, lowerOverlap.x(), lowerOverlap.y(), toBuffer );
         if( neighbors[ UpRight ] != -1 )
            Helper::BufferEntities( meshFunction, buffers[ UpRight ].getData(), right, up, upperOverlap.x(), lowerOverlap.y(), toBuffer );
         if( neighbors[ DownLeft ] != -1 )        
            Helper::BufferEntities( meshFunction, buffers[ DownLeft ].getData(), left, down, lowerOverlap.x(), upperOverlap.y(), toBuffer );
         if( neighbors[ DownRight ] != -1 )
            Helper::BufferEntities( meshFunction, buffers[ DownRight ].getData(), right, down, upperOverlap.x(), upperOverlap.y(), toBuffer );
      }
      
      DistributedGridType *distributedGrid;

      Containers::Array<RealType, Device, Index> sendbuffs[8];
      Containers::Array<RealType, Device, Index> rcvbuffs[8];
      Containers::StaticArray< 8, int > sizes;

      int leftSrc;
      int rightSrc;
      int upSrc;
      int downSrc;
      int xcenter;
      int ycenter;
      int leftDst;
      int rightDst;
      int upDst;
      int downDst;

      bool isSet;      
};


} // namespace DistributedMeshes
} // namespace Meshes
} // namespace TNL


