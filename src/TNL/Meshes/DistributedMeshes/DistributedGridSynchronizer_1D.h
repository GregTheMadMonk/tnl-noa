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
      typedef typename Functions::MeshFunction< Grid< 1, GridReal, Device, Index >,EntityDimension, RealType> MeshFunctionType;
      typedef typename Grid< 1, GridReal, Device, Index >::DistributedMeshType DistributedGridType;
      //template< typename Real_ >
      //using BufferEntitiesHelperType = BufferEntitiesHelper< MeshFunctionType, 1, Real_, Device >;
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

         sendbuffs[ Left ].setSize( lowerOverlap.x() );
         sendbuffs[ Right ].setSize( upperOverlap.x() );
         rcvbuffs[ Left ].setSize( lowerOverlap.x() );
         rcvbuffs[ Right ].setSize( upperOverlap.x() );
      };

      template<typename CommunicatorType>
      void synchronize( MeshFunctionType &meshFunction,
                        bool periodicBoundaries = false )
      {
         TNL_ASSERT_TRUE( isSet, "Synchronizer is not set, but used to synchronize" );

         if( !distributedGrid->isDistributed() )
            return;

         const int *neighbors = distributedGrid->getNeighbors();
         const int *periodicNeighbors = distributedGrid->getPeriodicNeighbors();

         int totalSize = meshFunction.getMesh().getDimensions().x();
         const SubdomainOverlapsType& lowerOverlap = distributedGrid->getLowerOverlap();
         const SubdomainOverlapsType& upperOverlap = distributedGrid->getUpperOverlap();
         
         copyBuffers( meshFunction, sendbuffs, true,
                      lowerOverlap.x(),
                      totalSize - 2 * upperOverlap.x(),
                      lowerOverlap, upperOverlap,
                      neighbors,
                      periodicBoundaries );

         //async send
         typename CommunicatorType::Request requests[ 4 ];
         typename CommunicatorType::CommunicationGroup group;
         group=*((typename CommunicatorType::CommunicationGroup *)(distributedGrid->getCommunicationGroup()));
         int requestsCount( 0 );
         
         //send everything, recieve everything 
         if( neighbors[ Left ] != -1 )
         {
            requests[ requestsCount++ ] = CommunicatorType::ISend( sendbuffs[ Left ].getData(), lowerOverlap.x(), neighbors[ Left ], group );
            requests[ requestsCount++ ] = CommunicatorType::IRecv( rcvbuffs[ Left ].getData(), lowerOverlap.x(), neighbors[ Left ], group );
         }
         else if( periodicBoundaries )
         {
            requests[ requestsCount++ ] = CommunicatorType::ISend( sendbuffs[ Left ].getData(), lowerOverlap.x(), periodicNeighbors[ Left ], group );
            requests[ requestsCount++ ] = CommunicatorType::IRecv( rcvbuffs[ Left ].getData(), lowerOverlap.x(), periodicNeighbors[ Left ], group );
         }        

         if( neighbors[ Right ] != -1 )
         {
            requests[ requestsCount++ ] = CommunicatorType::ISend( sendbuffs[ Right ].getData(), upperOverlap.x(), neighbors[ Right ], group );
            requests[ requestsCount++ ] = CommunicatorType::IRecv( rcvbuffs[ Right ].getData(), upperOverlap.x(), neighbors[ Right ], group );
         }
         else if( periodicBoundaries )
         {
            requests[ requestsCount++ ] = CommunicatorType::ISend( sendbuffs[ Right ].getData(), upperOverlap.x(), periodicNeighbors[ Right ], group );
            requests[ requestsCount++ ] = CommunicatorType::IRecv( rcvbuffs[ Right ].getData(), upperOverlap.x(), periodicNeighbors[ Right ], group );
         }
         
         //wait until send and recv is done
         CommunicatorType::WaitAll( requests, requestsCount );

         copyBuffers( meshFunction, rcvbuffs, false,
            0, totalSize - upperOverlap.x(),
            lowerOverlap,
            upperOverlap,
            neighbors,
            periodicBoundaries );
      }
      
   private:
      template <typename Real_ >
      void copyBuffers( MeshFunctionType meshFunction, TNL::Containers::Array<Real_,Device>* buffers, bool toBuffer,
         int left, int right,
         const SubdomainOverlapsType& lowerOverlap,
         const SubdomainOverlapsType& upperOverlap,
         const int* neighbors,
         bool periodicBoundaries )
      
      {
         typedef BufferEntitiesHelper< MeshFunctionType, 1, Real_, Device > Helper;
         if( neighbors[ Left ] != -1 )
            Helper::BufferEntities( meshFunction, buffers[ Left ].getData(), left, lowerOverlap.x(), toBuffer );
         else if( periodicBoundaries )
            Helper::BufferEntities( meshFunction, buffers[ Left ].getData(), left, lowerOverlap.x(), toBuffer );

         if( neighbors[ Right ] != -1 )
            Helper::BufferEntities( meshFunction, buffers[ Right ].getData(), right, upperOverlap.x(), toBuffer );
         else if( periodicBoundaries )
            Helper::BufferEntities( meshFunction, buffers[ Right ].getData(), right, upperOverlap.x(), toBuffer );
      }

      Containers::Array<RealType, Device> sendbuffs[ 2 ], rcvbuffs[ 2 ];

      DistributedGridType *distributedGrid;

      bool isSet;
};

} // namespace DistributedMeshes
} // namespace Meshes
} // namespace TNL
