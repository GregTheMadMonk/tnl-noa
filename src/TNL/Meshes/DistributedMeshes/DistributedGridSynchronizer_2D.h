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
         
         sizes[ ZzYzXm ]      = localSize.y()    * lowerOverlap.x();
         sizes[ ZzYzXp ]     = localSize.y()    * upperOverlap.x();
         sizes[ ZzYmXz ]        = localSize.x()    * lowerOverlap.y();
         sizes[ ZzYpXz ]      = localSize.x()    * upperOverlap.y();
         sizes[ ZzYmXm ]    = lowerOverlap.x() * lowerOverlap.y();
         sizes[ ZzYpXm ]  = lowerOverlap.x() * upperOverlap.y();
         sizes[ ZzYmXp ]   = upperOverlap.x() * lowerOverlap.y();
         sizes[ ZzYpXp ] = upperOverlap.x() * upperOverlap.y();
          
         for(int i=0;i<8;i++)
         {
            sendBuffers[ i ].setSize( sizes[ i ] );
            receiveBuffers[ i ].setSize( sizes[ i ] );
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

         if( !distributedGrid->isDistributed() )
            return;
         
         const SubdomainOverlapsType& lowerOverlap = distributedGrid->getLowerOverlap();
         const SubdomainOverlapsType& upperOverlap = distributedGrid->getUpperOverlap();
         const CoordinatesType& localSize = distributedGrid->getLocalSize();
         const CoordinatesType& localGridSize = this->distributedGrid->getLocalGridSize();
         
         ZzYzXm_Source  = lowerOverlap.x();
         ZzYzXp_Source = localGridSize.x() - 2 * upperOverlap.x();
         ZzYmXz_Source    = lowerOverlap.y();                             
         ZzYpXz_Source  = localGridSize.y() - 2 * upperOverlap.y();     

         xCenter  = lowerOverlap.x();
         yCenter  = lowerOverlap.y();

         ZzYzXm_Destination  = 0;
         ZzYzXp_Destination = localGridSize.x() - upperOverlap.x();
         ZzYmXz_Destination    = 0;
         ZzYpXz_Destination  = localGridSize.y() - upperOverlap.y();                       
         
         const int *neighbors = distributedGrid->getNeighbors();
         const int *periodicNeighbors = distributedGrid->getPeriodicNeighbors();
         
         if( periodicBoundaries )
         {
            if( neighbors[ ZzYzXm ] == -1 )
               swap( ZzYzXm_Source, ZzYzXm_Destination );
            if( neighbors[ ZzYzXp ] == -1 )
               swap( ZzYzXp_Source, ZzYzXp_Destination );
            if( neighbors[ ZzYmXz ] == -1 )
               swap( ZzYmXz_Source, ZzYmXz_Destination );
            if( neighbors[ ZzYpXz ] == -1 )
               swap( ZzYpXz_Source, ZzYpXz_Destination );
         }
         
         copyBuffers(meshFunction, sendBuffers, true,
            ZzYzXm_Source, ZzYzXp_Source, ZzYmXz_Source, ZzYpXz_Source,
            xCenter, yCenter,
            lowerOverlap, upperOverlap, localSize,
            neighbors,
            periodicBoundaries,
            PeriodicBoundariesMaskPointer( nullptr ) ); // the mask is used only when receiving data

         //async send and receive
         typename CommunicatorType::Request requests[ 16 ];
         typename CommunicatorType::CommunicationGroup group;
         group=*((typename CommunicatorType::CommunicationGroup *)(distributedGrid->getCommunicationGroup()));
         int requestsCount( 0 );
         
         //send everything, receive everything 
         for( int i = 0; i < 8; i++ )
         {
            if( neighbors[ i ] != -1 )
            {
               requests[ requestsCount++ ] = CommunicatorType::ISend( sendBuffers[ i ].getData(), sizes[ i ], neighbors[ i ], 0, group );
               requests[ requestsCount++ ] = CommunicatorType::IRecv( receiveBuffers[ i ].getData(), sizes[ i ], neighbors[ i ], 0, group );
            }
            else if( periodicBoundaries )
            {
               requests[ requestsCount++ ] = CommunicatorType::ISend( sendBuffers[ i ].getData(), sizes[ i ], periodicNeighbors[ i ], 1, group );
               requests[ requestsCount++ ] = CommunicatorType::IRecv( receiveBuffers[ i ].getData(), sizes[ i ], periodicNeighbors[ i ], 1, group );
            }
         }

         //wait until send is done
         CommunicatorType::WaitAll( requests, requestsCount );

         //copy data from receive buffers
         copyBuffers(meshFunction, receiveBuffers, false,
              ZzYzXm_Destination, ZzYzXp_Destination, ZzYmXz_Destination, ZzYpXz_Destination,
              xCenter, yCenter,
              lowerOverlap, upperOverlap, localSize,
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
         bool toBuffer,
         int ZzYzXm_Position, int ZzYzXp_Position, int ZzYmXz_Position, int ZzYpXz_Position,
         int xcenter, int ycenter,
         const CoordinatesType& lowerOverlap,
         const CoordinatesType& upperOverlap,
         const CoordinatesType& localSize,
         const int *neighbors,
         bool periodicBoundaries,
         const PeriodicBoundariesMaskPointer& mask )
      {
         bool ZzYzXm_IsBoundary = ( neighbors[ ZzYzXm ] == -1 );
         bool ZzYzXp_IsBoundary = ( neighbors[ ZzYzXp ] == -1 );
         bool ZzYmXz_IsBoundary = ( neighbors[ ZzYmXz ] == -1 );
         bool ZzYpXz_IsBoundary = ( neighbors[ ZzYpXz ] == -1 );
         bool ZzYmXm_IsBoundary = ( neighbors[ ZzYmXm ] == -1 );
         bool ZzYmXp_IsBoundary = ( neighbors[ ZzYmXp ] == -1 );
         bool ZzYpXm_IsBoundary = ( neighbors[ ZzYpXm ] == -1 );
         bool ZzYpXp_IsBoundary = ( neighbors[ ZzYpXp ] == -1 );
         
         using Helper = BufferEntitiesHelper< MeshFunctionType, PeriodicBoundariesMaskPointer, 2, Real_, Device >;
         if( ! ZzYzXm_IsBoundary || periodicBoundaries )
            Helper::BufferEntities( meshFunction, mask, buffers[ ZzYzXm      ].getData(), ZzYzXm_IsBoundary,      ZzYzXm_Position,    ycenter, lowerOverlap.x(), localSize.y(),    toBuffer );
         if( ! ZzYzXp_IsBoundary || periodicBoundaries )
            Helper::BufferEntities( meshFunction, mask, buffers[ ZzYzXp     ].getData(), ZzYzXp_IsBoundary,     ZzYzXp_Position,   ycenter, upperOverlap.x(), localSize.y(),    toBuffer );
         if( ! ZzYmXz_IsBoundary || periodicBoundaries )
            Helper::BufferEntities( meshFunction, mask, buffers[ ZzYmXz        ].getData(), ZzYmXz_IsBoundary,        xcenter, ZzYmXz_Position,      localSize.x(),    lowerOverlap.y(), toBuffer );
         if( ! ZzYpXz_IsBoundary || periodicBoundaries )
            Helper::BufferEntities( meshFunction, mask, buffers[ ZzYpXz      ].getData(), ZzYpXz_IsBoundary,      xcenter, ZzYpXz_Position,    localSize.x(),    upperOverlap.y(), toBuffer );
         if( ! ZzYmXm_IsBoundary || periodicBoundaries )
            Helper::BufferEntities( meshFunction, mask, buffers[ ZzYmXm    ].getData(), ZzYmXm_IsBoundary,    ZzYzXm_Position,    ZzYmXz_Position,      lowerOverlap.x(), lowerOverlap.y(), toBuffer );
         if( ! ZzYmXp_IsBoundary || periodicBoundaries )
            Helper::BufferEntities( meshFunction, mask, buffers[ ZzYmXp   ].getData(), ZzYmXp_IsBoundary,   ZzYzXp_Position,   ZzYmXz_Position,      upperOverlap.x(), lowerOverlap.y(), toBuffer );
         if( ! ZzYpXm_IsBoundary || periodicBoundaries )        
            Helper::BufferEntities( meshFunction, mask, buffers[ ZzYpXm  ].getData(), ZzYpXm_IsBoundary,  ZzYzXm_Position,    ZzYpXz_Position,    lowerOverlap.x(), upperOverlap.y(), toBuffer );
         if( ! ZzYpXp_IsBoundary || periodicBoundaries )
            Helper::BufferEntities( meshFunction, mask, buffers[ ZzYpXp ].getData(), ZzYpXp_IsBoundary, ZzYzXp_Position,   ZzYpXz_Position,    upperOverlap.x(), upperOverlap.y(), toBuffer );
      }
      
      DistributedGridType *distributedGrid;

      Containers::Array<RealType, Device, Index> sendBuffers[8];
      Containers::Array<RealType, Device, Index> receiveBuffers[8];
      Containers::StaticArray< 8, int > sizes;

      int ZzYzXm_Source;
      int ZzYzXp_Source;
      int ZzYmXz_Source;
      int ZzYpXz_Source;
      int xCenter;
      int yCenter;
      int ZzYzXm_Destination;
      int ZzYzXp_Destination;
      int ZzYmXz_Destination;
      int ZzYpXz_Destination;

      bool isSet;      
};

} // namespace DistributedMeshes
} // namespace Meshes
} // namespace TNL


