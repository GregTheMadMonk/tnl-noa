/***************************************************************************
                          DistributedGridSynchronizer_3D.h  -  description
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
class DistributedMeshSynchronizer< Functions::MeshFunction< Grid< 3, GridReal, Device, Index >,EntityDimension, RealType>>
{

   public:
      typedef typename Grid< 3, GridReal, Device, Index >::Cell Cell;
      // FIXME: clang does not like this (incomplete type error)
//      typedef typename Functions::MeshFunction< Grid< 3, GridReal, Device, Index >,EntityDimension, RealType> MeshFunctionType;
      typedef typename Grid< 3, GridReal, Device, Index >::DistributedMeshType DistributedGridType; 
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

      void setDistributedGrid( DistributedGridType *distributedGrid )
      {
         isSet = true;

         this->distributedGrid = distributedGrid;
         
         const SubdomainOverlapsType& lowerOverlap = distributedGrid->getLowerOverlap();
         const SubdomainOverlapsType& upperOverlap = distributedGrid->getUpperOverlap();
         const CoordinatesType& localSize = distributedGrid->getLocalSize(); 
         const CoordinatesType& localGridSize = this->distributedGrid->getLocalGridSize();         

         sizes[ ZzYzXm ]   = localSize.y() * localSize.z() * lowerOverlap.x();
         sizes[ ZzYzXp ]   = localSize.y() * localSize.z() * upperOverlap.x();
         sizes[ ZzYmXz ]  = localSize.x() * localSize.z() * lowerOverlap.y();
         sizes[ ZzYpXz ]  = localSize.x() * localSize.z() * upperOverlap.y();
         sizes[ ZmYzXz ] = localSize.x() * localSize.y() * lowerOverlap.z();
         sizes[ ZpYzXz ]    = localSize.x() * localSize.y() * upperOverlap.z();
  
         sizes[ ZzYmXm ]   = localSize.z() * lowerOverlap.x() * lowerOverlap.y();
         sizes[ ZzYmXp ]   = localSize.z() * upperOverlap.x() * lowerOverlap.y();
         sizes[ ZzYpXm ]   = localSize.z() * lowerOverlap.x() * upperOverlap.y();
         sizes[ ZzYpXp ]   = localSize.z() * upperOverlap.x() * upperOverlap.y();
         sizes[ ZmYzXp ]  = localSize.y() * lowerOverlap.x() * lowerOverlap.z();
         sizes[ ZmYzXm ]  = localSize.y() * upperOverlap.x() * lowerOverlap.z();
         sizes[ ZpYzXp ]     = localSize.y() * lowerOverlap.x() * upperOverlap.z();
         sizes[ ZpYzXm ]     = localSize.y() * upperOverlap.x() * upperOverlap.z();
         sizes[ ZmYmXz ] = localSize.x() * lowerOverlap.y() * lowerOverlap.z();
         sizes[ ZmYpXz ] = localSize.x() * upperOverlap.y() * lowerOverlap.z();
         sizes[ ZpYmXz ]    = localSize.x() * lowerOverlap.y() * upperOverlap.z();
         sizes[ ZpYpXz ]    = localSize.x() * upperOverlap.y() * upperOverlap.z();
        
         sizes[ ZmYmXm ] = lowerOverlap.x() * lowerOverlap.y() * lowerOverlap.z();
         sizes[ ZmYmXp ] = upperOverlap.x() * lowerOverlap.y() * lowerOverlap.z();
         sizes[ ZmYpXm ] = lowerOverlap.x() * upperOverlap.y() * lowerOverlap.z();
         sizes[ ZmYpXp ] = upperOverlap.x() * upperOverlap.y() * lowerOverlap.z();
         sizes[ ZpYmXm    ] = lowerOverlap.x() * lowerOverlap.y() * upperOverlap.z();
         sizes[ ZpYmXp    ] = upperOverlap.x() * lowerOverlap.y() * upperOverlap.z();
         sizes[ ZpYpXm    ] = lowerOverlap.x() * upperOverlap.y() * upperOverlap.z();
         sizes[ ZpYpXp    ] = upperOverlap.x() * upperOverlap.y() * upperOverlap.z();

         for( int i=0; i<26; i++ )
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

    	   if( !distributedGrid->isDistributed() ) return;
         
         const SubdomainOverlapsType& lowerOverlap = distributedGrid->getLowerOverlap();
         const SubdomainOverlapsType& upperOverlap = distributedGrid->getUpperOverlap();
         const CoordinatesType& localSize = distributedGrid->getLocalSize(); 
         const CoordinatesType& localGridSize = this->distributedGrid->getLocalGridSize();
        
         ZzYzXm_Source   = lowerOverlap.x();
         ZzYzXp_Source   = localGridSize.x() - 2 * upperOverlap.x();
         ZzYmXz_Source  = lowerOverlap.y();
         ZzYpXz_Source  = localGridSize.y() - 2 * upperOverlap.y();
         ZmYzXz_Source = lowerOverlap.z();
         ZpYzXz_Source    = localGridSize.z() - 2 * upperOverlap.z();
            
         xCenter = lowerOverlap.x();
         yCenter = lowerOverlap.y();
         zCenter = lowerOverlap.z();
        
         ZzYzXm_Destination   = 0;
         ZzYzXp_Destination   = localGridSize.x() - upperOverlap.x();
         ZzYmXz_Destination  = 0;
         ZzYpXz_Destination  = localGridSize.y() - upperOverlap.y();
         ZmYzXz_Destination = 0;
         ZpYzXz_Destination    = localGridSize.z() - upperOverlap.z();         
        
         const int *neighbors = distributedGrid->getNeighbors();
         const int *periodicNeighbors = distributedGrid->getPeriodicNeighbors();
         
        if( periodicBoundaries )
         {
            if( neighbors[ ZzYzXm ] == -1 )
               swap( ZzYzXm_Source, ZzYzXm_Destination );
            if( neighbors[ ZzYzXp ] == -1 )
               swap( ZzYzXp_Source, ZzYzXp_Destination );
            if( neighbors[ ZzYpXz ] == -1 )
               swap( ZzYpXz_Source, ZzYpXz_Destination );
            if( neighbors[ ZzYmXz ] == -1 )
               swap( ZzYmXz_Source, ZzYmXz_Destination );
            if( neighbors[ ZmYzXz ] == -1 )
               swap( ZmYzXz_Source, ZmYzXz_Destination );
            if( neighbors[ ZpYzXz ] == -1 )
               swap( ZpYzXz_Source, ZpYzXz_Destination );            
         }         
        
         //fill send buffers
         copyBuffers( meshFunction, sendBuffers, true,
            ZzYzXm_Source, ZzYzXp_Source, ZzYmXz_Source, ZzYpXz_Source, ZmYzXz_Source, ZpYzXz_Source,
            xCenter, yCenter, zCenter,
            lowerOverlap, upperOverlap, localSize,
            neighbors,
            periodicBoundaries,
            PeriodicBoundariesMaskPointer( nullptr ) ); // the mask is used only when receiving data );
        
         //async send and receive
         typename CommunicatorType::Request requests[52];
         typename CommunicatorType::CommunicationGroup group;
         group=*((typename CommunicatorType::CommunicationGroup *)(distributedGrid->getCommunicationGroup()));
         int requestsCount( 0 );
		                
         //send everything, receive everything
         for( int i = 0; i <  26; i++ )
            if( neighbors[ i ] != -1 )
            {
               requests[ requestsCount++ ] = CommunicatorType::ISend( sendBuffers[ i ].getData(),  sizes[ i ], neighbors[ i ], 0, group );
               requests[ requestsCount++ ] = CommunicatorType::IRecv( receiveBuffers[ i ].getData(),  sizes[ i ], neighbors[ i ], 0, group );
            }
            else if( periodicBoundaries )
      	   {
               requests[ requestsCount++ ] = CommunicatorType::ISend( sendBuffers[ i ].getData(),  sizes[ i ], periodicNeighbors[ i ], 1, group );
               requests[ requestsCount++ ] = CommunicatorType::IRecv( receiveBuffers[ i ].getData(),  sizes[ i ], periodicNeighbors[ i ], 1, group );
            }

        //wait until send is done
        CommunicatorType::WaitAll( requests, requestsCount );

        //copy data from receive buffers
        copyBuffers(meshFunction, receiveBuffers, false,
            ZzYzXm_Destination, ZzYzXp_Destination, ZzYmXz_Destination, ZzYpXz_Destination, ZmYzXz_Destination, ZpYzXz_Destination,
            xCenter, yCenter, zCenter,
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
         int ZzYzXm, int ZzYzXp, int ZzYmXz, int ZzYpXz, int ZmYzXz, int ZpYzXz,
         int xcenter, int ycenter, int zcenter,
         const CoordinatesType& lowerOverlap,
         const CoordinatesType& upperOverlap,
         const CoordinatesType& localSize,
         const int* neighbor,
         bool periodicBoundaries,
         const PeriodicBoundariesMaskPointer& mask )
      {
         //TODO: refactor to array IsBoundary[number of neighbor]
         bool ZzYzXm_IsBoundary = ( neighbor[ ZzYzXm ] == -1 );
         bool ZzYzXp_IsBoundary = ( neighbor[ ZzYzXp ] == -1 );
         bool ZzYmXz_IsBoundary = ( neighbor[ ZzYmXz ] == -1 );
         bool ZzYpXz_IsBoundary = ( neighbor[ ZzYpXz ] == -1 );
         bool ZmYzXz_IsBoundary = ( neighbor[ ZmYzXz ] == -1 );
         bool ZpYzXz_IsBoundary = ( neighbor[ ZpYzXz ] == -1 );

         bool ZzYmXm_IsBoundary = ( neighbor[ ZzYmXm ] == -1 );
         bool ZzYmXp_IsBoundary = ( neighbor[ ZzYmXp ] == -1 );
         bool ZzYpXm_IsBoundary = ( neighbor[ ZzYpXm ] == -1 );
         bool ZzYpXp_IsBoundary = ( neighbor[ ZzYpXp ] == -1 );
         
         bool ZmYzXp_IsBoundary = ( neighbor[ ZmYzXp ] == -1 );
         bool ZmYzXm_IsBoundary = ( neighbor[ ZmYzXm ] == -1 );
         bool ZmYmXz_IsBoundary = ( neighbor[ ZmYmXz ] == -1 );
         bool ZmYpXz_IsBoundary = ( neighbor[ ZmYpXz ] == -1 );

         bool ZpYzXp_IsBoundary = ( neighbor[ ZpYzXp ] == -1 );
         bool ZpYzXm_IsBoundary = ( neighbor[ ZpYzXm ] == -1 );
         bool ZpYmXz_IsBoundary = ( neighbor[ ZpYmXz ] == -1 );
         bool ZpYpXz_IsBoundary = ( neighbor[ ZpYpXz ] == -1 );

         bool ZmYmXm_IsBoundary = ( neighbor[ ZmYmXm ] == -1 );
         bool ZmYmXp_IsBoundary = ( neighbor[ ZmYmXp ] == -1 );
         bool ZmYpXm_IsBoundary = ( neighbor[ ZmYpXm ] == -1 );
         bool ZmYpXp_IsBoundary = ( neighbor[ ZmYpXp ] == -1 );

         bool ZpYmXm_IsBoundary = ( neighbor[ ZpYmXm ] == -1 );
         bool ZpYmXp_IsBoundary = ( neighbor[ ZpYmXp ] == -1 );
         bool ZpYpXm_IsBoundary = ( neighbor[ ZpYpXm ] == -1 );
         bool ZpYpXp_IsBoundary = ( neighbor[ ZpYpXp ] == -1 );
         
         using Helper = BufferEntitiesHelper< MeshFunctionType, PeriodicBoundariesMaskPointer, 3, Real_, Device >;
         //X-Y-Z
         if( ! ZzYzXm_IsBoundary || periodicBoundaries )
            Helper::BufferEntities( meshFunction, mask, buffers[ ZzYzXm ].getData(),   ZzYzXm_IsBoundary,   ZzYzXm,    ycenter, zcenter, lowerOverlap.x(), localSize.y(),     localSize.z(),    toBuffer );
         if( ! ZzYzXp_IsBoundary || periodicBoundaries )
            Helper::BufferEntities( meshFunction, mask, buffers[ ZzYzXp ].getData(),   ZzYzXp_IsBoundary,   ZzYzXp,    ycenter, zcenter, upperOverlap.x(), localSize.y(),     localSize.z(),    toBuffer );
         if( ! ZzYmXz_IsBoundary || periodicBoundaries )
            Helper::BufferEntities( meshFunction, mask, buffers[ ZzYmXz ].getData(),  ZzYmXz_IsBoundary,  xcenter, ZzYmXz,   zcenter, localSize.x(),    lowerOverlap.y(),  localSize.z(),    toBuffer );
         if( ! ZzYpXz_IsBoundary || periodicBoundaries )
            Helper::BufferEntities( meshFunction, mask, buffers[ ZzYpXz ].getData(),  ZzYpXz_IsBoundary,  xcenter, ZzYpXz,   zcenter, localSize.x(),     upperOverlap.y(), localSize.z(),    toBuffer );
         if( ! ZmYzXz_IsBoundary || periodicBoundaries )
            Helper::BufferEntities( meshFunction, mask, buffers[ ZmYzXz ].getData(), ZmYzXz_IsBoundary, xcenter, ycenter, ZmYzXz,  localSize.x(),     localSize.y(),    lowerOverlap.z(), toBuffer );
         if( ! ZpYzXz_IsBoundary || periodicBoundaries )
            Helper::BufferEntities( meshFunction, mask, buffers[ ZpYzXz ].getData(),    ZpYzXz_IsBoundary,    xcenter, ycenter, ZpYzXz,     localSize.x(),     localSize.y(),    upperOverlap.z(), toBuffer );	
         
         //XY
         if( ! ZzYmXm_IsBoundary || periodicBoundaries )
            Helper::BufferEntities( meshFunction, mask, buffers[ ZzYmXm ].getData(), ZzYmXm_IsBoundary, ZzYzXm, ZzYmXz, zcenter, lowerOverlap.x(), lowerOverlap.y(), localSize.z(), toBuffer );
         if( ! ZzYmXp_IsBoundary || periodicBoundaries )
            Helper::BufferEntities( meshFunction, mask, buffers[ ZzYmXp ].getData(), ZzYmXp_IsBoundary, ZzYzXp, ZzYmXz, zcenter, upperOverlap.x(), lowerOverlap.y(), localSize.z(), toBuffer );
         if( ! ZzYpXm_IsBoundary || periodicBoundaries )
            Helper::BufferEntities( meshFunction, mask, buffers[ ZzYpXm ].getData(), ZzYpXm_IsBoundary, ZzYzXm, ZzYpXz, zcenter, lowerOverlap.x(), upperOverlap.y(), localSize.z(), toBuffer );
         if( ! ZzYpXp_IsBoundary || periodicBoundaries )
            Helper::BufferEntities( meshFunction, mask, buffers[ ZzYpXp ].getData(), ZzYpXp_IsBoundary, ZzYzXp, ZzYpXz, zcenter, upperOverlap.x(), upperOverlap.y(), localSize.z(), toBuffer );
         
         //XZ
         if( ! ZmYzXp_IsBoundary || periodicBoundaries )
            Helper::BufferEntities( meshFunction, mask, buffers[ ZmYzXp ].getData(), ZmYzXp_IsBoundary, ZzYzXm, ycenter, ZmYzXz, lowerOverlap.x(), localSize.y(), lowerOverlap.z(), toBuffer );
         if( ! ZmYzXm_IsBoundary || periodicBoundaries )
            Helper::BufferEntities( meshFunction, mask, buffers[ ZmYzXm ].getData(), ZmYzXm_IsBoundary, ZzYzXp, ycenter, ZmYzXz, upperOverlap.x(), localSize.y(), lowerOverlap.z(), toBuffer );
         if( ! ZpYzXp_IsBoundary || periodicBoundaries )
            Helper::BufferEntities( meshFunction, mask, buffers[ ZpYzXp ].getData(),    ZpYzXp_IsBoundary,    ZzYzXm, ycenter, ZpYzXz,    lowerOverlap.x(), localSize.y(), upperOverlap.z(), toBuffer );
         if( ! ZpYzXm_IsBoundary || periodicBoundaries )
            Helper::BufferEntities( meshFunction, mask, buffers[ ZpYzXm ].getData(),    ZpYzXm_IsBoundary,    ZzYzXp, ycenter, ZpYzXz,    upperOverlap.x(), localSize.y(), upperOverlap.z(), toBuffer );   
         
         //YZ
         if( ! ZmYmXz_IsBoundary || periodicBoundaries )
            Helper::BufferEntities( meshFunction, mask, buffers[ ZmYmXz ].getData(), ZmYmXz_IsBoundary, xcenter, ZzYmXz, ZmYzXz, localSize.x(), lowerOverlap.y(), lowerOverlap.z(), toBuffer );
         if( ! ZmYpXz_IsBoundary || periodicBoundaries )
            Helper::BufferEntities( meshFunction, mask, buffers[ ZmYpXz ].getData(), ZmYpXz_IsBoundary, xcenter, ZzYpXz, ZmYzXz, localSize.x(), upperOverlap.y(), lowerOverlap.z(), toBuffer );
         if( ! ZpYmXz_IsBoundary || periodicBoundaries )
            Helper::BufferEntities( meshFunction, mask, buffers[ ZpYmXz ].getData(),    ZpYmXz_IsBoundary,    xcenter, ZzYmXz, ZpYzXz,    localSize.x(), lowerOverlap.y(), upperOverlap.z(), toBuffer );
         if( ! ZpYpXz_IsBoundary || periodicBoundaries )
            Helper::BufferEntities( meshFunction, mask, buffers[ ZpYpXz ].getData(),    ZpYpXz_IsBoundary,    xcenter, ZzYpXz, ZpYzXz,    localSize.x(), upperOverlap.y(), upperOverlap.z(), toBuffer );
         
         //XYZ
         if( ! ZmYmXm_IsBoundary || periodicBoundaries )
            Helper::BufferEntities( meshFunction, mask, buffers[ ZmYmXm ].getData(), ZmYmXm_IsBoundary, ZzYzXm, ZzYmXz, ZmYzXz, lowerOverlap.x(), lowerOverlap.y(), lowerOverlap.z(), toBuffer );
         if( ! ZmYmXp_IsBoundary || periodicBoundaries )
            Helper::BufferEntities( meshFunction, mask, buffers[ ZmYmXp ].getData(), ZmYmXp_IsBoundary, ZzYzXp, ZzYmXz, ZmYzXz, upperOverlap.x(), lowerOverlap.y(), lowerOverlap.z(), toBuffer );
         if( ! ZmYpXm_IsBoundary || periodicBoundaries )
            Helper::BufferEntities( meshFunction, mask, buffers[ ZmYpXm ].getData(), ZmYpXm_IsBoundary, ZzYzXm, ZzYpXz, ZmYzXz, lowerOverlap.x(), upperOverlap.y(), lowerOverlap.z(), toBuffer );
         if( ! ZmYpXp_IsBoundary || periodicBoundaries )
            Helper::BufferEntities( meshFunction, mask, buffers[ ZmYpXp ].getData(), ZmYpXp_IsBoundary, ZzYzXp, ZzYpXz, ZmYzXz, upperOverlap.x(), upperOverlap.y(), lowerOverlap.z(), toBuffer );
         if( ! ZpYmXm_IsBoundary || periodicBoundaries )
            Helper::BufferEntities( meshFunction, mask, buffers[ ZpYmXm ].getData(),    ZpYmXm_IsBoundary,    ZzYzXm, ZzYmXz, ZpYzXz,    lowerOverlap.x(), lowerOverlap.y(), upperOverlap.z(), toBuffer );
         if( ! ZpYmXp_IsBoundary || periodicBoundaries )
            Helper::BufferEntities( meshFunction, mask, buffers[ ZpYmXp ].getData(),    ZpYmXp_IsBoundary,    ZzYzXp, ZzYmXz, ZpYzXz,    upperOverlap.x(), lowerOverlap.y(), upperOverlap.z(), toBuffer );
         if( ! ZpYpXm_IsBoundary || periodicBoundaries )
            Helper::BufferEntities( meshFunction, mask, buffers[ ZpYpXm ].getData(),    ZpYpXp_IsBoundary,    ZzYzXm, ZzYpXz, ZpYzXz,    lowerOverlap.x(), upperOverlap.y(), upperOverlap.z(), toBuffer );
         if( ! ZpYpXp_IsBoundary || periodicBoundaries )
            Helper::BufferEntities( meshFunction, mask, buffers[ ZpYpXp ].getData(),    ZpYpXp_IsBoundary,    ZzYzXp, ZzYpXz, ZpYzXz,    upperOverlap.x(), upperOverlap.y(), upperOverlap.z(), toBuffer );   
      }
    
   private:
   
      Containers::Array<RealType, Device, Index> sendBuffers[26];
      Containers::Array<RealType, Device, Index> receiveBuffers[26];
      Containers::StaticArray< 26, int > sizes;
      
      DistributedGridType *distributedGrid;
        
      int ZzYzXm_Source;
      int ZzYzXp_Source;
      int ZzYmXz_Source;
      int ZzYpXz_Source;
      int ZmYzXz_Source;
      int ZpYzXz_Source;
      int xCenter;
      int yCenter;
      int zCenter;
      int ZzYzXm_Destination;
      int ZzYzXp_Destination;
      int ZzYmXz_Destination;
      int ZzYpXz_Destination;
      int ZmYzXz_Destination;
      int ZpYzXz_Destination;
        
      CoordinatesType overlap;
      CoordinatesType localSize;

      bool isSet;
    
};


} // namespace DistributedMeshes
} // namespace Meshes
} // namespace TNL

