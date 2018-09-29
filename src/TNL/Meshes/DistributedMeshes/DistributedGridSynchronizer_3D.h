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

         sizes[ West ]   = localSize.y() * localSize.z() * lowerOverlap.x();
         sizes[ East ]   = localSize.y() * localSize.z() * upperOverlap.x();
         sizes[ North ]  = localSize.x() * localSize.z() * lowerOverlap.y();
         sizes[ South ]  = localSize.x() * localSize.z() * upperOverlap.y();
         sizes[ Bottom ] = localSize.x() * localSize.y() * lowerOverlap.z();
         sizes[ Top ]    = localSize.x() * localSize.y() * upperOverlap.z();
  
         sizes[ NorthWest ]   = localSize.z() * lowerOverlap.x() * lowerOverlap.y();
         sizes[ NorthEast ]   = localSize.z() * upperOverlap.x() * lowerOverlap.y();
         sizes[ SouthWest ]   = localSize.z() * lowerOverlap.x() * upperOverlap.y();
         sizes[ SouthEast ]   = localSize.z() * upperOverlap.x() * upperOverlap.y();
         sizes[ BottomWest ]  = localSize.y() * lowerOverlap.x() * lowerOverlap.z();
         sizes[ BottomEast ]  = localSize.y() * upperOverlap.x() * lowerOverlap.z();
         sizes[ TopWest ]     = localSize.y() * lowerOverlap.x() * upperOverlap.z();
         sizes[ TopEast ]     = localSize.y() * upperOverlap.x() * upperOverlap.z();
         sizes[ BottomNorth ] = localSize.x() * lowerOverlap.y() * lowerOverlap.z();
         sizes[ BottomSouth ] = localSize.x() * upperOverlap.y() * lowerOverlap.z();
         sizes[ TopNorth ]    = localSize.x() * lowerOverlap.y() * upperOverlap.z();
         sizes[ TopSouth ]    = localSize.x() * upperOverlap.y() * upperOverlap.z();
        
         sizes[ BottomNorthWest ] = lowerOverlap.x() * lowerOverlap.y() * lowerOverlap.z();
         sizes[ BottomNorthEast ] = upperOverlap.x() * lowerOverlap.y() * lowerOverlap.z();
         sizes[ BottomSouthWest ] = lowerOverlap.x() * upperOverlap.y() * lowerOverlap.z();
         sizes[ BottomSouthEast ] = upperOverlap.x() * upperOverlap.y() * lowerOverlap.z();
         sizes[ TopNorthWest    ] = lowerOverlap.x() * lowerOverlap.y() * upperOverlap.z();
         sizes[ TopNorthEast    ] = upperOverlap.x() * lowerOverlap.y() * upperOverlap.z();
         sizes[ TopSouthWest    ] = lowerOverlap.x() * upperOverlap.y() * upperOverlap.z();
         sizes[ TopSouthEast    ] = upperOverlap.x() * upperOverlap.y() * upperOverlap.z();

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
        
         westSource   = lowerOverlap.x();
         eastSource   = localGridSize.x() - 2 * upperOverlap.x();
         northSource  = lowerOverlap.y();
         southSource  = localGridSize.y() - 2 * upperOverlap.y();
         bottomSource = lowerOverlap.z();
         topSource    = localGridSize.z() - 2 * upperOverlap.z();
            
         xCenter = lowerOverlap.x();
         yCenter = lowerOverlap.y();
         zCenter = lowerOverlap.z();
        
         westDestination   = 0;
         eastDestination   = localGridSize.x() - upperOverlap.x();
         northDestination  = 0;
         southDestination  = localGridSize.y() - upperOverlap.y();
         bottomDestination = 0;
         topDestination    = localGridSize.z() - upperOverlap.z();         
        
         const int *neighbors = distributedGrid->getNeighbors();
         const int *periodicNeighbors = distributedGrid->getPeriodicNeighbors();
         
        if( periodicBoundaries )
         {
            if( neighbors[ West ] == -1 )
               swap( westSource, westDestination );
            if( neighbors[ East ] == -1 )
               swap( eastSource, eastDestination );
            if( neighbors[ South ] == -1 )
               swap( southSource, southDestination );
            if( neighbors[ North ] == -1 )
               swap( northSource, northDestination );
            if( neighbors[ Bottom ] == -1 )
               swap( bottomSource, bottomDestination );
            if( neighbors[ Top ] == -1 )
               swap( topSource, topDestination );            
         }         
        
         //fill send buffers
         copyBuffers( meshFunction, sendBuffers, true,
            westSource, eastSource, northSource, southSource, bottomSource, topSource,
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
		                
         //send everything, recieve everything 
         for( int i=0; i<26; i++ )
            if( neighbors[ i ] != -1 )
            {
               requests[ requestsCount++ ] = CommunicatorType::ISend( sendBuffers[ i ].getData(),  sizes[ i ], neighbors[ i ], 0, group );
               requests[ requestsCount++ ] = CommunicatorType::IRecv( receiveBuffers[ i ].getData(),  sizes[ i ], neighbors[ i ], 0, group );
            }
            else if( periodicBoundaries )
      	   {
               requests[ requestsCount++ ] = CommunicatorType::ISend( sendBuffers[ i ].getData(),  sizes[ i ], periodicNeighbors[ i ], 0, group );
               requests[ requestsCount++ ] = CommunicatorType::IRecv( receiveBuffers[ i ].getData(),  sizes[ i ], periodicNeighbors[ i ], 0, group );
            }

        //wait until send is done
        CommunicatorType::WaitAll( requests, requestsCount );

        //copy data from receive buffers
        copyBuffers(meshFunction, receiveBuffers, false,
            westDestination, eastDestination, northDestination, southDestination, bottomDestination, topDestination,
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
         int west, int east, int north, int south, int bottom, int top,
         int xcenter, int ycenter, int zcenter,
         const CoordinatesType& lowerOverlap,
         const CoordinatesType& upperOverlap,
         const CoordinatesType& localSize,
         const int* neighbor,
         bool periodicBoundaries,
         const PeriodicBoundariesMaskPointer& mask )
      {
         bool westIsBoundary = ( neighbor[ West ] == -1 );
         bool eastIsBoundary = ( neighbor[ East ] == -1 );
         bool northIsBoundary = ( neighbor[ North ] == -1 );
         bool southIsBoundary = ( neighbor[ South ] == -1 );
         bool bottomIsBoundary = ( neighbor[ Bottom ] == -1 );
         bool topIsBoundary = ( neighbor[ Top ] == -1 );

         bool northWestIsBoundary = ( neighbor[ NorthWest ] == -1 );
         bool northEastIsBoundary = ( neighbor[ NorthEast ] == -1 );
         bool southWestIsBoundary = ( neighbor[ SouthWest ] == -1 );
         bool southEastIsBoundary = ( neighbor[ SouthEast ] == -1 );
         
         bool bottomWestIsBoundary = ( neighbor[ BottomWest ] == -1 );
         bool bottomEastIsBoundary = ( neighbor[ BottomEast ] == -1 );
         bool bottomNorthIsBoundary = ( neighbor[ BottomNorth ] == -1 );
         bool bottomSouthIsBoundary = ( neighbor[ BottomSouth ] == -1 );

         bool topWestIsBoundary = ( neighbor[ TopWest ] == -1 );
         bool topEastIsBoundary = ( neighbor[ TopEast ] == -1 );
         bool topNorthIsBoundary = ( neighbor[ TopNorth ] == -1 );
         bool topSouthIsBoundary = ( neighbor[ TopSouth ] == -1 );

         bool bottomNorthWestIsBoundary = ( neighbor[ BottomNorthWest ] == -1 );
         bool bottomNorthEastIsBoundary = ( neighbor[ BottomNorthEast ] == -1 );
         bool bottomSouthWestIsBoundary = ( neighbor[ BottomSouthWest ] == -1 );
         bool bottomSouthEastIsBoundary = ( neighbor[ BottomSouthEast ] == -1 );

         bool topNorthWestIsBoundary = ( neighbor[ TopNorthWest ] == -1 );
         bool topNorthEastIsBoundary = ( neighbor[ TopNorthEast ] == -1 );
         bool topSouthWestIsBoundary = ( neighbor[ TopSouthWest ] == -1 );
         bool topSouthEastIsBoundary = ( neighbor[ TopSouthEast ] == -1 );
         
         using Helper = BufferEntitiesHelper< MeshFunctionType, PeriodicBoundariesMaskPointer, 3, Real_, Device >;
         //X-Y-Z
         if( ! westIsBoundary || periodicBoundaries )
            Helper::BufferEntities( meshFunction, mask, buffers[ West ].getData(),   westIsBoundary,   west,    ycenter, zcenter, lowerOverlap.x(), localSize.y(),     localSize.z(),    toBuffer );
         if( ! eastIsBoundary || periodicBoundaries )
            Helper::BufferEntities( meshFunction, mask, buffers[ East ].getData(),   eastIsBoundary,   east,    ycenter, zcenter, upperOverlap.x(), localSize.y(),     localSize.z(),    toBuffer );
         if( ! northIsBoundary || periodicBoundaries )
            Helper::BufferEntities( meshFunction, mask, buffers[ North ].getData(),  northIsBoundary,  xcenter, north,   zcenter, localSize.x(),    lowerOverlap.y(),  localSize.z(),    toBuffer );
         if( ! southIsBoundary || periodicBoundaries )
            Helper::BufferEntities( meshFunction, mask, buffers[ South ].getData(),  southIsBoundary,  xcenter, south,   zcenter, localSize.x(),     upperOverlap.y(), localSize.z(),    toBuffer );
         if( ! bottomIsBoundary || periodicBoundaries )
            Helper::BufferEntities( meshFunction, mask, buffers[ Bottom ].getData(), bottomIsBoundary, xcenter, ycenter, bottom,  localSize.x(),     localSize.y(),    lowerOverlap.z(), toBuffer );
         if( ! topIsBoundary || periodicBoundaries )
            Helper::BufferEntities( meshFunction, mask, buffers[ Top ].getData(),    topIsBoundary,    xcenter, ycenter, top,     localSize.x(),     localSize.y(),    upperOverlap.z(), toBuffer );	
         
         //XY
         if( ! northWestIsBoundary || periodicBoundaries )
            Helper::BufferEntities( meshFunction, mask, buffers[ NorthWest ].getData(), northWestIsBoundary, west, north, zcenter, lowerOverlap.x(), lowerOverlap.y(), localSize.z(), toBuffer );
         if( ! northEastIsBoundary || periodicBoundaries )
            Helper::BufferEntities( meshFunction, mask, buffers[ NorthEast ].getData(), northEastIsBoundary, east, north, zcenter, upperOverlap.x(), lowerOverlap.y(), localSize.z(), toBuffer );
         if( ! southWestIsBoundary || periodicBoundaries )
            Helper::BufferEntities( meshFunction, mask, buffers[ SouthWest ].getData(), southWestIsBoundary, west, south, zcenter, lowerOverlap.x(), upperOverlap.y(), localSize.z(), toBuffer );
         if( ! southEastIsBoundary || periodicBoundaries )
            Helper::BufferEntities( meshFunction, mask, buffers[ SouthEast ].getData(), southEastIsBoundary, east, south, zcenter, upperOverlap.x(), upperOverlap.y(), localSize.z(), toBuffer );
         
         //XZ
         if( ! bottomWestIsBoundary || periodicBoundaries )
            Helper::BufferEntities( meshFunction, mask, buffers[ BottomWest ].getData(), bottomWestIsBoundary, west, ycenter, bottom, lowerOverlap.x(), localSize.y(), lowerOverlap.z(), toBuffer );
         if( ! bottomEastIsBoundary || periodicBoundaries )
            Helper::BufferEntities( meshFunction, mask, buffers[ BottomEast ].getData(), bottomEastIsBoundary, east, ycenter, bottom, upperOverlap.x(), localSize.y(), lowerOverlap.z(), toBuffer );
         if( ! topWestIsBoundary || periodicBoundaries )
            Helper::BufferEntities( meshFunction, mask, buffers[ TopWest ].getData(),    topWestIsBoundary,    west, ycenter, top,    lowerOverlap.x(), localSize.y(), upperOverlap.z(), toBuffer );
         if( ! topEastIsBoundary || periodicBoundaries )
            Helper::BufferEntities( meshFunction, mask, buffers[ TopEast ].getData(),    topEastIsBoundary,    east, ycenter, top,    upperOverlap.x(), localSize.y(), upperOverlap.z(), toBuffer );   
         
         //YZ
         if( ! bottomNorthIsBoundary || periodicBoundaries )
            Helper::BufferEntities( meshFunction, mask, buffers[ BottomNorth ].getData(), bottomNorthIsBoundary, xcenter, north, bottom, localSize.x(), lowerOverlap.y(), lowerOverlap.z(), toBuffer );
         if( ! bottomSouthIsBoundary || periodicBoundaries )
            Helper::BufferEntities( meshFunction, mask, buffers[ BottomSouth ].getData(), bottomSouthIsBoundary, xcenter, south, bottom, localSize.x(), upperOverlap.y(), lowerOverlap.z(), toBuffer );
         if( ! topNorthIsBoundary || periodicBoundaries )
            Helper::BufferEntities( meshFunction, mask, buffers[ TopNorth ].getData(),    topNorthIsBoundary,    xcenter, north, top,    localSize.x(), lowerOverlap.y(), upperOverlap.z(), toBuffer );
         if( ! topSouthIsBoundary || periodicBoundaries )
            Helper::BufferEntities( meshFunction, mask, buffers[ TopSouth ].getData(),    topSouthIsBoundary,    xcenter, south, top,    localSize.x(), upperOverlap.y(), upperOverlap.z(), toBuffer );
         
         //XYZ
         if( ! bottomNorthWestIsBoundary || periodicBoundaries )
            Helper::BufferEntities( meshFunction, mask, buffers[ BottomNorthWest ].getData(), bottomNorthWestIsBoundary, west, north, bottom, lowerOverlap.x(), lowerOverlap.y(), lowerOverlap.z(), toBuffer );
         if( ! bottomNorthEastIsBoundary || periodicBoundaries )
            Helper::BufferEntities( meshFunction, mask, buffers[ BottomNorthEast ].getData(), bottomNorthEastIsBoundary, east, north, bottom, upperOverlap.x(), lowerOverlap.y(), lowerOverlap.z(), toBuffer );
         if( ! bottomSouthWestIsBoundary || periodicBoundaries )
            Helper::BufferEntities( meshFunction, mask, buffers[ BottomSouthWest ].getData(), bottomSouthWestIsBoundary, west, south, bottom, lowerOverlap.x(), upperOverlap.y(), lowerOverlap.z(), toBuffer );
         if( ! bottomSouthEastIsBoundary || periodicBoundaries )
            Helper::BufferEntities( meshFunction, mask, buffers[ BottomSouthEast ].getData(), bottomSouthEastIsBoundary, east, south, bottom, upperOverlap.x(), upperOverlap.y(), lowerOverlap.z(), toBuffer );
         if( ! topNorthWestIsBoundary || periodicBoundaries )
            Helper::BufferEntities( meshFunction, mask, buffers[ TopNorthWest ].getData(),    topNorthWestIsBoundary,    west, north, top,    lowerOverlap.x(), lowerOverlap.y(), upperOverlap.z(), toBuffer );
         if( ! topNorthEastIsBoundary || periodicBoundaries )
            Helper::BufferEntities( meshFunction, mask, buffers[ TopNorthEast ].getData(),    topNorthEastIsBoundary,    east, north, top,    upperOverlap.x(), lowerOverlap.y(), upperOverlap.z(), toBuffer );
         if( ! topSouthWestIsBoundary || periodicBoundaries )
            Helper::BufferEntities( meshFunction, mask, buffers[ TopSouthWest ].getData(),    topSouthEastIsBoundary,    west, south, top,    lowerOverlap.x(), upperOverlap.y(), upperOverlap.z(), toBuffer );
         if( ! topSouthEastIsBoundary || periodicBoundaries )
            Helper::BufferEntities( meshFunction, mask, buffers[ TopSouthEast ].getData(),    topSouthEastIsBoundary,    east, south, top,    upperOverlap.x(), upperOverlap.y(), upperOverlap.z(), toBuffer );   
      }
    
   private:
   
      Containers::Array<RealType, Device, Index> sendBuffers[26];
      Containers::Array<RealType, Device, Index> receiveBuffers[26];
      Containers::StaticArray< 26, int > sizes;
      
      DistributedGridType *distributedGrid;
        
      int westSource;
      int eastSource;
      int northSource;
      int southSource;
      int bottomSource;
      int topSource;
      int xCenter;
      int yCenter;
      int zCenter;
      int westDestination;
      int eastDestination;
      int northDestination;
      int southDestination;
      int bottomDestination;
      int topDestination;
        
      CoordinatesType overlap;
      CoordinatesType localSize;

      bool isSet;
    
};


} // namespace DistributedMeshes
} // namespace Meshes
} // namespace TNL

