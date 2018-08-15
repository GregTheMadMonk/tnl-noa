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
      typedef typename Functions::MeshFunction< Grid< 3, GridReal, Device, Index >,EntityDimension, RealType> MeshFunctionType;
      typedef typename Grid< 3, GridReal, Device, Index >::DistributedMeshType DistributedGridType; 
      typedef typename MeshFunctionType::RealType Real;
      typedef typename DistributedGridType::CoordinatesType CoordinatesType;
      template< typename Real_ >
      using BufferEntitiesHelperType = BufferEntitiesHelper< MeshFunctionType, 3, Real_, Device >;
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
            sendbuffs[ i ].setSize( sizes[ i ] );
            rcvbuffs[ i ].setSize( sizes[ i ] );
         }
        
         westSrc   = lowerOverlap.x();
         eastSrc   = localGridSize.x() - 2 * upperOverlap.x();
         nordSrc   = lowerOverlap.y();
         southSrc  = localGridSize.y() - 2 * upperOverlap.y();
         bottomSrc = lowerOverlap.z();
         topSrc    = localGridSize.z() - 2 * upperOverlap.z();
            
         xcenter = lowerOverlap.x();
         ycenter = lowerOverlap.y();
         zcenter = lowerOverlap.z();
        
         westDst   = 0;
         eastDst   = localGridSize.x() - upperOverlap.x();
         nordDst   = 0;
         southDst  = localGridSize.y() - upperOverlap.y();
         bottomDst = 0;
         topDst    = localGridSize.z() - upperOverlap.z();
     }
        
      template<typename CommunicatorType>
      void synchronize( MeshFunctionType &meshFunction,
                        bool periodicBoundaries = false )
      {

         TNL_ASSERT_TRUE( isSet, "Synchronizer is not set, but used to synchronize" );

    	   if( !distributedGrid->isDistributed() ) return;
         
         const SubdomainOverlapsType& lowerOverlap = distributedGrid->getLowerOverlap();
         const SubdomainOverlapsType& upperOverlap = distributedGrid->getUpperOverlap();
         const CoordinatesType& localSize = distributedGrid->getLocalSize(); 
         const CoordinatesType& localGridSize = this->distributedGrid->getLocalGridSize();
        
        
         const int *neighbors = distributedGrid->getNeighbors();
         const int *periodicNeighbors = distributedGrid->getPeriodicNeighbors();
        
        //fill send buffers
        copyBuffers( meshFunction, sendbuffs, true,
            westSrc, eastSrc, nordSrc, southSrc, bottomSrc, topSrc,
            xcenter, ycenter, zcenter,
            lowerOverlap, upperOverlap, localSize,
            neighbors);
        
        //async send and rcv
        typename CommunicatorType::Request requests[52];
        typename CommunicatorType::CommunicationGroup group;
        group=*((typename CommunicatorType::CommunicationGroup *)(distributedGrid->getCommunicationGroup()));
        int requestsCount( 0 );
		                
        //send everithing, recieve everything 
        for( int i=0; i<26; i++ )
            if( neighbors[ i ] != -1 )
            {
               requests[ requestsCount++ ] = CommunicatorType::ISend( sendbuffs[ i ].getData(),  sizes[ i ], neighbors[ i ], group );
               requests[ requestsCount++ ] = CommunicatorType::IRecv( rcvbuffs[ i ].getData(),  sizes[ i ], neighbors[ i ], group );
            }
            else if( periodicBoundaries )
      	   {
               requests[ requestsCount++ ] = CommunicatorType::ISend( sendbuffs[ i ].getData(),  sizes[ i ], periodicNeighbors[ i ], group );
               requests[ requestsCount++ ] = CommunicatorType::IRecv( rcvbuffs[ i ].getData(),  sizes[ i ], periodicNeighbors[ i ], group );
            }

        //wait until send is done
        CommunicatorType::WaitAll( requests, requestsCount );

        //copy data form rcv buffers
        copyBuffers(meshFunction, rcvbuffs, false,
            westDst, eastDst, nordDst, southDst, bottomDst, topDst,
            xcenter, ycenter, zcenter,
            lowerOverlap, upperOverlap, localSize,
            neighbors ); 
 
    }
    
   private:
      
      template< typename Real_ >
      void copyBuffers( MeshFunctionType meshFunction, Containers::Array<Real_, Device, Index>* buffers, bool toBuffer,
              int west, int east, int north, int south, int bottom, int top,
              int xcenter, int ycenter, int zcenter,
              const CoordinatesType& lowerOverlap,
              const CoordinatesType& upperOverlap,
              const CoordinatesType& localSize,
              const int *neighbor)
      {
         using Helper = BufferEntitiesHelper< MeshFunctionType, 3, Real_, Device >;
         //X-Y-Z
         if( neighbor[ West ] != -1 )
            Helper::BufferEntities( meshFunction, buffers[ West ].getData(),   west,    ycenter, zcenter, lowerOverlap.x(), localSize.y(),     localSize.z(),    toBuffer );
         if( neighbor[ East ] != -1 )
            Helper::BufferEntities( meshFunction, buffers[ East ].getData(),   east,    ycenter, zcenter, upperOverlap.x(), localSize.y(),     localSize.z(),    toBuffer );
         if( neighbor[ North ] != -1 )
            Helper::BufferEntities( meshFunction, buffers[ North ].getData(),  xcenter, north,   zcenter, localSize.x(),    lowerOverlap.y(),  localSize.z(),    toBuffer );
         if( neighbor[ South ] != -1 )
            Helper::BufferEntities( meshFunction, buffers[ South ].getData(),  xcenter, south,   zcenter, localSize.x(),     upperOverlap.y(), localSize.z(),    toBuffer );
         if( neighbor[ Bottom ] != -1 )
            Helper::BufferEntities( meshFunction, buffers[ Bottom ].getData(), xcenter, ycenter, bottom,  localSize.x(),     localSize.y(),    lowerOverlap.z(), toBuffer );
         if( neighbor[ Top ] != -1 )
            Helper::BufferEntities( meshFunction, buffers[ Top ].getData(),    xcenter, ycenter, top,     localSize.x(),     localSize.y(),    upperOverlap.z(), toBuffer );	
         
         //XY
         if( neighbor[ NorthWest ] != -1 )
            Helper::BufferEntities( meshFunction, buffers[ NorthWest ].getData(), west, north, zcenter, lowerOverlap.x(), lowerOverlap.y(), localSize.z(), toBuffer );
         if( neighbor[ NorthEast ] != -1 )
            Helper::BufferEntities( meshFunction, buffers[ NorthEast ].getData(), east, north, zcenter, upperOverlap.x(), lowerOverlap.y(), localSize.z(), toBuffer );
         if( neighbor[ SouthWest ] != -1 )
            Helper::BufferEntities( meshFunction, buffers[ SouthWest ].getData(), west, south, zcenter, lowerOverlap.x(), upperOverlap.y(), localSize.z(), toBuffer );
         if( neighbor[ SouthEast ] != -1 )
            Helper::BufferEntities( meshFunction, buffers[ SouthEast ].getData(), east, south, zcenter, upperOverlap.x(), upperOverlap.y(), localSize.z(), toBuffer );
         
         //XZ
         if( neighbor[ BottomWest ] != -1 )
            Helper::BufferEntities( meshFunction, buffers[ BottomWest ].getData(), west, ycenter, bottom, lowerOverlap.x(), localSize.y(), lowerOverlap.z(), toBuffer );
         if( neighbor[ BottomEast ] != -1 )
            Helper::BufferEntities( meshFunction, buffers[ BottomEast ].getData(), east, ycenter, bottom, upperOverlap.x(), localSize.y(), lowerOverlap.z(), toBuffer );
         if( neighbor[ TopWest ] != -1 )
            Helper::BufferEntities( meshFunction, buffers[ TopWest ].getData(),    west, ycenter, top,    lowerOverlap.x(), localSize.y(), upperOverlap.z(), toBuffer );
         if( neighbor[ TopEast ] != -1 )
            Helper::BufferEntities( meshFunction, buffers[ TopEast ].getData(),    east, ycenter, top,    upperOverlap.x(), localSize.y(), upperOverlap.z(), toBuffer );   
         
         //YZ
         if( neighbor[ BottomNorth ] != -1 )
            Helper::BufferEntities( meshFunction, buffers[ BottomNorth ].getData(), xcenter, north, bottom, localSize.x(), lowerOverlap.y(), lowerOverlap.z(), toBuffer );
         if( neighbor[ BottomSouth ] != -1 )
            Helper::BufferEntities( meshFunction, buffers[ BottomSouth ].getData(), xcenter, south, bottom, localSize.x(), upperOverlap.y(), lowerOverlap.z(), toBuffer );
         if( neighbor[ TopNorth ] != -1 )
            Helper::BufferEntities( meshFunction, buffers[ TopNorth ].getData(),    xcenter, north, top,    localSize.x(), lowerOverlap.y(), upperOverlap.z(), toBuffer );
         if( neighbor[ TopSouth ] != -1 )
            Helper::BufferEntities( meshFunction, buffers[ TopSouth ].getData(),    xcenter, south, top,    localSize.x(), upperOverlap.y(), upperOverlap.z(), toBuffer );
         
         //XYZ
         if( neighbor[ BottomNorthWest ] != -1 )
            Helper::BufferEntities( meshFunction, buffers[ BottomNorthWest ].getData(), west, north, bottom, lowerOverlap.x(), lowerOverlap.y(), lowerOverlap.z(), toBuffer );
         if( neighbor[ BottomNorthEast ] != -1 )
            Helper::BufferEntities( meshFunction, buffers[ BottomNorthEast ].getData(), east, north, bottom, upperOverlap.x(), lowerOverlap.y(), lowerOverlap.z(), toBuffer );
         if( neighbor[ BottomSouthWest ] != -1 )
            Helper::BufferEntities( meshFunction, buffers[ BottomSouthWest ].getData(), west, south, bottom, lowerOverlap.x(), upperOverlap.y(), lowerOverlap.z(), toBuffer );
         if( neighbor[ BottomSouthEast ] != -1 )
            Helper::BufferEntities( meshFunction, buffers[ BottomSouthEast ].getData(), east, south, bottom, upperOverlap.x(), upperOverlap.y(), lowerOverlap.z(), toBuffer );
         if( neighbor[ TopNorthWest ] != -1 )
            Helper::BufferEntities( meshFunction, buffers[ TopNorthWest ].getData(),    west, north, top,    lowerOverlap.x(), lowerOverlap.y(), upperOverlap.z(), toBuffer );
         if( neighbor[ TopNorthEast ] != -1 )
            Helper::BufferEntities( meshFunction, buffers[ TopNorthEast ].getData(),    east, north, top,    upperOverlap.x(), lowerOverlap.y(), upperOverlap.z(), toBuffer );
         if( neighbor[ TopSouthWest ] != -1 )
            Helper::BufferEntities( meshFunction, buffers[ TopSouthWest ].getData(),    west, south, top,    lowerOverlap.x(), upperOverlap.y(), upperOverlap.z(), toBuffer );
         if( neighbor[ TopSouthEast ] != -1 )
            Helper::BufferEntities( meshFunction, buffers[ TopSouthEast ].getData(),    east, south, top,    upperOverlap.x(), upperOverlap.y(), upperOverlap.z(), toBuffer );   
      }
    
   private:
   
      Containers::Array<RealType, Device, Index> sendbuffs[26];
      Containers::Array<RealType, Device, Index> rcvbuffs[26];
      Containers::StaticArray< 26, int > sizes;
      
      DistributedGridType *distributedGrid;
        
      int westSrc;
      int eastSrc;
      int nordSrc;
      int southSrc;
      int bottomSrc;
      int topSrc;
      int xcenter;
      int ycenter;
      int zcenter;
      int westDst;
      int eastDst;
      int nordDst;
      int southDst;
      int bottomDst;
      int topDst;
        
      CoordinatesType overlap;
      CoordinatesType localSize;

      bool isSet;
    
};


} // namespace DistributedMeshes
} // namespace Meshes
} // namespace TNL

