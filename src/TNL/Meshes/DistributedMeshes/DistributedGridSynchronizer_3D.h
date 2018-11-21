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
#include <TNL/Meshes/DistributedMeshes/Directions.h>

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
         
         const SubdomainOverlapsType& lowerOverlap = this->distributedGrid->getLowerOverlap();
         const SubdomainOverlapsType& upperOverlap = this->distributedGrid->getUpperOverlap();
         const CoordinatesType& localSize = this->distributedGrid->getLocalSize(); 
         const CoordinatesType& localGridSize = this->distributedGrid->getLocalGridSize();         

         for( int i=0; i<this->getNeighborCount(); i++ )
         {
            Index sendSize=1;
            Index rcvSize=1;
            auto directions=Directions::template getXYZ<this->getMeshDimension()>(i);
            for(int j=0;j<this->getMeshDimension();j++)
            {
               if(directions[j]==-1)
               {
                  sendSize*=upperOverlap[j];
                  rcvSize*=lowerOverlap[j];
               }
               if(directions[j]==0)
               {
                  sendSize*=localSize[j];
                  rcvSize*=localSize[j];
               }
               if(directions[j]==1)
               {
                  sendSize*=lowerOverlap[j];
                  rcvSize*=upperOverlap[j];
               }
            }

            sendSizes[ i ] = sendSize;
            recieveSizes[ i ] = rcvSize;
            sendBuffers[ i ].setSize( sendSize );
            recieveBuffers[ i ].setSize( rcvSize);

             int world_rank;
             MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
             std::cout<< world_rank<<": " << " "<<lowerOverlap << upperOverlap << std::endl;
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
         const CoordinatesType& localSize = this->distributedGrid->getLocalSize(); 
         const CoordinatesType& localBegin = this->distributedGrid->getLocalBegin();
        
         const int *neighbors = distributedGrid->getNeighbors();
         const int *periodicNeighbors = distributedGrid->getPeriodicNeighbors();
         
        if( periodicBoundaries )
         {
            std::cerr<<"TOTALY DEMAGED by refactorization" << std::endl;
         }         
        
         //fill send buffers
         copyBuffers( meshFunction, sendBuffers, true,
            localBegin,localSize,
            lowerOverlap, upperOverlap,
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
               requests[ requestsCount++ ] = CommunicatorType::ISend( sendBuffers[ i ].getData(),  sendSizes[ i ], neighbors[ i ],0, group );
               requests[ requestsCount++ ] = CommunicatorType::IRecv( recieveBuffers[ i ].getData(),  recieveSizes[ i ], neighbors[ i ], 0, group );
            }
            else if( periodicBoundaries )
      	   {
               requests[ requestsCount++ ] = CommunicatorType::ISend( sendBuffers[ i ].getData(),  sendSizes[ i ], periodicNeighbors[ i ], 1, group );
               requests[ requestsCount++ ] = CommunicatorType::IRecv( recieveBuffers[ i ].getData(),  recieveSizes[ i ], periodicNeighbors[ i ],1, group );
            }

        //wait until send is done
        CommunicatorType::WaitAll( requests, requestsCount );

        int world_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        for(int i=0;i<this->getNeighborCount();i++)
          std::cout<< world_rank<<": " << i << " send:"<<sendBuffers[ i ]<<" recv:"<<recieveBuffers[ i ] << std::endl;

        //copy data from receive buffers
        copyBuffers(meshFunction, recieveBuffers, false,
            localBegin, localSize,
            lowerOverlap, upperOverlap,
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
         const CoordinatesType& localBegin,
         const CoordinatesType& localSize,
         const CoordinatesType& lowerOverlap,
         const CoordinatesType& upperOverlap,
         const int* neighbor,
         bool periodicBoundaries,
         const PeriodicBoundariesMaskPointer& mask )
      {

         using Helper = BufferEntitiesHelper< MeshFunctionType, PeriodicBoundariesMaskPointer, this->getMeshDimension(), Real_, Device >;
       
         for(int i=0;i<this->getNeighborCount();i++)//performace isssue - this should be buffered when Synchronizer is created
         {
            bool isBoundary=( neighbor[ i ] == -1 );
            
            CoordinatesType begin=localBegin;
            CoordinatesType size=localSize;
            auto directions=Directions::template getXYZ<this->getMeshDimension()>(i);
            for(int j=0;j<this->getMeshDimension();j++)
            {
               if(toBuffer)
               {
                  if(directions[j]==-1)
                  {
                     size[j]=upperOverlap[j];
                  }
                  if(directions[j]==1)
                  {
                     begin[j]=localBegin[j]+localSize[j]-lowerOverlap[j];
                     size[j]=lowerOverlap[j];
                  }
               }
               else
               {  
                  if(directions[j]==-1)
                  {
                     //tady se asi bude řešit periodic boundary
                     begin[j]=0;
                     size[j]=lowerOverlap[j];
                  }
                  if(directions[j]==1)
                  {
                     begin[j]=localBegin[j]+localSize[j];
                     size[j]=upperOverlap[j];
                  }
               }
            }

            if( ! isBoundary || periodicBoundaries )
                  Helper::BufferEntities( meshFunction, mask, buffers[ i ].getData(), isBoundary, begin, size, toBuffer );
 
         }
      }
    
   private:
   
      Containers::Array<RealType, Device, Index> sendBuffers[DirectionCount<MeshDimension>::get()];
      Containers::Array<RealType, Device, Index> recieveBuffers[DirectionCount<MeshDimension>::get()];
      Containers::StaticArray< DirectionCount<MeshDimension>::get(), int > sendSizes;
      Containers::StaticArray< DirectionCount<MeshDimension>::get(), int > recieveSizes;
      
      DistributedGridType *distributedGrid;

      bool isSet;
    
};


} // namespace DistributedMeshes
} // namespace Meshes
} // namespace TNL

