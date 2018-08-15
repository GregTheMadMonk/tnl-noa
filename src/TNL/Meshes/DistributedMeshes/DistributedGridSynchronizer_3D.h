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
      
    private:
        Containers::Array<RealType, Device, Index> sendbuffs[26];
        Containers::Array<RealType, Device, Index> rcvbuffs[26];
        int sizes[26];
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
    
    public:
    
    DistributedMeshSynchronizer()
    {
        isSet=false;
    };

    DistributedMeshSynchronizer(DistributedGridType *distributedGrid)
    {
        isSet=false;
        setDistributedGrid(distributedGrid);
    };

    void setDistributedGrid(DistributedGridType *distributedGrid)
    {
        isSet=true;

        this->distributedGrid=distributedGrid;        
        overlap = this->distributedGrid->getOverlap();
        localSize = this->distributedGrid->getLocalSize();
        
        CoordinatesType localBegin=this->distributedGrid->getLocalBegin();
        CoordinatesType localGridSize = this->distributedGrid->getLocalGridSize();

        sizes[West]=sizes[East]=localSize.y()*localSize.z()*overlap.x();
        sizes[North]=sizes[South]=localSize.x()*localSize.z()*overlap.y();
        sizes[Bottom]=sizes[Top]=localSize.x()*localSize.y()*overlap.z();
        
        sizes[NorthWest]=sizes[NorthEast]=sizes[SouthWest]=sizes[SouthEast]=localSize.z()*overlap.x()*overlap.y();
        sizes[BottomWest]=sizes[BottomEast]=sizes[TopWest]=sizes[TopEast]=localSize.y()*overlap.x()*overlap.z();
        sizes[BottomNorth]=sizes[BottomSouth]=sizes[TopNorth]=sizes[TopSouth]=localSize.x()*overlap.y()*overlap.z();
        
        sizes[BottomNorthWest]=sizes[BottomNorthEast]=sizes[BottomSouthWest]=sizes[BottomSouthEast]=
                sizes[TopNorthWest]=sizes[TopNorthEast]=sizes[TopSouthWest]=sizes[TopSouthEast]= 
                overlap.x()*overlap.y()*overlap.z();

        for(int i=0;i<26;i++)
        {
                sendbuffs[i].setSize(sizes[i]);
                rcvbuffs[i].setSize(sizes[i]);
        }
        
        westSrc=localBegin.x();
        eastSrc=localGridSize.x()-2*overlap.x();
        nordSrc=localBegin.y();
        southSrc=localGridSize.y()-2*overlap.y();
        bottomSrc=localBegin.z();
        topSrc=localGridSize.z()-2*overlap.z();
            
        xcenter=localBegin.x();
        ycenter=localBegin.y();
        zcenter=localBegin.z();
        
        westDst=0;
        eastDst=localGridSize.x()-overlap.x();
        nordDst=0;
        southDst=localGridSize.y()-overlap.y();
        bottomDst=0;
        topDst=localGridSize.z()-overlap.z();
        
    }
        
    template<typename CommunicatorType>
    void synchronize(MeshFunctionType &meshFunction)
    {

        TNL_ASSERT_TRUE(isSet,"Synchronizer is not set, but used to synchronize");

    	if(!distributedGrid->isDistributed())
            return;
        
        const int *neighbor=distributedGrid->getNeighbors();
        
        //fill send buffers
        copyBuffers(meshFunction, sendbuffs, true,
            westSrc, eastSrc, nordSrc, southSrc, bottomSrc, topSrc,
            xcenter, ycenter, zcenter,
            overlap, localSize,
            neighbor);
        
        //async send and rcv
        typename CommunicatorType::Request req[52];
        typename CommunicatorType::CommunicationGroup group;
        group=*((typename CommunicatorType::CommunicationGroup *)(distributedGrid->getCommunicationGroup()));
		                
        //send everithing, recieve everything 
        for(int i=0;i<26;i++)	
           if(neighbor[i]!=-1)
           {
               req[i]=CommunicatorType::ISend(sendbuffs[i].getData(), sizes[i], neighbor[i],group);
               req[26+i]=CommunicatorType::IRecv(rcvbuffs[i].getData(), sizes[i], neighbor[i],group);
           }
		   else
      	   {
               req[i]=CommunicatorType::NullRequest;
               req[26+i]=CommunicatorType::NullRequest;
           }

        //wait until send is done
        CommunicatorType::WaitAll(req,52);

        //copy data form rcv buffers
        copyBuffers(meshFunction, rcvbuffs, false,
            westDst, eastDst, nordDst, southDst, bottomDst, topDst,
            xcenter, ycenter, zcenter,
            overlap, localSize,
            neighbor); 
 
    }
    
    private:    
    template< typename Real_ >
    void copyBuffers(MeshFunctionType meshFunction, Containers::Array<Real_, Device, Index> * buffers, bool toBuffer,
            int west, int east, int nord, int south, int bottom, int top,
            int xcenter, int ycenter, int zcenter,
            CoordinatesType shortDim, CoordinatesType longDim,
            const int *neighbor)
    {
       //X-Y-Z
        if(neighbor[West]!=-1)
            BufferEntitiesHelper< MeshFunctionType, 3, Real_, Device >::BufferEntities(meshFunction,buffers[West].getData(),west,ycenter,zcenter,shortDim.x(),longDim.y(),longDim.z(),toBuffer);
        if(neighbor[East]!=-1)
            BufferEntitiesHelper< MeshFunctionType, 3, Real_, Device >::BufferEntities(meshFunction,buffers[East].getData(),east,ycenter,zcenter,shortDim.x(),longDim.y(),longDim.z(),toBuffer);
        if(neighbor[North]!=-1)
            BufferEntitiesHelper< MeshFunctionType, 3, Real_, Device >::BufferEntities(meshFunction,buffers[North].getData(),xcenter,nord,zcenter,longDim.x(),shortDim.y(),longDim.z(),toBuffer);
        if(neighbor[South]!=-1)
            BufferEntitiesHelper< MeshFunctionType, 3, Real_, Device >::BufferEntities(meshFunction,buffers[South].getData(),xcenter,south,zcenter,longDim.x(),shortDim.y(),longDim.z(),toBuffer);
        if(neighbor[Bottom]!=-1)
            BufferEntitiesHelper< MeshFunctionType, 3, Real_, Device >::BufferEntities(meshFunction,buffers[Bottom].getData(),xcenter,ycenter,bottom,longDim.x(),longDim.y(),shortDim.z(),toBuffer);
        if(neighbor[Top]!=-1)
            BufferEntitiesHelper< MeshFunctionType, 3, Real_, Device >::BufferEntities(meshFunction,buffers[Top].getData(),xcenter,ycenter,top,longDim.x(),longDim.y(),shortDim.z(),toBuffer);	
        //XY
        if(neighbor[NorthWest]!=-1)
            BufferEntitiesHelper< MeshFunctionType, 3, Real_, Device >::BufferEntities(meshFunction,buffers[NorthWest].getData(),west,nord,zcenter,shortDim.x(),shortDim.y(),longDim.z(),toBuffer);
        if(neighbor[NorthEast]!=-1)
            BufferEntitiesHelper< MeshFunctionType, 3, Real_, Device >::BufferEntities(meshFunction,buffers[NorthEast].getData(),east,nord,zcenter,shortDim.x(),shortDim.y(),longDim.z(),toBuffer);
        if(neighbor[SouthWest]!=-1)
            BufferEntitiesHelper< MeshFunctionType, 3, Real_, Device >::BufferEntities(meshFunction,buffers[SouthWest].getData(),west,south,zcenter,shortDim.x(),shortDim.y(),longDim.z(),toBuffer);
        if(neighbor[SouthEast]!=-1)
            BufferEntitiesHelper< MeshFunctionType, 3, Real_, Device >::BufferEntities(meshFunction,buffers[SouthEast].getData(),east,south,zcenter,shortDim.x(),shortDim.y(),longDim.z(),toBuffer);
        //XZ
        if(neighbor[BottomWest]!=-1)
            BufferEntitiesHelper< MeshFunctionType, 3, Real_, Device >::BufferEntities(meshFunction,buffers[BottomWest].getData(),west,ycenter,bottom,shortDim.x(),longDim.y(),shortDim.z(),toBuffer);
        if(neighbor[BottomEast]!=-1)
            BufferEntitiesHelper< MeshFunctionType, 3, Real_, Device >::BufferEntities(meshFunction,buffers[BottomEast].getData(),east,ycenter,bottom,shortDim.x(),longDim.y(),shortDim.z(),toBuffer);
        if(neighbor[TopWest]!=-1)
            BufferEntitiesHelper< MeshFunctionType, 3, Real_, Device >::BufferEntities(meshFunction,buffers[TopWest].getData(),west,ycenter,top,shortDim.x(),longDim.y(),shortDim.z(),toBuffer);
        if(neighbor[TopEast]!=-1)
            BufferEntitiesHelper< MeshFunctionType, 3, Real_, Device >::BufferEntities(meshFunction,buffers[TopEast].getData(),east,ycenter,top,shortDim.x(),longDim.y(),shortDim.z(),toBuffer);   
        //YZ
        if(neighbor[BottomNorth]!=-1)
            BufferEntitiesHelper< MeshFunctionType, 3, Real_, Device >::BufferEntities(meshFunction,buffers[BottomNorth].getData(),xcenter,nord,bottom,longDim.x(),shortDim.y(),shortDim.z(),toBuffer);
        if(neighbor[BottomSouth]!=-1)
            BufferEntitiesHelper< MeshFunctionType, 3, Real_, Device >::BufferEntities(meshFunction,buffers[BottomSouth].getData(),xcenter,south,bottom,longDim.x(),shortDim.y(),shortDim.z(),toBuffer);
        if(neighbor[TopNorth]!=-1)
            BufferEntitiesHelper< MeshFunctionType, 3, Real_, Device >::BufferEntities(meshFunction,buffers[TopNorth].getData(),xcenter,nord,top,longDim.x(),shortDim.y(),shortDim.z(),toBuffer);
        if(neighbor[TopSouth]!=-1)
            BufferEntitiesHelper< MeshFunctionType, 3, Real_, Device >::BufferEntities(meshFunction,buffers[TopSouth].getData(),xcenter,south,top,longDim.x(),shortDim.y(),shortDim.z(),toBuffer);
        //XYZ
        if(neighbor[BottomNorthWest]!=-1)
            BufferEntitiesHelper< MeshFunctionType, 3, Real_, Device >::BufferEntities(meshFunction,buffers[BottomNorthWest].getData(),west,nord,bottom,shortDim.x(),shortDim.y(),shortDim.z(),toBuffer);
        if(neighbor[BottomNorthEast]!=-1)
            BufferEntitiesHelper< MeshFunctionType, 3, Real_, Device >::BufferEntities(meshFunction,buffers[BottomNorthEast].getData(),east,nord,bottom,shortDim.x(),shortDim.y(),shortDim.z(),toBuffer);
        if(neighbor[BottomSouthWest]!=-1)
            BufferEntitiesHelper< MeshFunctionType, 3, Real_, Device >::BufferEntities(meshFunction,buffers[BottomSouthWest].getData(),west,south,bottom,shortDim.x(),shortDim.y(),shortDim.z(),toBuffer);
        if(neighbor[BottomSouthEast]!=-1)
            BufferEntitiesHelper< MeshFunctionType, 3, Real_, Device >::BufferEntities(meshFunction,buffers[BottomSouthEast].getData(),east,south,bottom,shortDim.x(),shortDim.y(),shortDim.z(),toBuffer);
        if(neighbor[TopNorthWest]!=-1)
            BufferEntitiesHelper< MeshFunctionType, 3, Real_, Device >::BufferEntities(meshFunction,buffers[TopNorthWest].getData(),west,nord,top,shortDim.x(),shortDim.y(),shortDim.z(),toBuffer);
        if(neighbor[TopNorthEast]!=-1)
            BufferEntitiesHelper< MeshFunctionType, 3, Real_, Device >::BufferEntities(meshFunction,buffers[TopNorthEast].getData(),east,nord,top,shortDim.x(),shortDim.y(),shortDim.z(),toBuffer);
        if(neighbor[TopSouthWest]!=-1)
            BufferEntitiesHelper< MeshFunctionType, 3, Real_, Device >::BufferEntities(meshFunction,buffers[TopSouthWest].getData(),west,south,top,shortDim.x(),shortDim.y(),shortDim.z(),toBuffer);
        if(neighbor[TopSouthEast]!=-1)
            BufferEntitiesHelper< MeshFunctionType, 3, Real_, Device >::BufferEntities(meshFunction,buffers[TopSouthEast].getData(),east,south,top,shortDim.x(),shortDim.y(),shortDim.z(),toBuffer);   
    }
};


} // namespace DistributedMeshes
} // namespace Meshes
} // namespace TNL

