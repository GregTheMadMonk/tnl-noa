/***************************************************************************
                          DistributedGrid.h  -  description
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


//=============================================1D==================================

template <typename RealType,
          int EntityDimension,
          typename Index,
          typename Device,
          typename GridReal>  
class DistributedMeshSynchronizer< Functions::MeshFunction< Grid< 1, GridReal, Device, Index >,EntityDimension, RealType>>
{

public:
        typedef typename Grid< 1, GridReal, Device, Index >::Cell Cell;
        typedef typename Functions::MeshFunction< Grid< 1, GridReal, Device, Index >,EntityDimension, RealType> MeshFunctionType;
        typedef typename Grid< 1, GridReal, Device, Index >::DistributedMeshType DistributedGridType; 
        typedef RealType Real;


private:  
        Containers::Array<RealType, Device> sendbuffs[2];
        Containers::Array<RealType, Device> rcvbuffs[2];
        int overlapSize;

        DistributedGridType *distributedgrid;

        bool isSet;

    
    public:
    DistributedMeshSynchronizer()
    {
        isSet=false;
    };

    DistributedMeshSynchronizer(DistributedGridType *distrgrid)
    {
        isSet=false;
        SetDistributedGrid(distrgrid);
    };

    void SetDistributedGrid(DistributedGridType *distrgrid)
    {
        isSet=true;

        this->distributedgrid=distrgrid;

        overlapSize = distributedgrid->getOverlap().x();

        sendbuffs[0].setSize(overlapSize);
        sendbuffs[1].setSize(overlapSize);
        rcvbuffs[0].setSize(overlapSize);
        rcvbuffs[1].setSize(overlapSize);      
    };

    template<typename CommunicatorType>
    void Synchronize(MeshFunctionType &meshfunction)
    {
        TNL_ASSERT_TRUE(isSet,"Synchronizer is not set, but used to Synchronize");

        if(!distributedgrid->IsDistributed())
                return;

        int leftN=distributedgrid->getLeft();
        int rightN=distributedgrid->getRight();

        int totalSize = meshfunction.getMesh().getDimensions().x();

        CopyBuffers(meshfunction, sendbuffs, true,
                overlapSize, totalSize-2*overlapSize, overlapSize,
                leftN, rightN);

        //async send
        typename CommunicatorType::Request req[4];

        //send everithing, recieve everything 
        if(leftN!=-1)
        {
            req[0]=CommunicatorType::ISend(sendbuffs[Left].getData(), overlapSize, leftN);
            req[2]=CommunicatorType::IRecv(rcvbuffs[Left].getData(), overlapSize, leftN);
        }
        else
        {
            req[0]=CommunicatorType::NullRequest;
            req[2]=CommunicatorType::NullRequest;
        }        

        if(rightN!=-1)
        {
            req[1]=CommunicatorType::ISend(sendbuffs[Right].getData(), overlapSize, rightN);
            req[3]=CommunicatorType::IRecv(rcvbuffs[Right].getData(), overlapSize, rightN);
        }
        else
        {
            req[1]=CommunicatorType::NullRequest;
            req[3]=CommunicatorType::NullRequest;
        }

        //wait until send and recv is done
        CommunicatorType::WaitAll(req, 4);

        CopyBuffers(meshfunction, rcvbuffs, false,
                0, totalSize-overlapSize, overlapSize,
                leftN, rightN);
    }

    private:
    template <typename Real>
    void CopyBuffers(MeshFunctionType meshfunction, TNL::Containers::Array<Real,Device> * buffers, bool toBuffer,
            int left, int right,
            int size,
            int leftNeighbor, int rightNeighbor)
    {
        if(leftNeighbor!=-1)
            BufferEntitiesHelper<MeshFunctionType,1,Real,Device>::BufferEntities(meshfunction,buffers[Left].getData(),left,size,toBuffer);
        if(rightNeighbor!=-1)
            BufferEntitiesHelper<MeshFunctionType,1,Real,Device>::BufferEntities(meshfunction,buffers[Right].getData(),right,size,toBuffer);  
    }

};

//=========================2D=================================================
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

    private:
        DistributedGridType *distributedgrid;

        Containers::Array<RealType, Device, Index> sendbuffs[8];
        Containers::Array<RealType, Device, Index> rcvbuffs[8];
        int sizes[8];
        
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
        
        CoordinatesType overlap;
        CoordinatesType localsize;

        bool isSet;

    public:
    DistributedMeshSynchronizer()
    {
        isSet=false;
    };

    DistributedMeshSynchronizer(DistributedGridType *distrgrid)
    {
        isSet=false;
        SetDistributedGrid(distrgrid);
    };

    void SetDistributedGrid(DistributedGridType *distrgrid)
    {
        isSet=true;
        
        this->distributedgrid=distrgrid;

        overlap = distributedgrid->getOverlap();
        localsize = distributedgrid->getLocalSize();
        
        CoordinatesType localgridsize = this->distributedgrid->getLocalGridSize();
        CoordinatesType localbegin=this->distributedgrid->getLocalBegin();

        int updownsize=localsize.x()*overlap.y();
        int leftrightsize=localsize.y()*overlap.x();
        int connersize=overlap.x()*overlap.y();

        sizes[Left]=leftrightsize;
        sizes[Right]=leftrightsize;
        sizes[Up]=updownsize;
        sizes[Down]=updownsize;
        sizes[UpLeft]=connersize;
        sizes[DownLeft]=connersize;
        sizes[UpRight]=connersize;
        sizes[DownRight]=connersize;

        for(int i=0;i<8;i++)
        {
            sendbuffs[i].setSize(sizes[i]);
            rcvbuffs[i].setSize(sizes[i]);
        }

        leftSrc=localbegin.x();
        rightSrc=localgridsize.x()-2*overlap.x();
        upSrc=localbegin.y();
        downSrc=localgridsize.y()-2*overlap.y();
            
        xcenter=localbegin.x();
        ycenter=localbegin.y();
        
        leftDst=0;
        rightDst=localgridsize.x()-overlap.x();
        upDst=0;
        downDst=localgridsize.y()-overlap.y();                       
    }
       
    template<typename CommunicatorType>
    void Synchronize( MeshFunctionType &meshfunction)
    {

        TNL_ASSERT_TRUE(isSet,"Synchronizer is not set, but used to Synchronize");

	    if(!distributedgrid->IsDistributed())
            return;

        int *neighbor=distributedgrid->getNeighbors();

        CopyBuffers(meshfunction, sendbuffs, true,
            leftSrc, rightSrc, upSrc, downSrc,
            xcenter, ycenter,
            overlap,localsize,
            neighbor);
	
        //async send and rcv
        typename CommunicatorType::Request req[16];
		                
        //send everithing, recieve everything 
        for(int i=0;i<8;i++)	
           if(neighbor[i]!=-1)
           {
               req[i]=CommunicatorType::ISend(sendbuffs[i].getData(), sizes[i], neighbor[i]);
               req[8+i]=CommunicatorType::IRecv(rcvbuffs[i].getData(), sizes[i], neighbor[i]);
           }
		   else
      	   {
               req[i]=CommunicatorType::NullRequest;
               req[8+i]=CommunicatorType::NullRequest;
           }

        //wait until send is done
        CommunicatorType::WaitAll(req,16);
        
        //copy data form rcv buffers
        CopyBuffers(meshfunction, rcvbuffs, false,
            leftDst, rightDst, upDst, downDst,
            xcenter, ycenter,
            overlap,localsize,
            neighbor);
    }
    
    private:
    template <typename Real>
    void CopyBuffers(MeshFunctionType meshfunction, Containers::Array<Real, Device, Index> * buffers, bool toBuffer,
            int left, int right, int up, int down,
            int xcenter, int ycenter,
            CoordinatesType shortDim, CoordinatesType longDim,
            int *neighbor)
    {
       	if(neighbor[Left]!=-1)        
            BufferEntitiesHelper<MeshFunctionType,2,Real,Device>::BufferEntities(meshfunction,buffers[Left].getData(),left,ycenter,shortDim.x(),longDim.y(),toBuffer);
        if(neighbor[Right]!=-1)
            BufferEntitiesHelper<MeshFunctionType,2,Real,Device>::BufferEntities(meshfunction,buffers[Right].getData(),right,ycenter,shortDim.x(),longDim.y(),toBuffer);
        if(neighbor[Up]!=-1)
            BufferEntitiesHelper<MeshFunctionType,2,Real,Device>::BufferEntities(meshfunction,buffers[Up].getData(),xcenter,up,longDim.x(),shortDim.y(),toBuffer);
        if(neighbor[Down]!=-1)
            BufferEntitiesHelper<MeshFunctionType,2,Real,Device>::BufferEntities(meshfunction,buffers[Down].getData(),xcenter,down,longDim.x(),shortDim.y(),toBuffer);
        if(neighbor[UpLeft]!=-1)
            BufferEntitiesHelper<MeshFunctionType,2,Real,Device>::BufferEntities(meshfunction,buffers[UpLeft].getData(),left,up,shortDim.x(),shortDim.y(),toBuffer);
        if(neighbor[UpRight]!=-1)
            BufferEntitiesHelper<MeshFunctionType,2,Real,Device>::BufferEntities(meshfunction,buffers[UpRight].getData(),right,up,shortDim.x(),shortDim.y(),toBuffer);
        if(neighbor[DownLeft]!=-1)        
            BufferEntitiesHelper<MeshFunctionType,2,Real,Device>::BufferEntities(meshfunction,buffers[DownLeft].getData(),left,down,shortDim.x(),shortDim.y(),toBuffer);
        if(neighbor[DownRight]!=-1)
            BufferEntitiesHelper<MeshFunctionType,2,Real,Device>::BufferEntities(meshfunction,buffers[DownRight].getData(),right,down,shortDim.x(),shortDim.y(),toBuffer);
    }
};


//=========================3D=================================================
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
      
    private:
        Containers::Array<RealType, Device, Index> sendbuffs[26];
        Containers::Array<RealType, Device, Index> rcvbuffs[26];
        int sizes[26];
        DistributedGridType *distributedgrid;
        
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
        CoordinatesType localsize;

        bool isSet;
    
    public:
    
    DistributedMeshSynchronizer()
    {
        isSet=false;
    };

    DistributedMeshSynchronizer(DistributedGridType *distrgrid)
    {
        isSet=false;
        SetDistributedGrid(distrgrid);
    };

    void SetDistributedGrid(DistributedGridType *distrgrid)
    {
        isSet=true;

        this->distributedgrid=distrgrid;        
        overlap = this->distributedgrid->getOverlap();
        localsize = this->distributedgrid->getLocalSize();
        
        CoordinatesType localbegin=this->distributedgrid->getLocalBegin();
        CoordinatesType localgridsize = this->distributedgrid->getLocalGridSize();

        sizes[West]=sizes[East]=localsize.y()*localsize.z()*overlap.x();
        sizes[Nord]=sizes[South]=localsize.x()*localsize.z()*overlap.y();
        sizes[Bottom]=sizes[Top]=localsize.x()*localsize.y()*overlap.z();
        
        sizes[NordWest]=sizes[NordEast]=sizes[SouthWest]=sizes[SouthEast]=localsize.z()*overlap.x()*overlap.y();
        sizes[BottomWest]=sizes[BottomEast]=sizes[TopWest]=sizes[TopEast]=localsize.y()*overlap.x()*overlap.z();
        sizes[BottomNord]=sizes[BottomSouth]=sizes[TopNord]=sizes[TopSouth]=localsize.x()*overlap.y()*overlap.z();
        
        sizes[BottomNordWest]=sizes[BottomNordEast]=sizes[BottomSouthWest]=sizes[BottomSouthEast]=
                sizes[TopNordWest]=sizes[TopNordEast]=sizes[TopSouthWest]=sizes[TopSouthEast]= 
                overlap.x()*overlap.y()*overlap.z();

        for(int i=0;i<26;i++)
        {
                sendbuffs[i].setSize(sizes[i]);
                rcvbuffs[i].setSize(sizes[i]);
        }
        
        westSrc=localbegin.x();
        eastSrc=localgridsize.x()-2*overlap.x();
        nordSrc=localbegin.y();
        southSrc=localgridsize.y()-2*overlap.y();
        bottomSrc=localbegin.z();
        topSrc=localgridsize.z()-2*overlap.z();
            
        xcenter=localbegin.x();
        ycenter=localbegin.y();
        zcenter=localbegin.z();
        
        westDst=0;
        eastDst=localgridsize.x()-overlap.x();
        nordDst=0;
        southDst=localgridsize.y()-overlap.y();
        bottomDst=0;
        topDst=localgridsize.z()-overlap.z();
        
    }
        
    template<typename CommunicatorType>
    void Synchronize(MeshFunctionType &meshfunction)
    {

        TNL_ASSERT_TRUE(isSet,"Synchronizer is not set, but used to Synchronize");

    	if(!distributedgrid->IsDistributed())
            return;
        
        int *neighbor=distributedgrid->getNeighbors();
        
        //fill send buffers
        CopyBuffers(meshfunction, sendbuffs, true,
            westSrc, eastSrc, nordSrc, southSrc, bottomSrc, topSrc,
            xcenter, ycenter, zcenter,
            overlap, localsize,
            neighbor);
        
        //async send and rcv
        typename CommunicatorType::Request req[52];
		                
        //send everithing, recieve everything 
        for(int i=0;i<26;i++)	
           if(neighbor[i]!=-1)
           {
               req[i]=CommunicatorType::ISend(sendbuffs[i].getData(), sizes[i], neighbor[i]);
               req[26+i]=CommunicatorType::IRecv(rcvbuffs[i].getData(), sizes[i], neighbor[i]);
           }
		   else
      	   {
               req[i]=CommunicatorType::NullRequest;
               req[26+i]=CommunicatorType::NullRequest;
           }

        //wait until send is done
        CommunicatorType::WaitAll(req,52);

        //copy data form rcv buffers
        CopyBuffers(meshfunction, rcvbuffs, false,
            westDst, eastDst, nordDst, southDst, bottomDst, topDst,
            xcenter, ycenter, zcenter,
            overlap, localsize,
            neighbor); 
 
    }
    
    private:    
    template <typename Real>
    void CopyBuffers(MeshFunctionType meshfunction, Containers::Array<Real, Device, Index> * buffers, bool toBuffer,
            int west, int east, int nord, int south, int bottom, int top,
            int xcenter, int ycenter, int zcenter,
            CoordinatesType shortDim, CoordinatesType longDim,
            int *neighbor)
    {
       //X-Y-Z
        if(neighbor[West]!=-1)
            BufferEntitiesHelper<MeshFunctionType,3,Real,Device>::BufferEntities(meshfunction,buffers[West].getData(),west,ycenter,zcenter,shortDim.x(),longDim.y(),longDim.z(),toBuffer);
        if(neighbor[East]!=-1)
            BufferEntitiesHelper<MeshFunctionType,3,Real,Device>::BufferEntities(meshfunction,buffers[East].getData(),east,ycenter,zcenter,shortDim.x(),longDim.y(),longDim.z(),toBuffer);
        if(neighbor[Nord]!=-1)
            BufferEntitiesHelper<MeshFunctionType,3,Real,Device>::BufferEntities(meshfunction,buffers[Nord].getData(),xcenter,nord,zcenter,longDim.x(),shortDim.y(),longDim.z(),toBuffer);
        if(neighbor[South]!=-1)
            BufferEntitiesHelper<MeshFunctionType,3,Real,Device>::BufferEntities(meshfunction,buffers[South].getData(),xcenter,south,zcenter,longDim.x(),shortDim.y(),longDim.z(),toBuffer);
        if(neighbor[Bottom]!=-1)
            BufferEntitiesHelper<MeshFunctionType,3,Real,Device>::BufferEntities(meshfunction,buffers[Bottom].getData(),xcenter,ycenter,bottom,longDim.x(),longDim.y(),shortDim.z(),toBuffer);
        if(neighbor[Top]!=-1)
            BufferEntitiesHelper<MeshFunctionType,3,Real,Device>::BufferEntities(meshfunction,buffers[Top].getData(),xcenter,ycenter,top,longDim.x(),longDim.y(),shortDim.z(),toBuffer);	
        //XY
        if(neighbor[NordWest]!=-1)
            BufferEntitiesHelper<MeshFunctionType,3,Real,Device>::BufferEntities(meshfunction,buffers[NordWest].getData(),west,nord,zcenter,shortDim.x(),shortDim.y(),longDim.z(),toBuffer);
        if(neighbor[NordEast]!=-1)
            BufferEntitiesHelper<MeshFunctionType,3,Real,Device>::BufferEntities(meshfunction,buffers[NordEast].getData(),east,nord,zcenter,shortDim.x(),shortDim.y(),longDim.z(),toBuffer);
        if(neighbor[SouthWest]!=-1)
            BufferEntitiesHelper<MeshFunctionType,3,Real,Device>::BufferEntities(meshfunction,buffers[SouthWest].getData(),west,south,zcenter,shortDim.x(),shortDim.y(),longDim.z(),toBuffer);
        if(neighbor[SouthEast]!=-1)
            BufferEntitiesHelper<MeshFunctionType,3,Real,Device>::BufferEntities(meshfunction,buffers[SouthEast].getData(),east,south,zcenter,shortDim.x(),shortDim.y(),longDim.z(),toBuffer);
        //XZ
        if(neighbor[BottomWest]!=-1)
            BufferEntitiesHelper<MeshFunctionType,3,Real,Device>::BufferEntities(meshfunction,buffers[BottomWest].getData(),west,ycenter,bottom,shortDim.x(),longDim.y(),shortDim.z(),toBuffer);
        if(neighbor[BottomEast]!=-1)
            BufferEntitiesHelper<MeshFunctionType,3,Real,Device>::BufferEntities(meshfunction,buffers[BottomEast].getData(),east,ycenter,bottom,shortDim.x(),longDim.y(),shortDim.z(),toBuffer);
        if(neighbor[TopWest]!=-1)
            BufferEntitiesHelper<MeshFunctionType,3,Real,Device>::BufferEntities(meshfunction,buffers[TopWest].getData(),west,ycenter,top,shortDim.x(),longDim.y(),shortDim.z(),toBuffer);
        if(neighbor[TopEast]!=-1)
            BufferEntitiesHelper<MeshFunctionType,3,Real,Device>::BufferEntities(meshfunction,buffers[TopEast].getData(),east,ycenter,top,shortDim.x(),longDim.y(),shortDim.z(),toBuffer);   
        //YZ
        if(neighbor[BottomNord]!=-1)
            BufferEntitiesHelper<MeshFunctionType,3,Real,Device>::BufferEntities(meshfunction,buffers[BottomNord].getData(),xcenter,nord,bottom,longDim.x(),shortDim.y(),shortDim.z(),toBuffer);
        if(neighbor[BottomSouth]!=-1)
            BufferEntitiesHelper<MeshFunctionType,3,Real,Device>::BufferEntities(meshfunction,buffers[BottomSouth].getData(),xcenter,south,bottom,longDim.x(),shortDim.y(),shortDim.z(),toBuffer);
        if(neighbor[TopNord]!=-1)
            BufferEntitiesHelper<MeshFunctionType,3,Real,Device>::BufferEntities(meshfunction,buffers[TopNord].getData(),xcenter,nord,top,longDim.x(),shortDim.y(),shortDim.z(),toBuffer);
        if(neighbor[TopSouth]!=-1)
            BufferEntitiesHelper<MeshFunctionType,3,Real,Device>::BufferEntities(meshfunction,buffers[TopSouth].getData(),xcenter,south,top,longDim.x(),shortDim.y(),shortDim.z(),toBuffer);
        //XYZ
        if(neighbor[BottomNordWest]!=-1)
            BufferEntitiesHelper<MeshFunctionType,3,Real,Device>::BufferEntities(meshfunction,buffers[BottomNordWest].getData(),west,nord,bottom,shortDim.x(),shortDim.y(),shortDim.z(),toBuffer);
        if(neighbor[BottomNordEast]!=-1)
            BufferEntitiesHelper<MeshFunctionType,3,Real,Device>::BufferEntities(meshfunction,buffers[BottomNordEast].getData(),east,nord,bottom,shortDim.x(),shortDim.y(),shortDim.z(),toBuffer);
        if(neighbor[BottomSouthWest]!=-1)
            BufferEntitiesHelper<MeshFunctionType,3,Real,Device>::BufferEntities(meshfunction,buffers[BottomSouthWest].getData(),west,south,bottom,shortDim.x(),shortDim.y(),shortDim.z(),toBuffer);
        if(neighbor[BottomSouthEast]!=-1)
            BufferEntitiesHelper<MeshFunctionType,3,Real,Device>::BufferEntities(meshfunction,buffers[BottomSouthEast].getData(),east,south,bottom,shortDim.x(),shortDim.y(),shortDim.z(),toBuffer);
        if(neighbor[TopNordWest]!=-1)
            BufferEntitiesHelper<MeshFunctionType,3,Real,Device>::BufferEntities(meshfunction,buffers[TopNordWest].getData(),west,nord,top,shortDim.x(),shortDim.y(),shortDim.z(),toBuffer);
        if(neighbor[TopNordEast]!=-1)
            BufferEntitiesHelper<MeshFunctionType,3,Real,Device>::BufferEntities(meshfunction,buffers[TopNordEast].getData(),east,nord,top,shortDim.x(),shortDim.y(),shortDim.z(),toBuffer);
        if(neighbor[TopSouthWest]!=-1)
            BufferEntitiesHelper<MeshFunctionType,3,Real,Device>::BufferEntities(meshfunction,buffers[TopSouthWest].getData(),west,south,top,shortDim.x(),shortDim.y(),shortDim.z(),toBuffer);
        if(neighbor[TopSouthEast]!=-1)
            BufferEntitiesHelper<MeshFunctionType,3,Real,Device>::BufferEntities(meshfunction,buffers[TopSouthEast].getData(),east,south,top,shortDim.x(),shortDim.y(),shortDim.z(),toBuffer);   

    }
};


} // namespace DistributedMeshes
} // namespace Meshes
} // namespace TNL
