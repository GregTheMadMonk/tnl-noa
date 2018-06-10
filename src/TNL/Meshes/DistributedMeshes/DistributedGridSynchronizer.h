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
      template< typename Real_ >
      using BufferEntitiesHelperType = BufferEntitiesHelper< MeshFunctionType, 1, Real_, Device >;

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

          this->distributedGrid=distrgrid;

          overlapSize = distributedGrid->getOverlap().x();

          sendbuffs[0].setSize(overlapSize);
          sendbuffs[1].setSize(overlapSize);
          rcvbuffs[0].setSize(overlapSize);
          rcvbuffs[1].setSize(overlapSize);      
      };

      template<typename CommunicatorType>
      void Synchronize(MeshFunctionType &meshFunction)
      {
          TNL_ASSERT_TRUE(isSet,"Synchronizer is not set, but used to Synchronize");

          if(!distributedGrid->isDistributed())
                  return;

          int leftN=distributedGrid->getLeft();
          int rightN=distributedGrid->getRight();

          int totalSize = meshFunction.getMesh().getDimensions().x();

          CopyBuffers(meshFunction, sendbuffs, true,
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

          CopyBuffers(meshFunction, rcvbuffs, false,
                  0, totalSize-overlapSize, overlapSize,
                  leftN, rightN);
      }

   private:
      template <typename Real_ >
      void CopyBuffers(MeshFunctionType meshFunction, TNL::Containers::Array<Real_,Device> * buffers, bool toBuffer,
              int left, int right,
              int size,
              int leftNeighbor, int rightNeighbor)
      {
         if(leftNeighbor!=-1)
            BufferEntitiesHelperType< Real_ >::BufferEntities(meshFunction,buffers[Left].getData(),left,size,toBuffer);
         if(rightNeighbor!=-1)
            BufferEntitiesHelperType< Real_ >::BufferEntities(meshFunction,buffers[Right].getData(),right,size,toBuffer);  
      }


      Containers::Array<RealType, Device> sendbuffs[2];
      Containers::Array<RealType, Device> rcvbuffs[2];
      int overlapSize;

      DistributedGridType *distributedGrid;

      bool isSet;
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
        template< typename Real_ >
        using BufferEntitiesHelperType = BufferEntitiesHelper< MeshFunctionType, 2, Real_, Device >;


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

          this->distributedGrid=distrgrid;

          overlap = distributedGrid->getOverlap();
          localSize = distributedGrid->getLocalSize();

          CoordinatesType localGridSize = this->distributedGrid->getLocalGridSize();
          CoordinatesType localBegin=this->distributedGrid->getLocalBegin();

          int upDownSize=localSize.x()*overlap.y();
          int leftRightSize=localSize.y()*overlap.x();
          int connerSize=overlap.x()*overlap.y();

          sizes[Left]=leftRightSize;
          sizes[Right]=leftRightSize;
          sizes[Up]=upDownSize;
          sizes[Down]=upDownSize;
          sizes[UpLeft]=connerSize;
          sizes[DownLeft]=connerSize;
          sizes[UpRight]=connerSize;
          sizes[DownRight]=connerSize;

          for(int i=0;i<8;i++)
          {
              sendbuffs[i].setSize(sizes[i]);
              rcvbuffs[i].setSize(sizes[i]);
          }

          leftSrc=localBegin.x();
          rightSrc=localGridSize.x()-2*overlap.x();
          upSrc=localBegin.y();
          downSrc=localGridSize.y()-2*overlap.y();

          xcenter=localBegin.x();
          ycenter=localBegin.y();

          leftDst=0;
          rightDst=localGridSize.x()-overlap.x();
          upDst=0;
          downDst=localGridSize.y()-overlap.y();                       
      }

      template<typename CommunicatorType>
      void Synchronize( MeshFunctionType &meshFunction)
      {

         TNL_ASSERT_TRUE(isSet,"Synchronizer is not set, but used to Synchronize");

         if(!distributedGrid->isDistributed())
              return;

         const int *neighbor=distributedGrid->getNeighbors();

         CopyBuffers(meshFunction, sendbuffs, true,
              leftSrc, rightSrc, upSrc, downSrc,
              xcenter, ycenter,
              overlap,localSize,
              neighbor);

         //async send and rcv
         typename CommunicatorType::Request req[16];

         //send everything, recieve everything 
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
         CopyBuffers(meshFunction, rcvbuffs, false,
              leftDst, rightDst, upDst, downDst,
              xcenter, ycenter,
              overlap,localSize,
              neighbor);
      }
    
   private:
      
      template< typename Real_ >
      void CopyBuffers(MeshFunctionType meshFunction, Containers::Array<Real_, Device, Index> * buffers, bool toBuffer,
                       int left, int right, int up, int down,
                       int xcenter, int ycenter,
                       CoordinatesType shortDim, CoordinatesType longDim,
                       const int *neighbor)
      {
         if(neighbor[Left]!=-1)        
            BufferEntitiesHelperType< Real_ >::BufferEntities( meshFunction, buffers[Left].getData(), left,ycenter,shortDim.x(),longDim.y(),toBuffer);
         if(neighbor[Right]!=-1)
            BufferEntitiesHelperType< Real_ >::BufferEntities( meshFunction, buffers[Right].getData(),right,ycenter,shortDim.x(),longDim.y(),toBuffer);
         if(neighbor[Up]!=-1)
            BufferEntitiesHelperType< Real_ >::BufferEntities( meshFunction, buffers[Up].getData(), xcenter,up,longDim.x(),shortDim.y(),toBuffer);
         if(neighbor[Down]!=-1)
            BufferEntitiesHelperType< Real_ >::BufferEntities( meshFunction, buffers[Down].getData(),xcenter,down,longDim.x(),shortDim.y(),toBuffer);
         if(neighbor[UpLeft]!=-1)
            BufferEntitiesHelperType< Real_ >::BufferEntities( meshFunction, buffers[UpLeft].getData(),left,up,shortDim.x(),shortDim.y(),toBuffer);
         if(neighbor[UpRight]!=-1)
            BufferEntitiesHelperType< Real_ >::BufferEntities( meshFunction, buffers[UpRight].getData(),right,up,shortDim.x(),shortDim.y(),toBuffer);
         if(neighbor[DownLeft]!=-1)        
            BufferEntitiesHelperType< Real_ >::BufferEntities( meshFunction, buffers[DownLeft].getData(),left,down,shortDim.x(),shortDim.y(),toBuffer);
         if(neighbor[DownRight]!=-1)
            BufferEntitiesHelperType< Real_ >::BufferEntities( meshFunction, buffers[DownRight].getData(),right,down,shortDim.x(),shortDim.y(),toBuffer);
      }
      

      DistributedGridType *distributedGrid;

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
      CoordinatesType localSize;

      bool isSet;      
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

    DistributedMeshSynchronizer(DistributedGridType *distrgrid)
    {
        isSet=false;
        SetDistributedGrid(distrgrid);
    };

    void SetDistributedGrid(DistributedGridType *distrgrid)
    {
        isSet=true;

        this->distributedGrid=distrgrid;        
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
    void Synchronize(MeshFunctionType &meshFunction)
    {

        TNL_ASSERT_TRUE(isSet,"Synchronizer is not set, but used to Synchronize");

    	if(!distributedGrid->isDistributed())
            return;
        
        const int *neighbor=distributedGrid->getNeighbors();
        
        //fill send buffers
        CopyBuffers(meshFunction, sendbuffs, true,
            westSrc, eastSrc, nordSrc, southSrc, bottomSrc, topSrc,
            xcenter, ycenter, zcenter,
            overlap, localSize,
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
        CopyBuffers(meshFunction, rcvbuffs, false,
            westDst, eastDst, nordDst, southDst, bottomDst, topDst,
            xcenter, ycenter, zcenter,
            overlap, localSize,
            neighbor); 
 
    }
    
    private:    
    template< typename Real_ >
    void CopyBuffers(MeshFunctionType meshFunction, Containers::Array<Real_, Device, Index> * buffers, bool toBuffer,
            int west, int east, int nord, int south, int bottom, int top,
            int xcenter, int ycenter, int zcenter,
            CoordinatesType shortDim, CoordinatesType longDim,
            const int *neighbor)
    {
       //X-Y-Z
        if(neighbor[West]!=-1)
            BufferEntitiesHelperType< Real_ >::BufferEntities(meshFunction,buffers[West].getData(),west,ycenter,zcenter,shortDim.x(),longDim.y(),longDim.z(),toBuffer);
        if(neighbor[East]!=-1)
            BufferEntitiesHelperType< Real_ >::BufferEntities(meshFunction,buffers[East].getData(),east,ycenter,zcenter,shortDim.x(),longDim.y(),longDim.z(),toBuffer);
        if(neighbor[North]!=-1)
            BufferEntitiesHelperType< Real_ >::BufferEntities(meshFunction,buffers[North].getData(),xcenter,nord,zcenter,longDim.x(),shortDim.y(),longDim.z(),toBuffer);
        if(neighbor[South]!=-1)
            BufferEntitiesHelperType< Real_ >::BufferEntities(meshFunction,buffers[South].getData(),xcenter,south,zcenter,longDim.x(),shortDim.y(),longDim.z(),toBuffer);
        if(neighbor[Bottom]!=-1)
            BufferEntitiesHelperType< Real_ >::BufferEntities(meshFunction,buffers[Bottom].getData(),xcenter,ycenter,bottom,longDim.x(),longDim.y(),shortDim.z(),toBuffer);
        if(neighbor[Top]!=-1)
            BufferEntitiesHelperType< Real_ >::BufferEntities(meshFunction,buffers[Top].getData(),xcenter,ycenter,top,longDim.x(),longDim.y(),shortDim.z(),toBuffer);	
        //XY
        if(neighbor[NorthWest]!=-1)
            BufferEntitiesHelperType< Real_ >::BufferEntities(meshFunction,buffers[NorthWest].getData(),west,nord,zcenter,shortDim.x(),shortDim.y(),longDim.z(),toBuffer);
        if(neighbor[NorthEast]!=-1)
            BufferEntitiesHelperType< Real_ >::BufferEntities(meshFunction,buffers[NorthEast].getData(),east,nord,zcenter,shortDim.x(),shortDim.y(),longDim.z(),toBuffer);
        if(neighbor[SouthWest]!=-1)
            BufferEntitiesHelperType< Real_ >::BufferEntities(meshFunction,buffers[SouthWest].getData(),west,south,zcenter,shortDim.x(),shortDim.y(),longDim.z(),toBuffer);
        if(neighbor[SouthEast]!=-1)
            BufferEntitiesHelperType< Real_ >::BufferEntities(meshFunction,buffers[SouthEast].getData(),east,south,zcenter,shortDim.x(),shortDim.y(),longDim.z(),toBuffer);
        //XZ
        if(neighbor[BottomWest]!=-1)
            BufferEntitiesHelperType< Real_ >::BufferEntities(meshFunction,buffers[BottomWest].getData(),west,ycenter,bottom,shortDim.x(),longDim.y(),shortDim.z(),toBuffer);
        if(neighbor[BottomEast]!=-1)
            BufferEntitiesHelperType< Real_ >::BufferEntities(meshFunction,buffers[BottomEast].getData(),east,ycenter,bottom,shortDim.x(),longDim.y(),shortDim.z(),toBuffer);
        if(neighbor[TopWest]!=-1)
            BufferEntitiesHelperType< Real_ >::BufferEntities(meshFunction,buffers[TopWest].getData(),west,ycenter,top,shortDim.x(),longDim.y(),shortDim.z(),toBuffer);
        if(neighbor[TopEast]!=-1)
            BufferEntitiesHelperType< Real_ >::BufferEntities(meshFunction,buffers[TopEast].getData(),east,ycenter,top,shortDim.x(),longDim.y(),shortDim.z(),toBuffer);   
        //YZ
        if(neighbor[BottomNorth]!=-1)
            BufferEntitiesHelperType< Real_ >::BufferEntities(meshFunction,buffers[BottomNorth].getData(),xcenter,nord,bottom,longDim.x(),shortDim.y(),shortDim.z(),toBuffer);
        if(neighbor[BottomSouth]!=-1)
            BufferEntitiesHelperType< Real_ >::BufferEntities(meshFunction,buffers[BottomSouth].getData(),xcenter,south,bottom,longDim.x(),shortDim.y(),shortDim.z(),toBuffer);
        if(neighbor[TopNorth]!=-1)
            BufferEntitiesHelperType< Real_ >::BufferEntities(meshFunction,buffers[TopNorth].getData(),xcenter,nord,top,longDim.x(),shortDim.y(),shortDim.z(),toBuffer);
        if(neighbor[TopSouth]!=-1)
            BufferEntitiesHelperType< Real_ >::BufferEntities(meshFunction,buffers[TopSouth].getData(),xcenter,south,top,longDim.x(),shortDim.y(),shortDim.z(),toBuffer);
        //XYZ
        if(neighbor[BottomNorthWest]!=-1)
            BufferEntitiesHelperType< Real_ >::BufferEntities(meshFunction,buffers[BottomNorthWest].getData(),west,nord,bottom,shortDim.x(),shortDim.y(),shortDim.z(),toBuffer);
        if(neighbor[BottomNorthEast]!=-1)
            BufferEntitiesHelperType< Real_ >::BufferEntities(meshFunction,buffers[BottomNorthEast].getData(),east,nord,bottom,shortDim.x(),shortDim.y(),shortDim.z(),toBuffer);
        if(neighbor[BottomSouthWest]!=-1)
            BufferEntitiesHelperType< Real_ >::BufferEntities(meshFunction,buffers[BottomSouthWest].getData(),west,south,bottom,shortDim.x(),shortDim.y(),shortDim.z(),toBuffer);
        if(neighbor[BottomSouthEast]!=-1)
            BufferEntitiesHelperType< Real_ >::BufferEntities(meshFunction,buffers[BottomSouthEast].getData(),east,south,bottom,shortDim.x(),shortDim.y(),shortDim.z(),toBuffer);
        if(neighbor[TopNorthWest]!=-1)
            BufferEntitiesHelperType< Real_ >::BufferEntities(meshFunction,buffers[TopNorthWest].getData(),west,nord,top,shortDim.x(),shortDim.y(),shortDim.z(),toBuffer);
        if(neighbor[TopNorthEast]!=-1)
            BufferEntitiesHelperType< Real_ >::BufferEntities(meshFunction,buffers[TopNorthEast].getData(),east,nord,top,shortDim.x(),shortDim.y(),shortDim.z(),toBuffer);
        if(neighbor[TopSouthWest]!=-1)
            BufferEntitiesHelperType< Real_ >::BufferEntities(meshFunction,buffers[TopSouthWest].getData(),west,south,top,shortDim.x(),shortDim.y(),shortDim.z(),toBuffer);
        if(neighbor[TopSouthEast]!=-1)
            BufferEntitiesHelperType< Real_ >::BufferEntities(meshFunction,buffers[TopSouthEast].getData(),east,south,top,shortDim.x(),shortDim.y(),shortDim.z(),toBuffer);   
    }
};


} // namespace DistributedMeshes
} // namespace Meshes
} // namespace TNL
