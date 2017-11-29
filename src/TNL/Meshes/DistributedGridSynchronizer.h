/***************************************************************************
                          DistributedGrid.h  -  description
                             -------------------
    begin                : March 17, 2017
    copyright            : (C) 2017 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once


#include <TNL/Meshes/DistributedGrid.h>
//#include <TNL/Functions/MeshFunction.h>
#include <TNL/mpi-supp.h>

namespace TNL {
namespace Meshes {   

template <typename DistributedGridType,
		typename MeshFunctionType,
        int dim=DistributedGridType::getMeshDimension()>  
class DistributedGridSynchronizer
{

};

//=============================================1D==================================

template <typename DistributedGridType,
		typename MeshFunctionType>  
class DistributedGridSynchronizer<DistributedGridType,MeshFunctionType,1>
{

#ifdef USE_MPI
public:
        typedef typename MeshFunctionType::MeshType::Cell Cell;
        typedef typename MeshFunctionType::RealType Real;

private:  
        Real * leftsendbuf;
        Real * rightsendbuf;
        Real * leftrcvbuf;
        Real * rightrcvbuf;

        int size;

        DistributedGridType *distributedgrid;

        bool isSet;

    
    public:
    DistributedGridSynchronizer()
    {
        isSet=false;
    };

    DistributedGridSynchronizer(DistributedGridType *distrgrid)
    {
        isSet=false;
        SetDistributedGrid(distrgrid);
    };

    void SetDistributedGrid(DistributedGridType *distrgrid)
    {
        if(isSet)
        {
            DeleteBuffers();
        }
        isSet=true;

        this->distributedgrid=distrgrid;

        size = distributedgrid->getOverlap().x();

        leftsendbuf=new Real[size];
        rightsendbuf=new Real[size];
        leftrcvbuf=new Real[size];
        rightrcvbuf=new Real[size];      
    };

    ~DistributedGridSynchronizer()
    {
        if(isSet)
        {
            DeleteBuffers();
        };
    };

    void Synchronize(MeshFunctionType &meshfunction)
    {
        TNL_ASSERT_TRUE(isSet,"Synchronizer is not set, but used to Synchronize");

        if(!distributedgrid->isMPIUsed())
                return;

        Cell leftentity(meshfunction.getMesh());
        Cell rightentity(meshfunction.getMesh());

        int left=distributedgrid->getLeft();
        int right=distributedgrid->getRight();

        //fill send buffers
        for(int i=0;i<size;i++)
        {
            if(left!=-1)
            {
                leftentity.getCoordinates().x() = size+i;
                leftentity.refresh();
                leftsendbuf[i]=meshfunction.getData()[leftentity.getIndex()];
            }
    
            if(right!=-1)
            {
                rightentity.getCoordinates().x() = meshfunction.getMesh().getDimensions().x()-2*size+i;
                rightentity.refresh();            
                rightsendbuf[i]=meshfunction.getData()[rightentity.getIndex()];
            }
        }

        //async send
        MPI::Request req[4];

        //send everithing, recieve everything 
        if(left!=-1)
        {
            req[0]=TNLMPI::ISend(leftsendbuf, size, left);
            req[2]=TNLMPI::IRecv(leftrcvbuf, size, left);
        }
        else
        {
            req[0]=MPI::REQUEST_NULL;
            req[2]=MPI::REQUEST_NULL;
        }        

        if(right!=-1)
        {
            req[1]=TNLMPI::ISend(rightsendbuf, size, right);
            req[3]=TNLMPI::IRecv(rightrcvbuf, size, right);
        }
        else
        {
            req[1]=MPI::REQUEST_NULL;
            req[3]=MPI::REQUEST_NULL;
        }

        //wait until send and recv is done
        MPI::Request::Waitall(4, req);

        //copy data form rcv buffers
        if(left!=-1)
        {
            for(int i=0;i<size;i++)
            {
                leftentity.getCoordinates().x() = i;
                leftentity.refresh();
                meshfunction.getData()[leftentity.getIndex()]=leftrcvbuf[i];
            }
        }

        if(right!=-1)
        {
            for(int i=0;i<size;i++)
            {
                rightentity.getCoordinates().x() = meshfunction.getMesh().getDimensions().x()-size+i;
                rightentity.refresh();
                meshfunction.getData()[rightentity.getIndex()]=rightrcvbuf[i];
            }
        }
    };

    private:
    void DeleteBuffers(void)
    {
        delete [] leftrcvbuf;
        delete [] rightrcvbuf;
        delete [] leftsendbuf;
        delete [] rightsendbuf; 
    };
#endif

};

//=========================2D=================================================
template <typename DistributedGridType,
		typename MeshFunctionType>  
class DistributedGridSynchronizer<DistributedGridType,MeshFunctionType,2>
{

#ifdef USE_MPI
    public:
        typedef typename MeshFunctionType::RealType Real;
        typedef typename DistributedGridType::CoordinatesType CoordinatesType;

    private:
        DistributedGridType *distributedgrid;

        Real * sendbuffs[8];
        Real * rcvbuffs[8];
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
    DistributedGridSynchronizer()
    {
        isSet=false;
    };

    DistributedGridSynchronizer(DistributedGridType *distrgrid)
    {
        isSet=false;
        SetDistributedGrid(distrgrid);
    };

    void SetDistributedGrid(DistributedGridType *distrgrid)
    {
        if(isSet)
        {
            DeleteBuffers();
        }
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
            sendbuffs[i]=new Real[sizes[i]];
            rcvbuffs[i]=new Real[sizes[i]];
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

    ~DistributedGridSynchronizer()
    {
        if(isSet)
            DeleteBuffers();
    }
        
    void Synchronize(MeshFunctionType &meshfunction)
    {

        TNL_ASSERT_TRUE(isSet,"Synchronizer is not set, but used to Synchronize");

	    if(!distributedgrid->isMPIUsed())
            return;

        int *neighbor=distributedgrid->getNeighbors();

        CopyBuffers(meshfunction, sendbuffs, true,
            leftSrc, rightSrc, upSrc, downSrc,
            xcenter, ycenter,
            overlap,localsize,
            neighbor);
	
        //async send and rcv
        MPI::Request req[16];
		                
        //send everithing, recieve everything 
        for(int i=0;i<8;i++)	
           if(neighbor[i]!=-1)
           {
               req[i]=TNLMPI::ISend(sendbuffs[i], sizes[i], neighbor[i]);
               req[8+i]=TNLMPI::IRecv(rcvbuffs[i], sizes[i], neighbor[i]);
           }
		   else
      	   {
               req[i]=MPI::REQUEST_NULL;
               req[8+i]=MPI::REQUEST_NULL;
           }

        //wait until send is done
        MPI::Request::Waitall(16, req);
        
        //copy data form rcv buffers
        CopyBuffers(meshfunction, rcvbuffs, false,
            leftDst, rightDst, upDst, downDst,
            xcenter, ycenter,
            overlap,localsize,
            neighbor);
    };
    
    private:
    template <typename Real>
    void CopyBuffers(MeshFunctionType meshfunction, Real ** buffers, bool toBuffer,
            int left, int right, int up, int down,
            int xcenter, int ycenter,
            CoordinatesType shortDim, CoordinatesType longDim,
            int *neighbor)
    {
       	if(neighbor[Left]!=-1)        
            BufferEntities(meshfunction,buffers[Left],left,ycenter,shortDim.x(),longDim.y(),toBuffer);
        if(neighbor[Right]!=-1)
            BufferEntities(meshfunction,buffers[Right],right,ycenter,shortDim.x(),longDim.y(),toBuffer);
        if(neighbor[Up]!=-1)
            BufferEntities(meshfunction,buffers[Up],xcenter,up,longDim.x(),shortDim.y(),toBuffer);
        if(neighbor[Down]!=-1)
            BufferEntities(meshfunction,buffers[Down],xcenter,down,longDim.x(),shortDim.y(),toBuffer);
        if(neighbor[UpLeft]!=-1)
            BufferEntities(meshfunction,buffers[UpLeft],left,up,shortDim.x(),shortDim.y(),toBuffer);
        if(neighbor[UpRight]!=-1)
            BufferEntities(meshfunction,buffers[UpRight],right,up,shortDim.x(),shortDim.y(),toBuffer);
        if(neighbor[DownLeft]!=-1)        
            BufferEntities(meshfunction,buffers[DownLeft],left,down,shortDim.x(),shortDim.y(),toBuffer);
        if(neighbor[DownRight]!=-1)
            BufferEntities(meshfunction,buffers[DownRight],right,down,shortDim.x(),shortDim.y(),toBuffer);
    }
    
    template <typename Real>
    void BufferEntities(MeshFunctionType meshfunction, Real * buffer, int beginx, int beginy, int sizex, int sizey,bool tobuffer)
    {

        typename MeshFunctionType::MeshType::Cell entity(meshfunction.getMesh());
        for(int i=0;i<sizey;i++)
        {
            for(int j=0;j<sizex;j++)
            {
                    entity.getCoordinates().x()=beginx+j;
                    entity.getCoordinates().y()=beginy+i;				
                    entity.refresh();
                    if(tobuffer)
                            buffer[i*sizex+j]=meshfunction.getData()[entity.getIndex()];
                    else
                            meshfunction.getData()[entity.getIndex()]=buffer[i*sizex+j];
            }
        }
    };
    
    void DeleteBuffers(void)
    {
        for(int i=0;i<8;i++)
        {
            delete [] sendbuffs[i];
            delete [] rcvbuffs[i];
        }
    };

#endif

};

//=========================3D=================================================
template <typename DistributedGridType,
		typename MeshFunctionType>  
class DistributedGridSynchronizer<DistributedGridType,MeshFunctionType,3>
{

#ifdef USE_MPI
    typedef typename MeshFunctionType::RealType Real;
    typedef typename DistributedGridType::CoordinatesType CoordinatesType;
      
    private:
        Real ** sendbuffs=new Real*[26];
        Real ** rcvbuffs=new Real*[26];
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
    
    DistributedGridSynchronizer()
    {
        isSet=false;
    };

    DistributedGridSynchronizer(DistributedGridType *distrgrid)
    {
        isSet=false;
        SetDistributedGrid(distrgrid);
    };

    void SetDistributedGrid(DistributedGridType *distrgrid)
    {
        if(isSet)
        {
            DeleteBuffers();
        }
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
                sendbuffs[i]=new Real[sizes[i]];
                rcvbuffs[i]=new Real[sizes[i]];
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
    
    ~DistributedGridSynchronizer()
    {
        if(isSet)
        {
            DeleteBuffers();
        }
    }
        
    void Synchronize(MeshFunctionType &meshfunction)
    {

        TNL_ASSERT_TRUE(isSet,"Synchronizer is not set, but used to Synchronize");

    	if(!distributedgrid->isMPIUsed())
            return;
        
        int *neighbor=distributedgrid->getNeighbors();
        
        //fill send buffers
        CopyBuffers(meshfunction, sendbuffs, true,
            westSrc, eastSrc, nordSrc, southSrc, bottomSrc, topSrc,
            xcenter, ycenter, zcenter,
            overlap, localsize,
            neighbor);
        
        //async send and rcv
        MPI::Request req[52];
		                
        //send everithing, recieve everything 
        for(int i=0;i<26;i++)	
           if(neighbor[i]!=-1)
           {
               req[i]=TNLMPI::ISend(sendbuffs[i], sizes[i], neighbor[i]);
               req[26+i]=TNLMPI::IRecv(rcvbuffs[i], sizes[i], neighbor[i]);
           }
		   else
      	   {
               req[i]=MPI::REQUEST_NULL;
               req[26+i]=MPI::REQUEST_NULL;
           }

        //wait until send is done
        MPI::Request::Waitall(52, req);

        //copy data form rcv buffers
        CopyBuffers(meshfunction, rcvbuffs, false,
            westDst, eastDst, nordDst, southDst, bottomDst, topDst,
            xcenter, ycenter, zcenter,
            overlap, localsize,
            neighbor); 
 
    };
    
    private:    
    template <typename Real>
    void BufferEntities(MeshFunctionType meshfunction, Real * buffer, int beginx, int beginy, int beginz, int sizex, int sizey, int sizez, bool tobuffer)
    {

        typename MeshFunctionType::MeshType::Cell entity(meshfunction.getMesh());
        for(int k=0;k<sizez;k++)
        {
            for(int i=0;i<sizey;i++)
            {
                for(int j=0;j<sizex;j++)
                {
                        entity.getCoordinates().x()=beginx+j;
                        entity.getCoordinates().y()=beginy+i;
                        entity.getCoordinates().z()=beginz+k;
                        entity.refresh();
                        if(tobuffer)
                                buffer[k*sizex*sizey+i*sizex+j]=meshfunction.getData()[entity.getIndex()];
                        else
                                meshfunction.getData()[entity.getIndex()]=buffer[k*sizex*sizey+i*sizex+j];
                }
            }
        }
        
    };
    
    template <typename Real>
    void CopyBuffers(MeshFunctionType meshfunction, Real ** buffers, bool toBuffer,
            int west, int east, int nord, int south, int bottom, int top,
            int xcenter, int ycenter, int zcenter,
            CoordinatesType shortDim, CoordinatesType longDim,
            int *neighbor)
    {
       //X-Y-Z
        if(neighbor[West]!=-1)
            BufferEntities(meshfunction,buffers[West],west,ycenter,zcenter,shortDim.x(),longDim.y(),longDim.z(),toBuffer);
        if(neighbor[East]!=-1)
            BufferEntities(meshfunction,buffers[East],east,ycenter,zcenter,shortDim.x(),longDim.y(),longDim.z(),toBuffer);
        if(neighbor[Nord]!=-1)
            BufferEntities(meshfunction,buffers[Nord],xcenter,nord,zcenter,longDim.x(),shortDim.y(),longDim.z(),toBuffer);
        if(neighbor[South]!=-1)
            BufferEntities(meshfunction,buffers[South],xcenter,south,zcenter,longDim.x(),shortDim.y(),longDim.z(),toBuffer);
        if(neighbor[Bottom]!=-1)
            BufferEntities(meshfunction,buffers[Bottom],xcenter,ycenter,bottom,longDim.x(),longDim.y(),shortDim.z(),toBuffer);
        if(neighbor[Top]!=-1)
            BufferEntities(meshfunction,buffers[Top],xcenter,ycenter,top,longDim.x(),longDim.y(),shortDim.z(),toBuffer);	
        //XY
        if(neighbor[NordWest]!=-1)
            BufferEntities(meshfunction,buffers[NordWest],west,nord,zcenter,shortDim.x(),shortDim.y(),longDim.z(),toBuffer);
        if(neighbor[NordEast]!=-1)
            BufferEntities(meshfunction,buffers[NordEast],east,nord,zcenter,shortDim.x(),shortDim.y(),longDim.z(),toBuffer);
        if(neighbor[SouthWest]!=-1)
            BufferEntities(meshfunction,buffers[SouthWest],west,south,zcenter,shortDim.x(),shortDim.y(),longDim.z(),toBuffer);
        if(neighbor[SouthEast]!=-1)
            BufferEntities(meshfunction,buffers[SouthEast],east,south,zcenter,shortDim.x(),shortDim.y(),longDim.z(),toBuffer);
        //XZ
        if(neighbor[BottomWest]!=-1)
            BufferEntities(meshfunction,buffers[BottomWest],west,ycenter,bottom,shortDim.x(),longDim.y(),shortDim.z(),toBuffer);
        if(neighbor[BottomEast]!=-1)
            BufferEntities(meshfunction,buffers[BottomEast],east,ycenter,bottom,shortDim.x(),longDim.y(),shortDim.z(),toBuffer);
        if(neighbor[TopWest]!=-1)
            BufferEntities(meshfunction,buffers[TopWest],west,ycenter,top,shortDim.x(),longDim.y(),shortDim.z(),toBuffer);
        if(neighbor[TopEast]!=-1)
            BufferEntities(meshfunction,buffers[TopEast],east,ycenter,top,shortDim.x(),longDim.y(),shortDim.z(),toBuffer);   
        //YZ
        if(neighbor[BottomNord]!=-1)
            BufferEntities(meshfunction,buffers[BottomNord],xcenter,nord,bottom,longDim.x(),shortDim.y(),shortDim.z(),toBuffer);
        if(neighbor[BottomSouth]!=-1)
            BufferEntities(meshfunction,buffers[BottomSouth],xcenter,south,bottom,longDim.x(),shortDim.y(),shortDim.z(),toBuffer);
        if(neighbor[TopNord]!=-1)
            BufferEntities(meshfunction,buffers[TopNord],xcenter,nord,top,longDim.x(),shortDim.y(),shortDim.z(),toBuffer);
        if(neighbor[TopSouth]!=-1)
            BufferEntities(meshfunction,buffers[TopSouth],xcenter,south,top,longDim.x(),shortDim.y(),shortDim.z(),toBuffer);
        //XYZ
        if(neighbor[BottomNordWest]!=-1)
            BufferEntities(meshfunction,buffers[BottomNordWest],west,nord,bottom,shortDim.x(),shortDim.y(),shortDim.z(),toBuffer);
        if(neighbor[BottomNordEast]!=-1)
            BufferEntities(meshfunction,buffers[BottomNordEast],east,nord,bottom,shortDim.x(),shortDim.y(),shortDim.z(),toBuffer);
        if(neighbor[BottomSouthWest]!=-1)
            BufferEntities(meshfunction,buffers[BottomSouthWest],west,south,bottom,shortDim.x(),shortDim.y(),shortDim.z(),toBuffer);
        if(neighbor[BottomSouthEast]!=-1)
            BufferEntities(meshfunction,buffers[BottomSouthEast],east,south,bottom,shortDim.x(),shortDim.y(),shortDim.z(),toBuffer);
        if(neighbor[TopNordWest]!=-1)
            BufferEntities(meshfunction,buffers[TopNordWest],west,nord,top,shortDim.x(),shortDim.y(),shortDim.z(),toBuffer);
        if(neighbor[TopNordEast]!=-1)
            BufferEntities(meshfunction,buffers[TopNordEast],east,nord,top,shortDim.x(),shortDim.y(),shortDim.z(),toBuffer);
        if(neighbor[TopSouthWest]!=-1)
            BufferEntities(meshfunction,buffers[TopSouthWest],west,south,top,shortDim.x(),shortDim.y(),shortDim.z(),toBuffer);
        if(neighbor[TopSouthEast]!=-1)
            BufferEntities(meshfunction,buffers[TopSouthEast],east,south,top,shortDim.x(),shortDim.y(),shortDim.z(),toBuffer);   

    };

    void DeleteBuffers(void)
    {
        //free buffers
        for(int i=0;i<26;i++)
        {
            delete [] sendbuffs[i];
            delete [] rcvbuffs[i];
        }
    };
#endif
};

} // namespace Meshes
} // namespace TNL
