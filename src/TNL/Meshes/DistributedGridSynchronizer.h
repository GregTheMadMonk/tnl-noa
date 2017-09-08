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
#include <TNL/Functions/MeshFunction.h>
#include <TNL/mpi-supp.h>

namespace TNL {
namespace Meshes {   

template <typename DistributedGridType,
		typename MeshFunctionType,
        int dim>  
class DistributedGridSynchronizer
{

};

//=============================================1D==================================

template <typename DistributedGridType,
		typename MeshFunctionType>  
class DistributedGridSynchronizer<DistributedGridType,MeshFunctionType,1>
{
    typedef typename MeshFunctionType::RealType Real;
    
    public:
    static void Synchronize(DistributedGridType distributedgrid, MeshFunctionType meshfunction)
    {
        if(!distributedgrid.isMPIUsed())
                return;
#ifdef USE_MPI
        

        Real * leftsendbuf;
        Real * rightsendbuf;
        Real * leftrcvbuf;
        Real * rightrcvbuf;

        int size = distributedgrid.getOverlap().x();

        leftsendbuf=new Real[size];
        rightsendbuf=new Real[size];
        leftrcvbuf=new Real[size];
        rightrcvbuf=new Real[size];

        //fill send buffers
        typename MeshFunctionType::MeshType::Cell leftentity(meshfunction.getMesh());
        typename MeshFunctionType::MeshType::Cell rightentity(meshfunction.getMesh());
        for(int i=0;i<size;i++)
        {
            leftentity.getCoordinates().x() = size+i;
            leftentity.refresh();
            //leftsendbuf[i]=meshfunction.getValue(leftentity);
            leftsendbuf[i]=meshfunction.getData()[leftentity.getIndex()];

            rightentity.getCoordinates().x() = meshfunction.getMesh().getDimensions().x()-2*size+i;
            rightentity.refresh();
            //rightsendbuf[i]=meshfunction.getValue(rightentity);
            rightsendbuf[i]=meshfunction.getData()[rightentity.getIndex()];

        }


        //async send
        MPI::Request leftsendreq;
        MPI::Request rightsendreq;
        MPI::Request leftrcvreq;
        MPI::Request rightrcvreq;

        //send everithing, recieve everything 
        //cout << distributedgrid.getLeft() << "   " << distributedgrid.getRight() << endl;
        if(distributedgrid.getLeft()!=-1)
        {
            leftsendreq=MPI::COMM_WORLD.Isend((void*) leftsendbuf, size, MPI::DOUBLE , distributedgrid.getLeft(), 0);
            leftrcvreq=MPI::COMM_WORLD.Irecv((void*) leftrcvbuf, size, MPI::DOUBLE, distributedgrid.getLeft(), 0);
        }        
        if(distributedgrid.getRight()!=-1)
        {
            rightsendreq=MPI::COMM_WORLD.Isend((void*) rightsendbuf, size, MPI::DOUBLE , distributedgrid.getRight(), 0);
            rightrcvreq=MPI::COMM_WORLD.Irecv((void*) rightrcvbuf, size, MPI::DOUBLE, distributedgrid.getRight(), 0);
        }

        //wait until send is done
        if(distributedgrid.getLeft()!=-1)
        {
            leftrcvreq.Wait();
            leftsendreq.Wait();
        }        
        if(distributedgrid.getRight()!=-1)
        {
            rightrcvreq.Wait();
            rightsendreq.Wait();
        }

        //copy data form rcv buffers
        if(distributedgrid.getLeft()!=-1)
        {
            for(int i=0;i<size;i++)
            {
                leftentity.getCoordinates().x() = i;
                leftentity.refresh();
                //leftsendbuf[i]=meshfunction.getValue(leftentity);
                meshfunction.getData()[leftentity.getIndex()]=leftrcvbuf[i];
            }
        }


        if(distributedgrid.getRight()!=-1)
        {
            for(int i=0;i<size;i++)
            {
                rightentity.getCoordinates().x() = meshfunction.getMesh().getDimensions().x()-size+i;
                rightentity.refresh();
                //rightsendbuf[i]=meshfunction.getValue(rightentity);
                meshfunction.getData()[rightentity.getIndex()]=rightrcvbuf[i];
            }
        }

        //free buffers
        delete [] leftrcvbuf;
        delete [] rightrcvbuf;
        delete [] leftsendbuf;
        delete [] rightsendbuf;  
#endif
    };
};

//=========================2D=================================================
template <typename DistributedGridType,
		typename MeshFunctionType>  
class DistributedGridSynchronizer<DistributedGridType,MeshFunctionType,2>
{
    public:
    static void Synchronize(DistributedGridType distributedgrid, MeshFunctionType meshfunction)
    {
	if(!distributedgrid.isMPIUsed())
            return;
#ifdef USE_MPI
        typedef typename MeshFunctionType::RealType Real;
        typedef typename DistributedGridType::CoordinatesType CoordinatesType;


        Real ** sendbuffs=new Real*[8];
        Real ** rcvbuffs=new Real*[8];

        int *neighbor=distributedgrid.getNeighbors();
        
        CoordinatesType overlap = distributedgrid.getOverlap();
        CoordinatesType localgridsize = meshfunction.getMesh().getDimensions();
        
        CoordinatesType localsize=distributedgrid.getLocalSize();
        CoordinatesType localbegin=distributedgrid.getLocalBegin();

        int updownsize=localsize.x()*overlap.y();
        int leftrightsize=localsize.y()*overlap.x();
        int connersize=overlap.x()*overlap.y();

        int sizes[8];
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

        //fill send buffers
	BufferEntities(meshfunction,sendbuffs[Left],localbegin.x(),localbegin.y(),overlap.x(),localsize.y(),true);
        BufferEntities(meshfunction,sendbuffs[Right],localgridsize.x()-2*overlap.x(),localbegin.y(),overlap.x(),localsize.y(),true);
	BufferEntities(meshfunction,sendbuffs[Up],localbegin.x(),localbegin.y(),localsize.x(),overlap.y(),true);
	BufferEntities(meshfunction,sendbuffs[Down],localbegin.x(),localgridsize.y()-2*overlap.y(),localsize.x(),overlap.y(),true);
        
	BufferEntities(meshfunction,sendbuffs[UpLeft],localbegin.x(),localbegin.y(),overlap.x(),overlap.y(),true);
	BufferEntities(meshfunction,sendbuffs[UpRight],localgridsize.x()-2*overlap.x(),localbegin.y(),overlap.x(),overlap.y(),true);
	BufferEntities(meshfunction,sendbuffs[DownLeft],localbegin.x(),localgridsize.y()-2*overlap.y(),overlap.x(),overlap.y(),true);
	BufferEntities(meshfunction,sendbuffs[DownRight],localgridsize.x()-2*overlap.x(),localgridsize.y()-2*overlap.y(),overlap.x(),overlap.y(),true);
		
        //async send
        MPI::Request sendreq[8];
        MPI::Request rcvreq[8];
		
        
                
        //send everithing, recieve everything 
        //cout << distributedgrid.getLeft() << "   " << distributedgrid.getRight() << endl;
        for(int i=0;i<8;i++)	
                if(neighbor[i]!=-1)
                {
                        sendreq[i]=MPI::COMM_WORLD.Isend((void*) sendbuffs[i], sizes[i], MPI::DOUBLE , neighbor[i], 0);
                        rcvreq[i]=MPI::COMM_WORLD.Irecv((void*) rcvbuffs[i], sizes[i], MPI::DOUBLE, neighbor[i], 0);
                }        

        //wait until send is done
        for(int i=0;i<8;i++)
        {
                if(neighbor[i]!=-1)
                {
                        sendreq[i].Wait();
                        rcvreq[i].Wait();
                }       
        }

        //copy data form rcv buffers
        if(neighbor[Left]!=-1)        
            BufferEntities(meshfunction,rcvbuffs[Left],0,localbegin.y(),overlap.x(),localsize.y(),false);
        if(neighbor[Right]!=-1)
            BufferEntities(meshfunction,rcvbuffs[Right],localgridsize.x()-overlap.x(),localbegin.y(),overlap.x(),localsize.y(),false);
        if(neighbor[Up]!=-1)
            BufferEntities(meshfunction,rcvbuffs[Up],localbegin.x(),0,localsize.x(),overlap.y(),false);
        if(neighbor[Down]!=-1)
            BufferEntities(meshfunction,rcvbuffs[Down],localbegin.x(),localgridsize.y()-overlap.y(),localsize.x(),overlap.y(),false);
        if(neighbor[UpLeft]!=-1)
            BufferEntities(meshfunction,rcvbuffs[UpLeft],0,0,overlap.x(),overlap.y(),false);
        if(neighbor[UpRight]!=-1)
            BufferEntities(meshfunction,rcvbuffs[UpRight],localgridsize.x()-overlap.x(),0,overlap.x(),overlap.y(),false);
        if(neighbor[DownLeft]!=-1)        
            BufferEntities(meshfunction,rcvbuffs[DownLeft],0,localgridsize.y()-overlap.y(),overlap.x(),overlap.y(),false);
        if(neighbor[DownRight]!=-1)
            BufferEntities(meshfunction,rcvbuffs[DownRight],localgridsize.x()-overlap.x(),localgridsize.y()-overlap.y(),overlap.x(),overlap.y(),false);
		
        //free buffers
        for(int i=0;i<8;i++)
        {
            delete [] sendbuffs[i];
            delete [] rcvbuffs[i];
        }

    };
    
    private:    
    template <typename Real>
    static void BufferEntities(MeshFunctionType meshfunction, Real * buffer, int beginx, int beginy, int sizex, int sizey,bool tobuffer)
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
#endif
    };
};

//=========================3D=================================================
template <typename DistributedGridType,
		typename MeshFunctionType>  
class DistributedGridSynchronizer<DistributedGridType,MeshFunctionType,3>
{
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
    
    public:
    
    DistributedGridSynchronizer(DistributedGridType *distgrid)
    {
#ifdef USE_MPI
        this->distributedgrid=distgrid;        
        overlap = this->distributedgrid->getOverlap();
        localsize=this->distributedgrid->getLocalSize();
        
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
        
#endif  
    }
    
    ~DistributedGridSynchronizer()
    {
#ifdef USE_MPI
       //free buffers
        for(int i=0;i<26;i++)
        {
            delete [] sendbuffs[i];
            delete [] rcvbuffs[i];
        }
#endif          
        
    }
        
    void Synchronize(MeshFunctionType meshfunction)
    {
	if(!distributedgrid->isMPIUsed())
            return;
#ifdef USE_MPI
        
        int *neighbor=distributedgrid->getNeighbors();
        
        //fill send buffers
        CopyBuffers(meshfunction, sendbuffs, true,
            westSrc, eastSrc, nordSrc, southSrc, bottomSrc, topSrc,
            xcenter, ycenter, zcenter,
            overlap, localsize,
            neighbor);
        
        //async send
        MPI::Request sendreq[26];
        MPI::Request rcvreq[26];
               
        for(int i=0;i<26;i++)	
                if(neighbor[i]!=-1)
                {
                        sendreq[i]=MPI::COMM_WORLD.Isend((void*) sendbuffs[i], sizes[i], MPI::DOUBLE , neighbor[i], 0);
                        rcvreq[i]=MPI::COMM_WORLD.Irecv((void*) rcvbuffs[i], sizes[i], MPI::DOUBLE, neighbor[i], 0);
                }        

        //wait until send is done
        for(int i=0;i<26;i++)
        {
                if(neighbor[i]!=-1)
                {
                        sendreq[i].Wait();
                        rcvreq[i].Wait();
                }       
        }

        //copy data form rcv buffers
               //fill send buffers
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

#endif
    };
};

} // namespace Meshes
} // namespace TNL