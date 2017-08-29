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
    public:
    static void Synchronize(DistributedGridType distributedgrid, MeshFunctionType meshfunction)
    {
        if(!distributedgrid.isMPIUsed())
                return;
#ifdef HAVE_MPI
        typedef typename MeshFunctionType::RealType Real;

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
#ifdef HAVE_MPI
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
    public:
    static void Synchronize(DistributedGridType distributedgrid, MeshFunctionType meshfunction)
    {
	if(!distributedgrid.isMPIUsed())
            return;
#ifdef HAVE_MPI
        typedef typename MeshFunctionType::RealType Real;
        typedef typename DistributedGridType::CoordinatesType CoordinatesType;


        Real ** sendbuffs=new Real*[26];
        Real ** rcvbuffs=new Real*[26];

        CoordinatesType overlap = distributedgrid.getOverlap();
        CoordinatesType localgridsize = meshfunction.getMesh().getDimensions();
        CoordinatesType localsize=distributedgrid.getLocalSize();
        CoordinatesType localbegin=distributedgrid.getLocalBegin();

        int sizes[26];
        
        sizes[West]=sizes[East]=localsize.y()*localsize.z()*overlap.x();
        sizes[Nord]=sizes[South]=localsize.x()*localsize.z()*overlap.y();
        sizes[Top]=sizes[Bottom]=localsize.x()*localsize.y()*overlap.z();
        
        sizes[NordWest]=sizes[NordEast]=sizes[SouthWest]=sizes[SouthEast]=localsize.z()*overlap.x()*overlap.y();
        sizes[TopWest]=sizes[TopEast]=sizes[BottomWest]=sizes[BottomEast]=localsize.y()*overlap.x()*overlap.z();
        sizes[TopNord]=sizes[TopSouth]=sizes[BottomNord]=sizes[BottomSouth]=localsize.x()*overlap.y()*overlap.z();
        
        sizes[TopNordWest]=sizes[TopNordEast]=sizes[TopSouthWest]=sizes[TopSouthEast]=
                sizes[BottomNordWest]=sizes[BottomNordEast]=sizes[BottomSouthWest]=sizes[BottomSouthEast]=
                overlap.x()*overlap.y()*overlap.z();

        for(int i=0;i<26;i++)
        {
                sendbuffs[i]=new Real[sizes[i]];
                rcvbuffs[i]=new Real[sizes[i]];
        }
        
        
        //fill send buffers
        //X-Y-Z
	BufferEntities(meshfunction,sendbuffs[West],localbegin.x(),localbegin.y(),localbegin.z(),overlap.x(),localsize.y(),localsize.z(),true);
        BufferEntities(meshfunction,sendbuffs[East],localgridsize.x()-2*overlap.x(),localbegin.y(),localbegin.z(),overlap.x(),localsize.y(),localsize.z(),true);
	BufferEntities(meshfunction,sendbuffs[Nord],localbegin.x(),localbegin.y(),localbegin.z(),localsize.x(),overlap.y(),localsize.z(),true);
        BufferEntities(meshfunction,sendbuffs[South],localbegin.x(),localgridsize.y()-2*overlap.y(),localbegin.z(),localsize.x(),overlap.y(),localsize.z(),true);
        BufferEntities(meshfunction,sendbuffs[Top],localbegin.x(),localbegin.y(),localbegin.z(),localsize.x(),localsize.y(),overlap.z(),true);
        BufferEntities(meshfunction,sendbuffs[Bottom],localbegin.x(),localbegin.y(),localgridsize.z()-2*overlap.z(),localsize.x(),localsize.y(),overlap.z(),true);
	
        //XY
	BufferEntities(meshfunction,sendbuffs[NordWest],localbegin.x(),localbegin.y(),localbegin.z(),overlap.x(),overlap.y(),localsize.z(),true);
        BufferEntities(meshfunction,sendbuffs[NordEast],localgridsize.x()-2*overlap.x(),localbegin.y(),localbegin.z(),overlap.x(),overlap.y(),localsize.z(),true);
	BufferEntities(meshfunction,sendbuffs[SouthWest],localbegin.x(),localgridsize.y()-2*overlap.y(),localbegin.z(),overlap.x(),overlap.y(),localsize.z(),true);
        BufferEntities(meshfunction,sendbuffs[SouthEast],localgridsize.x()-2*overlap.x(),localgridsize.y()-2*overlap.y(),localbegin.z(),overlap.x(),overlap.y(),localsize.z(),true);
         
        //XZ
        BufferEntities(meshfunction,sendbuffs[TopWest],localbegin.x(),localbegin.y(),localbegin.z(),overlap.x(),localsize.y(),overlap.z(),true);
        BufferEntities(meshfunction,sendbuffs[TopEast],localgridsize.x()-2*overlap.x(),localbegin.y(),localbegin.z(),overlap.x(),localsize.y(),overlap.z(),true);
        BufferEntities(meshfunction,sendbuffs[BottomWest],localbegin.x(),localbegin.y(),localgridsize.z()-2*overlap.z(),overlap.x(),localsize.y(),overlap.z(),true);
        BufferEntities(meshfunction,sendbuffs[BottomEast],localgridsize.x()-2*overlap.x(),localbegin.y(),localgridsize.z()-2*overlap.z(),overlap.x(),localsize.y(),overlap.z(),true);
        
        //YZ
        BufferEntities(meshfunction,sendbuffs[TopNord],localbegin.x(),localbegin.y(),localbegin.z(),localsize.x(),overlap.y(),overlap.z(),true);
        BufferEntities(meshfunction,sendbuffs[TopSouth],localbegin.x(),localgridsize.y()-2*overlap.y(),localbegin.z(),localsize.x(),overlap.y(),overlap.z(),true);
        BufferEntities(meshfunction,sendbuffs[TopNord],localbegin.x(),localbegin.y(),localgridsize.z()-2*overlap.z(),localsize.x(),overlap.y(),overlap.z(),true);
        BufferEntities(meshfunction,sendbuffs[TopSouth],localbegin.x(),localgridsize.y()-2*overlap.y(),localgridsize.z()-2*overlap.z(),localsize.x(),overlap.y(),overlap.z(),true);

        //XYZ
        BufferEntities(meshfunction,sendbuffs[TopNordWest],localbegin.x(),localbegin.y(),localbegin.z(),overlap.x(),overlap.y(),overlap.z(),true);
        BufferEntities(meshfunction,sendbuffs[TopNordEast],localgridsize.x()-2*overlap.x(),localbegin.y(),localbegin.z(),overlap.x(),overlap.y(),overlap.z(),true);
        BufferEntities(meshfunction,sendbuffs[TopSouthWest],localbegin.x(),localgridsize.y()-2*overlap.y(),localbegin.z(),overlap.x(),overlap.y(),overlap.z(),true);
        BufferEntities(meshfunction,sendbuffs[TopSouthEast],localgridsize.x()-2*overlap.x(),localgridsize.y()-2*overlap.y(),localbegin.z(),overlap.x(),overlap.y(),overlap.z(),true);
        BufferEntities(meshfunction,sendbuffs[BottomNordWest],localbegin.x(),localbegin.y(),localgridsize.z()-2*overlap.z(),overlap.x(),overlap.y(),overlap.z(),true);
        BufferEntities(meshfunction,sendbuffs[BottomNordEast],localgridsize.x()-2*overlap.x(),localbegin.y(),localgridsize.z()-2*overlap.z(),overlap.x(),overlap.y(),overlap.z(),true);
        BufferEntities(meshfunction,sendbuffs[BottomSouthWest],localbegin.x(),localgridsize.y()-2*overlap.y(),localgridsize.z()-2*overlap.z(),overlap.x(),overlap.y(),overlap.z(),true);
        BufferEntities(meshfunction,sendbuffs[BottomSouthEast],localgridsize.x()-2*overlap.x(),localgridsize.y()-2*overlap.y(),localgridsize.z()-2*overlap.z(),overlap.x(),overlap.y(),overlap.z(),true);
 
        
        //async send
        MPI::Request sendreq[26];
        MPI::Request rcvreq[26];
		
        int *neighbor=distributedgrid.getNeighbors();
                
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
        /*if(neighbor[Left]!=-1)        
            BufferEntities(meshfunction,rcvbuffs[Left],0,overlap.y(),overlap.x(),localsize.y(),false);
        if(neighbor[Right]!=-1)
            BufferEntities(meshfunction,rcvbuffs[Right],localgridsize.x()-overlap.x(),overlap.y(),overlap.x(),localsize.y(),false);
        if(neighbor[Up]!=-1)
            BufferEntities(meshfunction,rcvbuffs[Up],overlap.x(),0,localsize.x(),overlap.y(),false);
        if(neighbor[Down]!=-1)
            BufferEntities(meshfunction,rcvbuffs[Down],overlap.x(),localgridsize.y()-overlap.y(),localsize.x(),overlap.y(),false);
        if(neighbor[UpLeft]!=-1)
            BufferEntities(meshfunction,rcvbuffs[UpLeft],0,0,overlap.x(),overlap.y(),false);
        if(neighbor[UpRight]!=-1)
            BufferEntities(meshfunction,rcvbuffs[UpRight],localgridsize.x()-overlap.x(),0,overlap.x(),overlap.y(),false);
        if(neighbor[DownLeft]!=-1)        
            BufferEntities(meshfunction,rcvbuffs[DownLeft],0,localgridsize.y()-overlap.y(),overlap.x(),overlap.y(),false);
        if(neighbor[DownRight]!=-1)
            BufferEntities(meshfunction,rcvbuffs[DownRight],localgridsize.x()-overlap.x(),localgridsize.y()-overlap.y(),overlap.x(),overlap.y(),false);
		*/
                
        //free buffers
        for(int i=0;i<26;i++)
        {
            delete [] sendbuffs[i];
            delete [] rcvbuffs[i];
        }

    };
    
    private:    
    template <typename Real>
    static void BufferEntities(MeshFunctionType meshfunction, Real * buffer, int beginx, int beginy, int beginz, int sizex, int sizey, int sizez, bool tobuffer)
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
#endif
    };
};

} // namespace Meshes
} // namespace TNL