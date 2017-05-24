/***************************************************************************
                          DistributedGrid.h  -  description
                             -------------------
    begin                : March 17, 2017
    copyright            : (C) 2017 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

//#include <mpi.h>
#include <TNL/Meshes/Grid.h>
#include <TNL/mpi-supp.h>
#include <iostream>

namespace TNL {
namespace Meshes {   

/*template< typename OutMeshFunction,
          typename InFunction,
          typename Real >*/

template<typename GridType,
        int meshDimensions= GridType::meshDimension>    
class DistributedGrid
{

};


//=============================================1D==================================
template<typename GridType>    
class DistributedGrid <GridType,1>
{

    public:
    
    typedef typename GridType::IndexType IndexType;
    typedef typename GridType::VertexType VertexType;
    typedef Containers::StaticVector< 1, IndexType > CoordinatesType;
    
    
    private : 
        
        GridType GlobalGrid;
        VertexType localorigin;
        CoordinatesType localsize;
        CoordinatesType overlap;
        
        
        IndexType Dimensions;        
        bool mpiInUse;
        
        int rank;
        int nproc;
        
        int numberoflarger;
        
        int left;
        int right;
        
     
   public:
       //compute everithing 
       DistributedGrid(GridType globalGrid)
       {
           
           //fuj
           overlap.x()=1;
           
           left=-1;
           right=-1;
           
           Dimensions= GridType::meshDimensions;
           GlobalGrid=globalGrid;
           //Detect MPI and number of process
           mpiInUse=false;
           if(MPI::Is_initialized())
           {
               rank=MPI::COMM_WORLD.Get_rank();
               this->nproc=MPI::COMM_WORLD.Get_size();
               //use MPI only if have more than one process
               if(this->nproc>1)
               {
                   mpiInUse=true;
               }
           }
           
           if(!mpiInUse)
           {
               //Without MPI
               rank=0;
               localorigin=GlobalGrid.getOrigin();
               localsize=GlobalGrid.getDimensions();
               return;
           }
           else
           {
               //With MPI
               
               //nearnodes
               if(rank!=0)
                   left=rank-1;
               if(rank!=nproc-1)
                   right=rank+1;
                  
               //compute local mesh size               
               numberoflarger=GlobalGrid.getDimensions().x()%nproc;
                 
               localsize.x()=(GlobalGrid.getDimensions().x()/nproc);               
               if(numberoflarger>rank)
                    localsize.x()+=1;                      
                                  
               if(numberoflarger>rank)
                   localorigin.x()=GlobalGrid.getOrigin().x()
                                +(rank*localsize.x()-overlap.x())*GlobalGrid.getSpaceSteps().x();
               else
                   localorigin.x()=GlobalGrid.getOrigin().x()
                                +(numberoflarger*(localsize.x()+1)+(rank-numberoflarger)*localsize.x()-overlap.x())
                                *GlobalGrid.getSpaceSteps().x();
               
               //vlevo neni prekryv
               if(left==-1)
                   localorigin.x()+=overlap.x()*GlobalGrid.getSpaceSteps().x();
               
               //add overlaps
               if(left==-1||right==-1)
                   localsize.x()+=overlap.x();
               else
                   localsize.x()+=2*overlap.x();
                         
           }
                     
           
                      
       };
       
       void SetupGrid( GridType& grid)
       {
           grid.setOrigin(localorigin);
           grid.setDimensions(localsize);
           //compute local proporions by sideefect
           grid.setSpaceSteps(GlobalGrid.getSpaceSteps());
           grid.SetDistGrid(this);
       };
       
       void printcoords(void)
       {
           std::cout<<rank<<":" <<endl;
       };
       
       bool isMPIUsed(void)
       {
           return this->mpiInUse;
       };
       
       int getLeft()
       {
           return this->left;
       };
       
       int getRight()
       {
           return this->right;
       };
       
       CoordinatesType getOverlap()
       {
           return this->overlap;
       };
};   

//========================2D======================================================
enum Directions { Left = 0 , Right = 1 , Up = 2, Down=3, UpLeft =4, UpRight=5, DownLeft=6, DownRight=7 }; 

template<typename GridType>    
class DistributedGrid <GridType,2>
{
    public:

    typedef typename GridType::IndexType IndexType;
    typedef typename GridType::PointType PointType;
    typedef Containers::StaticVector< 2, IndexType > CoordinatesType;
    
    
    private : 
        
        GridType GlobalGrid;
        PointType localorigin;
        CoordinatesType localsize;
        CoordinatesType overlap;
        
        
        IndexType Dimensions;        
        bool mpiInUse;
        
        int rank;
        int nproc;
        
        int procsdistr[2];
        int myproccoord[2];
        int numberoflarger[2];
        
        int neighbors[8];
     
   public:
       //compute everithing 
       DistributedGrid(GridType globalGrid)
       {
           
           //fuj
           overlap.x()=1;
           overlap.y()=1;
           
           for (int i=0;i<8;i++)
                neighbors[i]=-1;
           
           Dimensions= GridType::meshDimension;
           GlobalGrid=globalGrid;
           //Detect MPI and number of process
           mpiInUse=false;
           
           
           
           if(MPI::Is_initialized())
           {
               rank=MPI::COMM_WORLD.Get_rank();
               this->nproc=MPI::COMM_WORLD.Get_size();
               //use MPI only if have more than one process
               if(this->nproc>1)
               {
                   mpiInUse=true;
               }
           }
           
           if(!mpiInUse)
           {
               //Without MPI
               myproccoord[0]=0;
               myproccoord[1]=0;
               localorigin=GlobalGrid.getOrigin();
               localsize=GlobalGrid.getDimensions();
               return;
           }
           else
           {
               //With MPI
               //compute node distribution
               procsdistr[0]=0;
               procsdistr[1]=0;
               MPI_Dims_create(nproc, 2, procsdistr);
               myproccoord[0]=rank%procsdistr[0]; // CO je X a co Y? --x je 0 a je to sloupec
               myproccoord[1]=rank/procsdistr[0];        
               
               
               //compute local mesh size
               
               numberoflarger[0]=GlobalGrid.getDimensions().x()%procsdistr[0];
               numberoflarger[1]=GlobalGrid.getDimensions().y()%procsdistr[1];
                 
               localsize.x()=(GlobalGrid.getDimensions().x()/procsdistr[0]);
               localsize.y()=(GlobalGrid.getDimensions().y()/procsdistr[1]);
               
               if(numberoflarger[0]>myproccoord[0])
                    localsize.x()+=1;               
               if(numberoflarger[1]>myproccoord[1])
                   localsize.y()+=1;
                                  
               if(numberoflarger[0]>myproccoord[0])
                   localorigin.x()=GlobalGrid.getOrigin().x()
                                +(myproccoord[0]*localsize.x()-overlap.x())*GlobalGrid.getSpaceSteps().x();
               else
                   localorigin.x()=GlobalGrid.getOrigin().x()
                                +(numberoflarger[0]*(localsize.x()+1)+(myproccoord[0]-numberoflarger[0])*localsize.x()-overlap.x())
                                *GlobalGrid.getSpaceSteps().x();
               
               if(numberoflarger[1]>myproccoord[1])
                   localorigin.y()=GlobalGrid.getOrigin().y()
                                +(myproccoord[1]*localsize.y()-overlap.y())*GlobalGrid.getSpaceSteps().y();
               else
                   localorigin.y()=GlobalGrid.getOrigin().y()
                                +(numberoflarger[1]*(localsize.y()+1)+(myproccoord[1]-numberoflarger[1])*localsize.y()-overlap.y())
                                *GlobalGrid.getSpaceSteps().y();
                             
               //nearnodes
               if(myproccoord[0]>0)
                   neighbors[Left]=getRangOfProcCoord(myproccoord[0]-1,myproccoord[1]);
               if(myproccoord[0]<procsdistr[0]-1)
                   neighbors[Right]=getRangOfProcCoord(myproccoord[0]+1,myproccoord[1]);
               if(myproccoord[1]>0)
                   neighbors[Up]=getRangOfProcCoord(myproccoord[0],myproccoord[1]-1);
               if(myproccoord[1]<procsdistr[1]-1)
                   neighbors[Down]=getRangOfProcCoord(myproccoord[0],myproccoord[1]+1);
               if(myproccoord[0]>0 && myproccoord[1]>0)
                   neighbors[UpLeft]=getRangOfProcCoord(myproccoord[0]-1,myproccoord[1]-1);
               if(myproccoord[0]>0 && myproccoord[1]<procsdistr[1]-1)
                   neighbors[DownLeft]=getRangOfProcCoord(myproccoord[0]-1,myproccoord[1]+1);
               if(myproccoord[0]<procsdistr[0]-1 && myproccoord[1]>0)
                   neighbors[UpRight]=getRangOfProcCoord(myproccoord[0]+1,myproccoord[1]-1);
               if(myproccoord[0]<procsdistr[0]-1 && myproccoord[1]<procsdistr[1]-1)
                   neighbors[DownRight]=getRangOfProcCoord(myproccoord[0]+1,myproccoord[1]+1);
               
               
               if(neighbors[Left]==-1)
                   localorigin.x()+=overlap.x()*GlobalGrid.getSpaceSteps().x();
               if(neighbors[Up]==-1)
                   localorigin.y()+=overlap.y()*GlobalGrid.getSpaceSteps().y();
               
               //Tady je BUG pro distribuci v jednom řádku, jednom sloupci
               if(neighbors[Left]==-1||neighbors[Right]==-1)
                    localsize.x()+=overlap.x();
               else
                    localsize.x()+=2*overlap.x();
               
               if(neighbors[Up]==-1||neighbors[Down]==-1)
                    localsize.y()+=overlap.y();
               else
                    localsize.y()+=2*overlap.y();
           }
                     
           
                      
       };
       
       void SetupGrid( GridType& grid)
       {
           grid.setOrigin(localorigin);
           grid.setDimensions(localsize);
           //compute local proporions by sideefect
           grid.setSpaceSteps(GlobalGrid.getSpaceSteps());
           grid.SetDistGrid(this);
       };
       
       void printcoords(void)
       {
           std::cout<<"("<<myproccoord[0]<<","<<myproccoord[1]<<"):" <<endl;
       };
       
       bool isMPIUsed(void)
       {
           return this->mpiInUse;
       };
       
       CoordinatesType getOverlap()
       {
           return this->overlap;
       };
       
       int * getNeighbors()
       {
           return this->neighbors;
       }
       
       private:
           
        int getRangOfProcCoord(int x, int y)
        {
            return y*procsdistr[0]+x;
        }
};   

} // namespace Meshes
} // namespace TNL

