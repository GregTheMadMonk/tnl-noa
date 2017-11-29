/***************************************************************************
                          DistributedGrid.h  -  description
                             -------------------
    begin                : March 17, 2017
    copyright            : (C) 2017 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <iostream>

namespace TNL {
namespace Meshes {   

template<typename GridType,
        int meshDimensions= GridType::getMeshDimension()>    
class DistributedGrid
{

};
}
}

#include <TNL/Meshes/Grid.h>
#include <TNL/mpi-supp.h>

namespace TNL {
namespace Meshes { 

#ifndef USE_MPI

template<typename GridType>    
class DistributedGrid <GridType,1>
{
public:
    bool isMPIUsed(void)
    {
        return false;
    };
};

template<typename GridType>    
class DistributedGrid <GridType,2>
{
public:
    bool isMPIUsed(void)
    {
        return false;
    };
};

template<typename GridType>    
class DistributedGrid <GridType,3>
{
public:
    bool isMPIUsed(void)
    {
        return false;
    };
};

#else
//=============================================1D==================================
template<typename GridType>    
class DistributedGrid <GridType,1>
{

    public:
    
    typedef typename GridType::IndexType IndexType;
    typedef typename GridType::PointType PointType;
    typedef Containers::StaticVector< 1, IndexType > CoordinatesType;
    
    static constexpr int getMeshDimension() { return 1; };    

    private : 
        
        PointType localorigin;
        CoordinatesType localbegin;
        CoordinatesType localsize;
        CoordinatesType localgridsize;
        CoordinatesType overlap;
        PointType spaceSteps;
        
        
        IndexType Dimensions;        
        bool mpiInUse;
        
        int rank;
        int nproc;
        
        int numberoflarger;
        
        int left;
        int right;

        bool isSet;
        
     
   public:
       DistributedGrid()
       {
            isSet=false;
       };

       //compute everithing 
       DistributedGrid(GridType globalGrid, CoordinatesType overlap, int *distribution=NULL)
       {      
           SetGlobalGrid(globalGrid,overlap,distribution);      
       };

       void SetGlobalGrid(GridType globalGrid, CoordinatesType overlap, int *distribution=NULL)
       {
           isSet=true;

           this->overlap=overlap;
           
           left=-1;
           right=-1;
           
           Dimensions= GridType::getMeshDimension();
           spaceSteps=globalGrid.getSpaceSteps();
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

               std::cout <<"BEZ MPI"<<std::endl;
               rank=0;
               localorigin=globalGrid.getOrigin();
               localsize=globalGrid.getDimensions();
               localgridsize=globalGrid.getDimensions();
               
               localbegin=CoordinatesType(0);
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
               numberoflarger=globalGrid.getDimensions().x()%nproc;
                 
               localsize.x()=(globalGrid.getDimensions().x()/nproc);               
               if(numberoflarger>rank)
                    localsize.x()+=1;                      
                                  
               if(numberoflarger>rank)
                   localorigin.x()=globalGrid.getOrigin().x()
                                +(rank*localsize.x()-overlap.x())*globalGrid.getSpaceSteps().x();
               else
                   localorigin.x()=globalGrid.getOrigin().x()
                                +(numberoflarger*(localsize.x()+1)+(rank-numberoflarger)*localsize.x()-overlap.x())
                                *globalGrid.getSpaceSteps().x();
              
              localbegin=overlap;
               
               //vlevo neni prekryv
               if(left==-1)
               {
                   localorigin.x()+=overlap.x()*globalGrid.getSpaceSteps().x();
                   localbegin.x()=0;
               }
               
               localgridsize=localsize;
               //add overlaps
               if(left==-1||right==-1)
                   localgridsize.x()+=overlap.x();
               else
                   localgridsize.x()+=2*overlap.x();
                         
           }  
       }; 
       
       void SetupGrid( GridType& grid)
       {
           TNL_ASSERT_TRUE(isSet,"DistributedGrid is not set, but used by SetupGrid");
           grid.setOrigin(localorigin);
           grid.setDimensions(localgridsize);
           //compute local proporions by sideefect
           grid.setSpaceSteps(spaceSteps);
           grid.SetDistGrid(this);
       };
       
       void printcoords(std::ostream& out)
       {
           out<<rank<<":";
       };

       void printdistr(std::ostream& out)
       {
           out<<"("<<nproc<<"):";
       };       

       bool isMPIUsed(void)
       {
           return this->mpiInUse;
       };
       
       int getLeft()
       {
           TNL_ASSERT_TRUE(isSet,"DistributedGrid is not set, but used by getLeft");
           return this->left;
       };
       
       int getRight()
       {
           TNL_ASSERT_TRUE(isSet,"DistributedGrid is not set, but used by getRight");
           return this->right;
       };
       
       CoordinatesType getOverlap()
       {
           return this->overlap;
       };

       CoordinatesType getLocalSize()
       {
           return this->localsize;
       }

       CoordinatesType getLocalGridSize()
       {
           return this->localgridsize;
       }
       
              
       CoordinatesType getLocalBegin()
       {
           return this->localbegin;
       }
       
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
    
    static constexpr int getMeshDimension() { return 2; };
    
    private : 
        
        PointType spaceSteps;
        PointType localorigin;
        CoordinatesType localsize;//velikost gridu zpracovavane danym uzlem bez prekryvu
        CoordinatesType localbegin;//souradnice zacatku zpracovavane vypoctove oblasi
        CoordinatesType localgridsize;//velikost lokálního gridu včetně překryvů
        CoordinatesType overlap;
        
        
        IndexType Dimensions;        
        bool mpiInUse;
        
        int rank;
        int nproc;
        
        int procsdistr[2];
        int myproccoord[2];
        int numberoflarger[2];
        
        int neighbors[8];

        bool isSet;
     
   public:
       DistributedGrid()
       {
            isSet=false;
       };

       //compute everithing 
       DistributedGrid(GridType &globalGrid, CoordinatesType overlap, int *distribution=NULL)
       {      
           SetGlobalGrid(globalGrid,overlap,distribution);      
       };

       void SetGlobalGrid(GridType &globalGrid, CoordinatesType overlap, int *distribution=NULL)
       {
           isSet=true;

           this->overlap=overlap;
           
           for (int i=0;i<8;i++)
                neighbors[i]=-1;
           
           Dimensions= GridType::getMeshDimension();
           spaceSteps=globalGrid.getSpaceSteps();
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
               procsdistr[0]=1;
               procsdistr[1]=1;
               localorigin=globalGrid.getOrigin();
               localgridsize=globalGrid.getDimensions();
               localsize=globalGrid.getDimensions();
               localbegin.x()=0;
               localbegin.y()=0;
               
               return;
           }
           else
           {
               //With MPI
               //compute node distribution
               if(distribution!=NULL)
               {
                  procsdistr[0]=distribution[0];
                  procsdistr[1]=distribution[1];
               }
               else
               {
                  procsdistr[0]=0;
                  procsdistr[1]=0;
               }
               MPI_Dims_create(nproc, 2, procsdistr);
               myproccoord[0]=rank%procsdistr[0];
               myproccoord[1]=rank/procsdistr[0];        

               //compute local mesh size              
               numberoflarger[0]=globalGrid.getDimensions().x()%procsdistr[0];
               numberoflarger[1]=globalGrid.getDimensions().y()%procsdistr[1];

               localsize.x()=(globalGrid.getDimensions().x()/procsdistr[0]);
               localsize.y()=(globalGrid.getDimensions().y()/procsdistr[1]);

               if(numberoflarger[0]>myproccoord[0])
                    localsize.x()+=1;               
               if(numberoflarger[1]>myproccoord[1])
                   localsize.y()+=1;

               if(numberoflarger[0]>myproccoord[0])
                   localorigin.x()=globalGrid.getOrigin().x()
                                +(myproccoord[0]*localsize.x()-overlap.x())*globalGrid.getSpaceSteps().x();
               else
                   localorigin.x()=globalGrid.getOrigin().x()
                                +(numberoflarger[0]*(localsize.x()+1)+(myproccoord[0]-numberoflarger[0])*localsize.x()-overlap.x())
                                *globalGrid.getSpaceSteps().x();
               
               if(numberoflarger[1]>myproccoord[1])
                   localorigin.y()=globalGrid.getOrigin().y()
                                +(myproccoord[1]*localsize.y()-overlap.y())*globalGrid.getSpaceSteps().y();
               else
                   localorigin.y()=globalGrid.getOrigin().y()
                                +(numberoflarger[1]*(localsize.y()+1)+(myproccoord[1]-numberoflarger[1])*localsize.y()-overlap.y())
                                *globalGrid.getSpaceSteps().y();

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
               
               localbegin=overlap;

               if(neighbors[Left]==-1)
               {
                    localorigin.x()+=overlap.x()*globalGrid.getSpaceSteps().x();
                    localbegin.x()=0;
               }

               if(neighbors[Up]==-1)
               {
                   localorigin.y()+=overlap.y()*globalGrid.getSpaceSteps().y();
                   localbegin.y()=0;
               }

               localgridsize=localsize;
               //Add Overlaps
               if(neighbors[Left]!=-1)
                   localgridsize.x()+=overlap.x();
               if(neighbors[Right]!=-1)
                   localgridsize.x()+=overlap.x();

               if(neighbors[Up]!=-1)
                   localgridsize.y()+=overlap.y();
               if(neighbors[Down]!=-1)
                   localgridsize.y()+=overlap.y();
           }
                     
           
                      
       };
       
       void SetupGrid( GridType& grid)
       {
           TNL_ASSERT_TRUE(isSet,"DistributedGrid is not set, but used by SetupGrid");
           grid.setOrigin(localorigin);
           grid.setDimensions(localgridsize);
           //compute local proporions by sideefect
           grid.setSpaceSteps(spaceSteps);
           grid.SetDistGrid(this);
       };
       
       void printcoords(std::ostream& out)
       {
           out<<"("<<myproccoord[0]<<","<<myproccoord[1]<<"):";
       };

       void printdistr(std::ostream& out)
       {
           out<<"("<<procsdistr[0]<<","<<procsdistr[1]<<"):";
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
           TNL_ASSERT_TRUE(isSet,"DistributedGrid is not set, but used by getNeighbors");
           return this->neighbors;
       }
       
       CoordinatesType getLocalSize()
       {
           return this->localsize;
       }

       CoordinatesType getLocalGridSize()
       {
           return this->localgridsize;
       }
       
              
       CoordinatesType getLocalBegin()
       {
           return this->localbegin;
       }
       
       private:
           
        int getRangOfProcCoord(int x, int y)
        {
            return y*procsdistr[0]+x;
        }
};   

//========================3D======================================================
enum Directions3D { West = 0 , East = 1 , Nord = 2, South=3, Top =4, Bottom=5, 
                  NordWest=6, NordEast=7, SouthWest=8, SouthEast=9,
                  BottomWest=10,BottomEast=11,BottomNord=12,BottomSouth=13,
                  TopWest=14,TopEast=15,TopNord=16,TopSouth=17,
                  BottomNordWest=18,BottomNordEast=19,BottomSouthWest=20,BottomSouthEast=21,
                  TopNordWest=22,TopNordEast=23,TopSouthWest=24,TopSouthEast=25
                  };

template<typename GridType>    
class DistributedGrid <GridType,3>
{
    public:

    typedef typename GridType::IndexType IndexType;
    typedef typename GridType::PointType PointType;
    typedef Containers::StaticVector< 3, IndexType > CoordinatesType;
    
    static constexpr int getMeshDimension() { return 3; };    

    private : 
        
        PointType spaceSteps;
        PointType localorigin;
        CoordinatesType localsize;
        CoordinatesType localgridsize;
        CoordinatesType localbegin;
        CoordinatesType overlap;
        
        
        IndexType Dimensions;        
        bool mpiInUse;
        
        int rank;
        int nproc;
        
        int procsdistr[3];
        int myproccoord[3];
        int numberoflarger[3];
        
        int neighbors[26];

        bool isSet;
     
   public:
       DistributedGrid()
       {
            isSet=false;
       };

       //compute everithing 
       DistributedGrid(GridType &globalGrid, CoordinatesType overlap, int *distribution=NULL)
       {      
           SetGlobalGrid(globalGrid,overlap,distribution);      
       };

       void SetGlobalGrid(GridType &globalGrid, CoordinatesType overlap, int *distribution=NULL)
       {

           isSet=true;           

           this->overlap=overlap;
           
           for (int i=0;i<26;i++)
                neighbors[i]=-1;
           
           Dimensions= GridType::getMeshDimension();
           spaceSteps=globalGrid.getSpaceSteps();
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
               myproccoord[2]=0;
               
               procsdistr[0]=1;
               procsdistr[1]=1;
               procsdistr[2]=1;               
               
               localorigin=globalGrid.getOrigin();
               localsize=globalGrid.getDimensions();
               localgridsize=localsize;
               return;
           }
           else
           {
               //With MPI
               //compute node distribution
               if(distribution!=NULL)
               {
                  procsdistr[0]=distribution[0];
                  procsdistr[1]=distribution[1];
                  procsdistr[2]=distribution[2];
               }
               else
               {
                  procsdistr[0]=0;
                  procsdistr[1]=0;
                  procsdistr[2]=0;
               }
               MPI_Dims_create(nproc, 3, procsdistr);
               myproccoord[2]=rank/(procsdistr[0]*procsdistr[1]);
               myproccoord[1]=(rank%(procsdistr[0]*procsdistr[1]))/procsdistr[0];
               myproccoord[0]=(rank%(procsdistr[0]*procsdistr[1]))%procsdistr[0];

               //compute local mesh size               
               numberoflarger[0]=globalGrid.getDimensions().x()%procsdistr[0];
               numberoflarger[1]=globalGrid.getDimensions().y()%procsdistr[1];
               numberoflarger[2]=globalGrid.getDimensions().z()%procsdistr[2];
                 
               localsize.x()=(globalGrid.getDimensions().x()/procsdistr[0]);
               localsize.y()=(globalGrid.getDimensions().y()/procsdistr[1]);
               localsize.z()=(globalGrid.getDimensions().z()/procsdistr[2]);
               
               if(numberoflarger[0]>myproccoord[0])
                    localsize.x()+=1;               
               if(numberoflarger[1]>myproccoord[1])
                   localsize.y()+=1;
               if(numberoflarger[2]>myproccoord[2])
                   localsize.z()+=1;
                                  
               if(numberoflarger[0]>myproccoord[0])
                   localorigin.x()=globalGrid.getOrigin().x()
                                +(myproccoord[0]*localsize.x()-overlap.x())*globalGrid.getSpaceSteps().x();
               else
                   localorigin.x()=globalGrid.getOrigin().x()
                                +(numberoflarger[0]*(localsize.x()+1)+(myproccoord[0]-numberoflarger[0])*localsize.x()-overlap.x())
                                *globalGrid.getSpaceSteps().x();
               
               if(numberoflarger[1]>myproccoord[1])
                   localorigin.y()=globalGrid.getOrigin().y()
                                +(myproccoord[1]*localsize.y()-overlap.y())*globalGrid.getSpaceSteps().y();
               else
                   localorigin.y()=globalGrid.getOrigin().y()
                                +(numberoflarger[1]*(localsize.y()+1)+(myproccoord[1]-numberoflarger[1])*localsize.y()-overlap.y())
                                *globalGrid.getSpaceSteps().y();

               if(numberoflarger[2]>myproccoord[2])
                   localorigin.z()=globalGrid.getOrigin().z()
                                +(myproccoord[2]*localsize.z()-overlap.z())*globalGrid.getSpaceSteps().z();
               else
                   localorigin.z()=globalGrid.getOrigin().z()
                                +(numberoflarger[2]*(localsize.z()+1)+(myproccoord[2]-numberoflarger[2])*localsize.z()-overlap.z())
                                *globalGrid.getSpaceSteps().z();
               
               //nearnodes
               //X Y Z
               if(myproccoord[0]>0)
                   neighbors[West]=getRankOfProcCoord(myproccoord[0]-1,myproccoord[1],myproccoord[2]);               
               if(myproccoord[0]<procsdistr[0]-1)
                   neighbors[East]=getRankOfProcCoord(myproccoord[0]+1,myproccoord[1],myproccoord[2]);
               if(myproccoord[1]>0)
                   neighbors[Nord]=getRankOfProcCoord(myproccoord[0],myproccoord[1]-1,myproccoord[2]);
               if(myproccoord[1]<procsdistr[1]-1)
                   neighbors[South]=getRankOfProcCoord(myproccoord[0],myproccoord[1]+1,myproccoord[2]);
               if(myproccoord[2]>0)
                   neighbors[Bottom]=getRankOfProcCoord(myproccoord[0],myproccoord[1],myproccoord[2]-1);
               if(myproccoord[2]<procsdistr[2]-1)
                   neighbors[Top]=getRankOfProcCoord(myproccoord[0],myproccoord[1],myproccoord[2]+1);

               //XY
               if(myproccoord[0]>0 && myproccoord[1]>0)
                   neighbors[NordWest]=getRankOfProcCoord(myproccoord[0]-1,myproccoord[1]-1,myproccoord[2]);
               if(myproccoord[0]<procsdistr[0]-1 && myproccoord[1]>0)
                   neighbors[NordEast]=getRankOfProcCoord(myproccoord[0]+1,myproccoord[1]-1,myproccoord[2]);
               if(myproccoord[0]>0 && myproccoord[1]<procsdistr[1]-1)
                   neighbors[SouthWest]=getRankOfProcCoord(myproccoord[0]-1,myproccoord[1]+1,myproccoord[2]);
               if(myproccoord[0]<procsdistr[0]-1 && myproccoord[1]<procsdistr[1]-1)
                   neighbors[SouthEast]=getRankOfProcCoord(myproccoord[0]+1,myproccoord[1]+1,myproccoord[2]);             
               //XZ
               if(myproccoord[0]>0 && myproccoord[2]>0)
                   neighbors[BottomWest]=getRankOfProcCoord(myproccoord[0]-1,myproccoord[1],myproccoord[2]-1);
               if(myproccoord[0]<procsdistr[0]-1 && myproccoord[2]>0)
                   neighbors[BottomEast]=getRankOfProcCoord(myproccoord[0]+1,myproccoord[1],myproccoord[2]-1); 
               if(myproccoord[0]>0 && myproccoord[2]<procsdistr[2]-1)
                   neighbors[TopWest]=getRankOfProcCoord(myproccoord[0]-1,myproccoord[1],myproccoord[2]+1);
              if(myproccoord[0]<procsdistr[0]-1 && myproccoord[2]<procsdistr[2]-1)
                   neighbors[TopEast]=getRankOfProcCoord(myproccoord[0]+1,myproccoord[1],myproccoord[2]+1);
               //YZ
               if(myproccoord[1]>0 && myproccoord[2]>0)
                   neighbors[BottomNord]=getRankOfProcCoord(myproccoord[0],myproccoord[1]-1,myproccoord[2]-1);
               if(myproccoord[1]<procsdistr[1]-1 && myproccoord[2]>0)
                   neighbors[BottomSouth]=getRankOfProcCoord(myproccoord[0],myproccoord[1]+1,myproccoord[2]-1);
               if(myproccoord[1]>0 && myproccoord[2]<procsdistr[2]-1)
                   neighbors[TopNord]=getRankOfProcCoord(myproccoord[0],myproccoord[1]-1,myproccoord[2]+1);               
               if(myproccoord[1]<procsdistr[1]-1 && myproccoord[2]<procsdistr[2]-1)
                   neighbors[TopSouth]=getRankOfProcCoord(myproccoord[0],myproccoord[1]+1,myproccoord[2]+1);
               //XYZ
               if(myproccoord[0]>0 && myproccoord[1]>0 && myproccoord[2]>0 )
                   neighbors[BottomNordWest]=getRankOfProcCoord(myproccoord[0]-1,myproccoord[1]-1,myproccoord[2]-1);
               if(myproccoord[0]<procsdistr[0]-1 && myproccoord[1]>0 && myproccoord[2]>0 )
                   neighbors[BottomNordEast]=getRankOfProcCoord(myproccoord[0]+1,myproccoord[1]-1,myproccoord[2]-1);
               if(myproccoord[0]>0 && myproccoord[1]<procsdistr[1]-1 && myproccoord[2]>0 )
                   neighbors[BottomSouthWest]=getRankOfProcCoord(myproccoord[0]-1,myproccoord[1]+1,myproccoord[2]-1);
               if(myproccoord[0]<procsdistr[0]-1 && myproccoord[1]<procsdistr[1]-1 && myproccoord[2]>0 )
                   neighbors[BottomSouthEast]=getRankOfProcCoord(myproccoord[0]+1,myproccoord[1]+1,myproccoord[2]-1);
               if(myproccoord[0]>0 && myproccoord[1]>0 && myproccoord[2]<procsdistr[2]-1 )
                   neighbors[TopNordWest]=getRankOfProcCoord(myproccoord[0]-1,myproccoord[1]-1,myproccoord[2]+1);
               if(myproccoord[0]<procsdistr[0]-1 && myproccoord[1]>0 && myproccoord[2]<procsdistr[2]-1 )
                   neighbors[TopNordEast]=getRankOfProcCoord(myproccoord[0]+1,myproccoord[1]-1,myproccoord[2]+1);
               if(myproccoord[0]>0 && myproccoord[1]<procsdistr[1]-1 && myproccoord[2]<procsdistr[2]-1 )
                   neighbors[TopSouthWest]=getRankOfProcCoord(myproccoord[0]-1,myproccoord[1]+1,myproccoord[2]+1);
               if(myproccoord[0]<procsdistr[0]-1 && myproccoord[1]<procsdistr[1]-1 && myproccoord[2]<procsdistr[2]-1 )
                   neighbors[TopSouthEast]=getRankOfProcCoord(myproccoord[0]+1,myproccoord[1]+1,myproccoord[2]+1);   

               
               localbegin=overlap;
               
               if(neighbors[West]==-1)
               {
                   localorigin.x()+=overlap.x()*globalGrid.getSpaceSteps().x();
                   localbegin.x()=0;
               }
               if(neighbors[Nord]==-1)
               {
                   localorigin.y()+=overlap.y()*globalGrid.getSpaceSteps().y();
                   localbegin.y()=0;
               }
               if(neighbors[Bottom]==-1)
               {
                   localorigin.z()+=overlap.z()*globalGrid.getSpaceSteps().z();
                   localbegin.z()=0;
               }
               
               localgridsize=localsize;
               
               if(neighbors[West]!=-1)
                   localgridsize.x()+=overlap.x();
               if(neighbors[East]!=-1)
                   localgridsize.x()+=overlap.x();

               if(neighbors[Nord]!=-1)
                   localgridsize.y()+=overlap.y();
               if(neighbors[South]!=-1)
                   localgridsize.y()+=overlap.y();
               
               if(neighbors[Bottom]!=-1)
                   localgridsize.z()+=overlap.z();
               if(neighbors[Top]!=-1)
                   localgridsize.z()+=overlap.z();
               
           }                     
       };
       
       void SetupGrid( GridType& grid)
       {
           TNL_ASSERT_TRUE(isSet,"DistributedGrid is not set, but used by SetupGrid");
           grid.setOrigin(localorigin);
           grid.setDimensions(localgridsize);
           //compute local proporions by sideefect
           grid.setSpaceSteps(spaceSteps);
           grid.SetDistGrid(this);
       };
       
       void printcoords(std::ostream& out )
       {
           out<<"("<<myproccoord[0]<<","<<myproccoord[1]<<","<<myproccoord[2]<<"):";
       };

       void printdistr(std::ostream& out)
       {
           out<<"("<<procsdistr[0]<<","<<procsdistr[1]<<","<<procsdistr[2]<<"):";
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
           TNL_ASSERT_TRUE(isSet,"DistributedGrid is not set, but used by getNeighbors");
           return this->neighbors;
       }
       
       CoordinatesType getLocalSize()
       {
           return this->localsize;
       }
       
       CoordinatesType getLocalGridSize()
       {
           return this->localgridsize;
       }
       
       CoordinatesType getLocalBegin()
       {
           return this->localbegin;
       }
       
       private:
           
        int getRankOfProcCoord(int x, int y, int z)
        {
            return z*procsdistr[0]*procsdistr[1]+y*procsdistr[0]+x;
        }       
};  

#endif
} // namespace Meshes
} // namespace TNL

