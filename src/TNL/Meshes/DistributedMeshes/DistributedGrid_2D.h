/***************************************************************************
                          DistributedGrid_2D.h  -  description
                             -------------------
    begin                : January 09, 2018
    copyright            : (C) 2018 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Meshes/Grid.h>

namespace TNL {
namespace Meshes { 
namespace DistributedMeshes {

enum Directions2D { Left = 0 , Right = 1 , Up = 2, Down=3, UpLeft =4, UpRight=5, DownLeft=6, DownRight=7 }; 

template<typename RealType, typename Device, typename Index >     
class DistributedMesh<Grid< 2, RealType, Device, Index >>
{
    public:

    typedef Index IndexType;
    typedef Grid< 2, RealType, Device, IndexType > GridType;
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
        bool isDistributed;
        
        int rank;
        int nproc;
        
        int procsdistr[2];
        int myproccoord[2];
        int numberoflarger[2];
        
        int neighbors[8];

        bool isSet;
     
   public:
       DistributedMesh()
       {
            isSet=false;
       };

       //compute everithing
       template<typename CommunicatorType>
       void setGlobalGrid(GridType &globalGrid, CoordinatesType overlap, int *distribution=NULL)
       {
           isSet=true;

           this->overlap=overlap;
           
           for (int i=0;i<8;i++)
                neighbors[i]=-1;
           
           Dimensions= GridType::getMeshDimension();
           spaceSteps=globalGrid.getSpaceSteps();
           //Detect MPI and number of process
           isDistributed=false;
           
           
           
           if(CommunicatorType::IsInitialized())
           {
               rank=CommunicatorType::GetRank();
               this->nproc=CommunicatorType::GetSize();
               //use MPI only if have more than one process
               if(this->nproc>1)
               {
                   isDistributed=true;
               }
           }
           
           if(!isDistributed)
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
               CommunicatorType::DimsCreate(nproc, 2, procsdistr);
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
                   neighbors[Left]=getRankOfProcCoord(myproccoord[0]-1,myproccoord[1]);
               if(myproccoord[0]<procsdistr[0]-1)
                   neighbors[Right]=getRankOfProcCoord(myproccoord[0]+1,myproccoord[1]);
               if(myproccoord[1]>0)
                   neighbors[Up]=getRankOfProcCoord(myproccoord[0],myproccoord[1]-1);
               if(myproccoord[1]<procsdistr[1]-1)
                   neighbors[Down]=getRankOfProcCoord(myproccoord[0],myproccoord[1]+1);
               if(myproccoord[0]>0 && myproccoord[1]>0)
                   neighbors[UpLeft]=getRankOfProcCoord(myproccoord[0]-1,myproccoord[1]-1);
               if(myproccoord[0]>0 && myproccoord[1]<procsdistr[1]-1)
                   neighbors[DownLeft]=getRankOfProcCoord(myproccoord[0]-1,myproccoord[1]+1);
               if(myproccoord[0]<procsdistr[0]-1 && myproccoord[1]>0)
                   neighbors[UpRight]=getRankOfProcCoord(myproccoord[0]+1,myproccoord[1]-1);
               if(myproccoord[0]<procsdistr[0]-1 && myproccoord[1]<procsdistr[1]-1)
                   neighbors[DownRight]=getRankOfProcCoord(myproccoord[0]+1,myproccoord[1]+1);
               
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
                     
           
                      
       }
       
       void SetupGrid( GridType& grid)
       {
           TNL_ASSERT_TRUE(isSet,"DistributedGrid is not set, but used by SetupGrid");
           grid.setOrigin(localorigin);
           grid.setDimensions(localgridsize);
           //compute local proporions by sideefect
           grid.setSpaceSteps(spaceSteps);
           grid.SetDistMesh(this);
       };
       
       void printcoords(std::ostream& out)
       {
           out<<"("<<myproccoord[0]<<","<<myproccoord[1]<<"):";
       };

       void printdistr(std::ostream& out)
       {
           out<<"("<<procsdistr[0]<<","<<procsdistr[1]<<"):";
       };
       
       bool IsDistributed(void)
       {
           return this->isDistributed;
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
           
        int getRankOfProcCoord(int x, int y)
        {
            return y*procsdistr[0]+x;
        }


};

} // namespace DistributedMeshes
} // namespace Meshes
} // namespace TNL

