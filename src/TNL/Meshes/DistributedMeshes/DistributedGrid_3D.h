/***************************************************************************
                          DistributedGrid_3D.h  -  description
                             -------------------
    begin                : January 15, 2018
    copyright            : (C) 2018 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Meshes/Grid.h>

namespace TNL {
namespace Meshes { 
namespace DistributedMeshes {

enum Directions3D { West = 0 , East = 1 , Nord = 2, South=3, Top =4, Bottom=5, 
                  NordWest=6, NordEast=7, SouthWest=8, SouthEast=9,
                  BottomWest=10,BottomEast=11,BottomNord=12,BottomSouth=13,
                  TopWest=14,TopEast=15,TopNord=16,TopSouth=17,
                  BottomNordWest=18,BottomNordEast=19,BottomSouthWest=20,BottomSouthEast=21,
                  TopNordWest=22,TopNordEast=23,TopSouthWest=24,TopSouthEast=25
                  };


template<typename RealType, typename Device, typename Index >     
class DistributedMesh<Grid< 3, RealType, Device, Index >>
{

    public:

    typedef Index IndexType;
    typedef Grid< 3, RealType, Device, IndexType > GridType;
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
        CoordinatesType globalsize;
        CoordinatesType globalbegin;
        
        IndexType Dimensions;        
        bool isDistributed;
        
        int rank;
        int nproc;
        
        int procsdistr[3];
        int myproccoord[3];
        int numberoflarger[3];
        
        int neighbors[26];

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
           
           for (int i=0;i<26;i++)
                neighbors[i]=-1;
           
           Dimensions= GridType::getMeshDimension();
           spaceSteps=globalGrid.getSpaceSteps();

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
               myproccoord[2]=0;
               
               procsdistr[0]=1;
               procsdistr[1]=1;
               procsdistr[2]=1;               
               
               localorigin=globalGrid.getOrigin();
               localsize=globalGrid.getDimensions();
               globalsize=globalGrid.getDimensions();
               localgridsize=localsize;
               globalbegin=CoordinatesType(0);
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
               CommunicatorType::DimsCreate(nproc, 3, procsdistr);
               myproccoord[2]=rank/(procsdistr[0]*procsdistr[1]);
               myproccoord[1]=(rank%(procsdistr[0]*procsdistr[1]))/procsdistr[0];
               myproccoord[0]=(rank%(procsdistr[0]*procsdistr[1]))%procsdistr[0];

               //compute local mesh size 
               globalsize=globalGrid.getDimensions();                
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
                   globalbegin.x()=myproccoord[0]*localsize.x();
               else
                   globalbegin.x()=numberoflarger[0]*(localsize.x()+1)+(myproccoord[0]-numberoflarger[0])*localsize.x();
                 
               if(numberoflarger[1]>myproccoord[1])
                   globalbegin.y()=myproccoord[1]*localsize.y();
               else
                   globalbegin.y()=numberoflarger[1]*(localsize.y()+1)+(myproccoord[1]-numberoflarger[1])*localsize.y();
 
               if(numberoflarger[2]>myproccoord[2])
                   globalbegin.z()=myproccoord[2]*localsize.z();
               else
                   globalbegin.z()=numberoflarger[2]*(localsize.z()+1)+(myproccoord[2]-numberoflarger[2])*localsize.z();

               localorigin=globalGrid.getOrigin()+TNL::Containers::tnlDotProduct(globalGrid.getSpaceSteps(),globalbegin-overlap);
               
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
       }
       
       void setupGrid( GridType& grid)
       {
           TNL_ASSERT_TRUE(isSet,"DistributedGrid is not set, but used by SetupGrid");
           grid.setOrigin(localorigin);
           grid.setDimensions(localgridsize);
           //compute local proporions by sideefect
           grid.setSpaceSteps(spaceSteps);
           grid.SetDistMesh(this);
       };
       
       String printProcessCoords()
       {
           return convertToString(myproccoord[0])+String("-")+convertToString(myproccoord[1])+String("-")+convertToString(myproccoord[2]);
       };

       String printProcessDistr()
       {
           return convertToString(procsdistr[0])+String("-")+convertToString(procsdistr[1])+String("-")+convertToString(procsdistr[2]);
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

       //number of elements of global grid
       CoordinatesType getGlobalSize()
       {
           return this->globalsize;
       }

       //coordinates of begin of local subdomain without overlaps in global grid
       CoordinatesType getGlobalBegin()
       {
           return this->globalbegin;
       }
       
       private:
           
        int getRankOfProcCoord(int x, int y, int z)
        {
            return z*procsdistr[0]*procsdistr[1]+y*procsdistr[0]+x;
        }


};

} // namespace DistributedMeshes
} // namespace Meshes
} // namespace TNL
