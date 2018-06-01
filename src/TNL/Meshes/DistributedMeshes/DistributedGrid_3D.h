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

     

      DistributedMesh()
      : isSet( false ) {};


      template< typename CommunicatorType > 
      void setGlobalGrid( GridType &globalGrid,
                          CoordinatesType overlap,
                           int *distribution=NULL )
      {
         isSet=true;           

         this->overlap=overlap;

         for (int i=0;i<26;i++)
              neighbors[i]=-1;

         Dimensions= GridType::getMeshDimension();
         spaceSteps=globalGrid.getSpaceSteps();

         distributed=false;



         if(CommunicatorType::IsInitialized())
         {
            rank=CommunicatorType::GetRank();
            this->nproc=CommunicatorType::GetSize();
            //use MPI only if have more than one process
            if(this->nproc>1)
            {
               distributed=true;
            }
         }

         if(!distributed)
         {
            //Without MPI
            processesCoordinates[0]=0;
            processesCoordinates[1]=0;
            processesCoordinates[2]=0;

            procsdistr[0]=1;
            procsdistr[1]=1;
            procsdistr[2]=1;               

            localOrigin=globalGrid.getOrigin();
            localSize=globalGrid.getDimensions();
            globalSize=globalGrid.getDimensions();
            localGridSize=localSize;
            globalBegin=CoordinatesType(0);
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
            processesCoordinates[2]=rank/(procsdistr[0]*procsdistr[1]);
            processesCoordinates[1]=(rank%(procsdistr[0]*procsdistr[1]))/procsdistr[0];
            processesCoordinates[0]=(rank%(procsdistr[0]*procsdistr[1]))%procsdistr[0];

            //compute local mesh size 
            globalSize=globalGrid.getDimensions();                
            numberoflarger[0]=globalGrid.getDimensions().x()%procsdistr[0];
            numberoflarger[1]=globalGrid.getDimensions().y()%procsdistr[1];
            numberoflarger[2]=globalGrid.getDimensions().z()%procsdistr[2];

            localSize.x()=(globalGrid.getDimensions().x()/procsdistr[0]);
            localSize.y()=(globalGrid.getDimensions().y()/procsdistr[1]);
            localSize.z()=(globalGrid.getDimensions().z()/procsdistr[2]);

            if(numberoflarger[0]>processesCoordinates[0])
               localSize.x()+=1;               
            if(numberoflarger[1]>processesCoordinates[1])
               localSize.y()+=1;
            if(numberoflarger[2]>processesCoordinates[2])
               localSize.z()+=1;

            if(numberoflarger[0]>processesCoordinates[0])
               globalBegin.x()=processesCoordinates[0]*localSize.x();
            else
               globalBegin.x()=numberoflarger[0]*(localSize.x()+1)+(processesCoordinates[0]-numberoflarger[0])*localSize.x();

            if(numberoflarger[1]>processesCoordinates[1])
               globalBegin.y()=processesCoordinates[1]*localSize.y();
            else
               globalBegin.y()=numberoflarger[1]*(localSize.y()+1)+(processesCoordinates[1]-numberoflarger[1])*localSize.y();

            if(numberoflarger[2]>processesCoordinates[2])
               globalBegin.z()=processesCoordinates[2]*localSize.z();
            else
               globalBegin.z()=numberoflarger[2]*(localSize.z()+1)+(processesCoordinates[2]-numberoflarger[2])*localSize.z();

            localOrigin=globalGrid.getOrigin()+TNL::Containers::tnlDotProduct(globalGrid.getSpaceSteps(),globalBegin-overlap);

            //nearnodes
            //X Y Z
            if(processesCoordinates[0]>0)
               neighbors[West]=getRankOfProcCoord(processesCoordinates[0]-1,processesCoordinates[1],processesCoordinates[2]);               
            if(processesCoordinates[0]<procsdistr[0]-1)
               neighbors[East]=getRankOfProcCoord(processesCoordinates[0]+1,processesCoordinates[1],processesCoordinates[2]);
            if(processesCoordinates[1]>0)
               neighbors[Nord]=getRankOfProcCoord(processesCoordinates[0],processesCoordinates[1]-1,processesCoordinates[2]);
            if(processesCoordinates[1]<procsdistr[1]-1)
               neighbors[South]=getRankOfProcCoord(processesCoordinates[0],processesCoordinates[1]+1,processesCoordinates[2]);
            if(processesCoordinates[2]>0)
               neighbors[Bottom]=getRankOfProcCoord(processesCoordinates[0],processesCoordinates[1],processesCoordinates[2]-1);
            if(processesCoordinates[2]<procsdistr[2]-1)
               neighbors[Top]=getRankOfProcCoord(processesCoordinates[0],processesCoordinates[1],processesCoordinates[2]+1);

            //XY
            if(processesCoordinates[0]>0 && processesCoordinates[1]>0)
               neighbors[NordWest]=getRankOfProcCoord(processesCoordinates[0]-1,processesCoordinates[1]-1,processesCoordinates[2]);
            if(processesCoordinates[0]<procsdistr[0]-1 && processesCoordinates[1]>0)
               neighbors[NordEast]=getRankOfProcCoord(processesCoordinates[0]+1,processesCoordinates[1]-1,processesCoordinates[2]);
            if(processesCoordinates[0]>0 && processesCoordinates[1]<procsdistr[1]-1)
               neighbors[SouthWest]=getRankOfProcCoord(processesCoordinates[0]-1,processesCoordinates[1]+1,processesCoordinates[2]);
            if(processesCoordinates[0]<procsdistr[0]-1 && processesCoordinates[1]<procsdistr[1]-1)
               neighbors[SouthEast]=getRankOfProcCoord(processesCoordinates[0]+1,processesCoordinates[1]+1,processesCoordinates[2]);             
            //XZ
            if(processesCoordinates[0]>0 && processesCoordinates[2]>0)
               neighbors[BottomWest]=getRankOfProcCoord(processesCoordinates[0]-1,processesCoordinates[1],processesCoordinates[2]-1);
            if(processesCoordinates[0]<procsdistr[0]-1 && processesCoordinates[2]>0)
               neighbors[BottomEast]=getRankOfProcCoord(processesCoordinates[0]+1,processesCoordinates[1],processesCoordinates[2]-1); 
            if(processesCoordinates[0]>0 && processesCoordinates[2]<procsdistr[2]-1)
               neighbors[TopWest]=getRankOfProcCoord(processesCoordinates[0]-1,processesCoordinates[1],processesCoordinates[2]+1);
            if(processesCoordinates[0]<procsdistr[0]-1 && processesCoordinates[2]<procsdistr[2]-1)
               neighbors[TopEast]=getRankOfProcCoord(processesCoordinates[0]+1,processesCoordinates[1],processesCoordinates[2]+1);
            //YZ
            if(processesCoordinates[1]>0 && processesCoordinates[2]>0)
               neighbors[BottomNord]=getRankOfProcCoord(processesCoordinates[0],processesCoordinates[1]-1,processesCoordinates[2]-1);
            if(processesCoordinates[1]<procsdistr[1]-1 && processesCoordinates[2]>0)
               neighbors[BottomSouth]=getRankOfProcCoord(processesCoordinates[0],processesCoordinates[1]+1,processesCoordinates[2]-1);
            if(processesCoordinates[1]>0 && processesCoordinates[2]<procsdistr[2]-1)
               neighbors[TopNord]=getRankOfProcCoord(processesCoordinates[0],processesCoordinates[1]-1,processesCoordinates[2]+1);               
            if(processesCoordinates[1]<procsdistr[1]-1 && processesCoordinates[2]<procsdistr[2]-1)
               neighbors[TopSouth]=getRankOfProcCoord(processesCoordinates[0],processesCoordinates[1]+1,processesCoordinates[2]+1);
            //XYZ
            if(processesCoordinates[0]>0 && processesCoordinates[1]>0 && processesCoordinates[2]>0 )
               neighbors[BottomNordWest]=getRankOfProcCoord(processesCoordinates[0]-1,processesCoordinates[1]-1,processesCoordinates[2]-1);
            if(processesCoordinates[0]<procsdistr[0]-1 && processesCoordinates[1]>0 && processesCoordinates[2]>0 )
               neighbors[BottomNordEast]=getRankOfProcCoord(processesCoordinates[0]+1,processesCoordinates[1]-1,processesCoordinates[2]-1);
            if(processesCoordinates[0]>0 && processesCoordinates[1]<procsdistr[1]-1 && processesCoordinates[2]>0 )
               neighbors[BottomSouthWest]=getRankOfProcCoord(processesCoordinates[0]-1,processesCoordinates[1]+1,processesCoordinates[2]-1);
            if(processesCoordinates[0]<procsdistr[0]-1 && processesCoordinates[1]<procsdistr[1]-1 && processesCoordinates[2]>0 )
               neighbors[BottomSouthEast]=getRankOfProcCoord(processesCoordinates[0]+1,processesCoordinates[1]+1,processesCoordinates[2]-1);
            if(processesCoordinates[0]>0 && processesCoordinates[1]>0 && processesCoordinates[2]<procsdistr[2]-1 )
               neighbors[TopNordWest]=getRankOfProcCoord(processesCoordinates[0]-1,processesCoordinates[1]-1,processesCoordinates[2]+1);
            if(processesCoordinates[0]<procsdistr[0]-1 && processesCoordinates[1]>0 && processesCoordinates[2]<procsdistr[2]-1 )
               neighbors[TopNordEast]=getRankOfProcCoord(processesCoordinates[0]+1,processesCoordinates[1]-1,processesCoordinates[2]+1);
            if(processesCoordinates[0]>0 && processesCoordinates[1]<procsdistr[1]-1 && processesCoordinates[2]<procsdistr[2]-1 )
               neighbors[TopSouthWest]=getRankOfProcCoord(processesCoordinates[0]-1,processesCoordinates[1]+1,processesCoordinates[2]+1);
            if(processesCoordinates[0]<procsdistr[0]-1 && processesCoordinates[1]<procsdistr[1]-1 && processesCoordinates[2]<procsdistr[2]-1 )
               neighbors[TopSouthEast]=getRankOfProcCoord(processesCoordinates[0]+1,processesCoordinates[1]+1,processesCoordinates[2]+1);   


            localBegin=overlap;

            if(neighbors[West]==-1)
            {
               localOrigin.x()+=overlap.x()*globalGrid.getSpaceSteps().x();
               localBegin.x()=0;
            }
            if(neighbors[Nord]==-1)
            {
               localOrigin.y()+=overlap.y()*globalGrid.getSpaceSteps().y();
               localBegin.y()=0;
            }
            if(neighbors[Bottom]==-1)
            {
               localOrigin.z()+=overlap.z()*globalGrid.getSpaceSteps().z();
               localBegin.z()=0;
            }

            localGridSize=localSize;

            if(neighbors[West]!=-1)
               localGridSize.x()+=overlap.x();
            if(neighbors[East]!=-1)
               localGridSize.x()+=overlap.x();

            if(neighbors[Nord]!=-1)
               localGridSize.y()+=overlap.y();
            if(neighbors[South]!=-1)
               localGridSize.y()+=overlap.y();

            if(neighbors[Bottom]!=-1)
               localGridSize.z()+=overlap.z();
            if(neighbors[Top]!=-1)
               localGridSize.z()+=overlap.z();
         }                     
      }
       
      void setupGrid( GridType& grid)
      {
         TNL_ASSERT_TRUE(isSet,"DistributedGrid is not set, but used by SetupGrid");
         grid.setOrigin(localOrigin);
         grid.setDimensions(localGridSize);
         //compute local proporions by sideefect
         grid.setSpaceSteps(spaceSteps);
         grid.SetDistMesh(this);
      };
       
      String printProcessCoords()
      {
         return convertToString(processesCoordinates[0])+String("-")+convertToString(processesCoordinates[1])+String("-")+convertToString(processesCoordinates[2]);
      };

      String printProcessDistr()
      {
         return convertToString(procsdistr[0])+String("-")+convertToString(procsdistr[1])+String("-")+convertToString(procsdistr[2]);
      };  

      bool isDistributed(void)
      {
         return this->distributed;
      };
       
      CoordinatesType getOverlap()
      {
         return this->overlap;
      };
       
      int* getNeighbors()
      {
         TNL_ASSERT_TRUE(isSet,"DistributedGrid is not set, but used by getNeighbors");
         return this->neighbors;
      }
       
      CoordinatesType getLocalSize()
      {
         return this->localSize;
      }
       
      CoordinatesType getLocalGridSize()
      {
         return this->localGridSize;
      }
       
      CoordinatesType getLocalBegin()
      {
         return this->localBegin;
      }

      //number of elements of global grid
      CoordinatesType getGlobalSize()
      {
         return this->globalSize;
      }

      //coordinates of begin of local subdomain without overlaps in global grid
      CoordinatesType getGlobalBegin()
      {
         return this->globalBegin;
      }
       
           

   private:

      int getRankOfProcCoord(int x, int y, int z)
      {
         return z*procsdistr[0]*procsdistr[1]+y*procsdistr[0]+x;
      }
        
      PointType spaceSteps;
      PointType localOrigin;
      CoordinatesType localSize;
      CoordinatesType localGridSize;
      CoordinatesType localBegin;
      CoordinatesType overlap;
      CoordinatesType globalSize;
      CoordinatesType globalBegin;
        
      IndexType Dimensions;        
      bool distributed;
        
      int rank;
      int nproc;
        
      int procsdistr[3];
      CoordinatesType processesCoordinates;
      int numberoflarger[3];
        
      int neighbors[26];

      bool isSet;
};

} // namespace DistributedMeshes
} // namespace Meshes
} // namespace TNL
