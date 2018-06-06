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
    
     
   public:
       DistributedMesh()
       {
            isSet=false;
       };

       template< typename CommunicatorType >
       void setGlobalGrid( GridType &globalGrid,
                           CoordinatesType overlap,
                           int *distribution=NULL )
       {
           isSet=true;

           this->overlap=overlap;
           
           for (int i=0;i<8;i++)
                neighbors[i]=-1;
           
           Dimensions= GridType::getMeshDimension();
           spaceSteps=globalGrid.getSpaceSteps();
           //Detect MPI and number of process
           distributed=false;
           
           if( CommunicatorType::IsInitialized() )
           {
               rank=CommunicatorType::GetRank();
               this->nproc=CommunicatorType::GetSize();
               //use MPI only if have more than one process
               if(this->nproc>1)
               {
                   distributed=true;
               }
           }
           
           if( !distributed )
           {
               //Without MPI
               processesCoordinates[0]=0;
               processesCoordinates[1]=0;
               procsdistr[0]=1;
               procsdistr[1]=1;
               localOrigin=globalGrid.getOrigin();
               localGridSize=globalGrid.getDimensions();
               localSize=globalGrid.getDimensions();
               globalSize=globalGrid.getDimensions();
               globalBegin=CoordinatesType(0);
               localBegin.x()=0;
               localBegin.y()=0;
               
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

               processesCoordinates[0]=rank%procsdistr[0];
               processesCoordinates[1]=rank/procsdistr[0];        

               //compute local mesh size
               globalSize=globalGrid.getDimensions();              
               numberOfLarger[0]=globalGrid.getDimensions().x()%procsdistr[0];
               numberOfLarger[1]=globalGrid.getDimensions().y()%procsdistr[1];

               localSize.x()=(globalGrid.getDimensions().x()/procsdistr[0]);
               localSize.y()=(globalGrid.getDimensions().y()/procsdistr[1]);

               if(numberOfLarger[0]>processesCoordinates[0])
                    localSize.x()+=1;               
               if(numberOfLarger[1]>processesCoordinates[1])
                   localSize.y()+=1;

               if(numberOfLarger[0]>processesCoordinates[0])
                   globalBegin.x()=processesCoordinates[0]*localSize.x();
               else
                   globalBegin.x()=numberOfLarger[0]*(localSize.x()+1)+(processesCoordinates[0]-numberOfLarger[0])*localSize.x();

               if(numberOfLarger[1]>processesCoordinates[1])
                   globalBegin.y()=processesCoordinates[1]*localSize.y();

               else
                   globalBegin.y()=numberOfLarger[1]*(localSize.y()+1)+(processesCoordinates[1]-numberOfLarger[1])*localSize.y();

               localOrigin=globalGrid.getOrigin()+TNL::Containers::tnlDotProduct(globalGrid.getSpaceSteps(),globalBegin-overlap);

               //nearnodes
               if(processesCoordinates[0]>0)
                   neighbors[Left]=getRankOfProcCoord(processesCoordinates[0]-1,processesCoordinates[1]);
               if(processesCoordinates[0]<procsdistr[0]-1)
                   neighbors[Right]=getRankOfProcCoord(processesCoordinates[0]+1,processesCoordinates[1]);
               if(processesCoordinates[1]>0)
                   neighbors[Up]=getRankOfProcCoord(processesCoordinates[0],processesCoordinates[1]-1);
               if(processesCoordinates[1]<procsdistr[1]-1)
                   neighbors[Down]=getRankOfProcCoord(processesCoordinates[0],processesCoordinates[1]+1);
               if(processesCoordinates[0]>0 && processesCoordinates[1]>0)
                   neighbors[UpLeft]=getRankOfProcCoord(processesCoordinates[0]-1,processesCoordinates[1]-1);
               if(processesCoordinates[0]>0 && processesCoordinates[1]<procsdistr[1]-1)
                   neighbors[DownLeft]=getRankOfProcCoord(processesCoordinates[0]-1,processesCoordinates[1]+1);
               if(processesCoordinates[0]<procsdistr[0]-1 && processesCoordinates[1]>0)
                   neighbors[UpRight]=getRankOfProcCoord(processesCoordinates[0]+1,processesCoordinates[1]-1);
               if(processesCoordinates[0]<procsdistr[0]-1 && processesCoordinates[1]<procsdistr[1]-1)
                   neighbors[DownRight]=getRankOfProcCoord(processesCoordinates[0]+1,processesCoordinates[1]+1);
               
               localBegin=overlap;

               if(neighbors[Left]==-1)
               {
                    localOrigin.x()+=overlap.x()*globalGrid.getSpaceSteps().x();
                    localBegin.x()=0;
               }

               if(neighbors[Up]==-1)
               {
                   localOrigin.y()+=overlap.y()*globalGrid.getSpaceSteps().y();
                   localBegin.y()=0;
               }

               localGridSize=localSize;
               //Add Overlaps
               if(neighbors[Left]!=-1)
                   localGridSize.x()+=overlap.x();
               if(neighbors[Right]!=-1)
                   localGridSize.x()+=overlap.x();

               if(neighbors[Up]!=-1)
                   localGridSize.y()+=overlap.y();
               if(neighbors[Down]!=-1)
                   localGridSize.y()+=overlap.y();
           }
       }
       
       void setupGrid( GridType& grid)
       {
           TNL_ASSERT_TRUE(isSet,"DistributedGrid is not set, but used by SetupGrid");
           grid.setOrigin( localOrigin );
           grid.setDimensions( localGridSize );
           //compute local proporions by sideefect
           grid.setSpaceSteps( spaceSteps );
           grid.SetDistMesh(this);
       };
       
       String printProcessCoords()
       {
           return convertToString(processesCoordinates[0])+String("-")+convertToString(processesCoordinates[1]);
       };

       String printProcessDistr()
       {
           return convertToString(procsdistr[0])+String("-")+convertToString(procsdistr[1]);
       };  
       
       bool isDistributed()
       {
           return this->distributed;
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
           return this->localSize;
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

       CoordinatesType getLocalGridSize()
       {
           return this->localGridSize;
       }
       
              
       CoordinatesType getLocalBegin()
       {
           return this->localBegin;
       }
        
    private : 
       
        int getRankOfProcCoord(int x, int y)
        {
            return y*procsdistr[0]+x;
        }
        
        PointType spaceSteps;
        PointType localOrigin;
        CoordinatesType localSize;//velikost gridu zpracovavane danym uzlem bez prekryvu
        CoordinatesType localBegin;//souradnice zacatku zpracovavane vypoctove oblasi
        CoordinatesType localGridSize;//velikost lokálního gridu včetně překryvů
        CoordinatesType overlap;
        CoordinatesType globalSize;//velikost celé sítě
        CoordinatesType globalBegin;
        
        
        IndexType Dimensions;        
        bool distributed;
        
        int rank;
        int nproc;
        
        int procsdistr[2];
        CoordinatesType processesCoordinates;
        int numberOfLarger[2];
        
        int neighbors[8];

        bool isSet;
        

};

} // namespace DistributedMeshes
} // namespace Meshes
} // namespace TNL

