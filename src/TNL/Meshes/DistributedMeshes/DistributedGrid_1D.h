/***************************************************************************
                          DistributedGrid_1D.h  -  description
                             -------------------
    begin                : January 09, 2018
    copyright            : (C) 2018 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Meshes/Grid.h>
#include <TNL/Logger.h>

namespace TNL {
namespace Meshes { 
namespace DistributedMeshes {

template<typename RealType, typename Device, typename Index >     
class DistributedMesh<Grid< 1, RealType, Device, Index >>
{

    public:
    
      typedef Index IndexType;
      typedef Grid< 1, RealType, Device, IndexType > GridType;
      typedef typename GridType::PointType PointType;
      typedef Containers::StaticVector< 1, IndexType > CoordinatesType;

      static constexpr int getMeshDimension() { return 1; };    

      DistributedMesh()
      : isSet(false ){};

      bool setup( const Config::ParameterContainer& parameters,
                  const String& prefix )
      {
         this->domainDecomposition.x() = parameters.getParameter< int >( "grid-domain-decomposition-x" );
         return true;
      }      
      
      void setDomainDecomposition( const CoordinatesType& domainDecomposition )
      {
         this->domainDecomposition = domainDecomposition;
      }
      
      const CoordinatesType& getDomainDecomposition()
      {
         return this->domainDecomposition;
      }
      
      template<typename CommunicatorType>
      void setGlobalGrid(GridType globalGrid, CoordinatesType overlap, int *distribution=NULL)
      {
         isSet=true;

         this->overlap=overlap;

         left=-1;
         right=-1;

         Dimensions= GridType::getMeshDimension();
         spaceSteps=globalGrid.getSpaceSteps();

         distributed=false;
         if(CommunicatorType::IsInitialized())
         {
             rank=CommunicatorType::GetRank();
             this->nproc=CommunicatorType::GetSize();
             //use only if have more than one process
             if(this->nproc>1)
             {
                 distributed=true;
             }
         }

         if(!distributed)
         {
             //Without distribution

             std::cout <<"BEZ Distribuce"<<std::endl;
             rank=0;
             localOrigin=globalGrid.getOrigin();
             localSize=globalGrid.getDimensions();
             localGridSize=globalGrid.getDimensions();
             globalSize=globalGrid.getDimensions();
             globalBegin=CoordinatesType(0);

             localBegin=CoordinatesType(0);
             return;
         }
         else
         {            
             //nearnodes
             if(rank!=0)
                 left=rank-1;
             if(rank!=nproc-1)
                 right=rank+1;
             
             this->domainDecomposition[ 0 ] = rank;

             globalSize=globalGrid.getDimensions();                 

             //compute local mesh size               
             numberOfLarger=globalGrid.getDimensions().x()%nproc;

             localSize.x()=(globalGrid.getDimensions().x()/nproc);               
             if(numberOfLarger>rank)
                  localSize.x()+=1;                      

             if(numberOfLarger>rank)
             {
                 globalBegin.x()=rank*localSize.x();
                 localOrigin.x()=globalGrid.getOrigin().x()
                              +(globalBegin.x()-overlap.x())*globalGrid.getSpaceSteps().x();
             }
             else
             {
                 globalBegin.x()=numberOfLarger*(localSize.x()+1)+(rank-numberOfLarger)*localSize.x();
                 localOrigin.x()=(globalGrid.getOrigin().x()-overlap.x())
                              +globalBegin.x()*globalGrid.getSpaceSteps().x();
             }

            localBegin=overlap;

             //vlevo neni prekryv
             if(left==-1)
             {
                 localOrigin.x()+=overlap.x()*globalGrid.getSpaceSteps().x();
                 localBegin.x()=0;
             }

             localGridSize=localSize;
             //add overlaps
             if(left==-1||right==-1)
                 localGridSize.x()+=overlap.x();
             else
                 localGridSize.x()+=2*overlap.x();

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
         return convertToString(rank);
      };

      String printProcessDistr()
      {
         return convertToString(nproc);
      };       

      bool isDistributed()
      {
         return this->distributed;
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

      //number of elements of local sub domain WITHOUT overlap
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

      //number of elemnts of local sub domain WITH overlap
      CoordinatesType getLocalGridSize()
      {
         return this->localGridSize;
      }
       
      //coordinates of begin of local subdomain without overlaps in local grid       
      CoordinatesType getLocalBegin()
      {
         return this->localBegin;
      }
      
      void writeProlog( Logger& logger )
      {
         logger.writeParameter( "Domain decomposition:", this->getDomainDecomposition() );
      }
       
       
    private : 

      PointType localOrigin;
      CoordinatesType localBegin;
      CoordinatesType localSize;
      CoordinatesType localGridSize;
      CoordinatesType overlap;
      CoordinatesType globalSize;
      CoordinatesType globalBegin;
      PointType spaceSteps;
        
        
      IndexType Dimensions;        
      bool distributed;
        
      int rank;
      int nproc;
      
      CoordinatesType domainDecomposition;
      CoordinatesType subdomainCoordinates;      
        
      int numberOfLarger;
        
      int left;
      int right;

      bool isSet;
        
       
};



} // namespace DistributedMeshes
} // namespace Meshes
} // namespace TNL
