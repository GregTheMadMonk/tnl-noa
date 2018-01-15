/***************************************************************************
                          DistributedGrid_1D.h  -  description
                             -------------------
    begin                : January 09, 2018
    copyright            : (C) 2018 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/Meshes/Grid.h>

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

    private : 

        PointType localorigin;
        CoordinatesType localbegin;
        CoordinatesType localsize;
        CoordinatesType localgridsize;
        CoordinatesType overlap;
        PointType spaceSteps;
        
        
        IndexType Dimensions;        
        bool isDistributed;
        
        int rank;
        int nproc;
        
        int numberoflarger;
        
        int left;
        int right;

        bool isSet;
        
     
   public:
       
       DistributedMesh()
       {
            isSet=false;
       };

       //compute everithing 
       template<typename Communicator>
       DistributedMesh(Communicator &comm,GridType globalGrid, CoordinatesType overlap, int *distribution=NULL)
       {      
           SetGlobalGrid(comm,globalGrid,overlap,distribution);      
       };

       template<typename Communicator>
       void SetGlobalGrid(Communicator comm, GridType globalGrid, CoordinatesType overlap, int *distribution=NULL)
       {
           isSet=true;

           this->overlap=overlap;
           
           left=-1;
           right=-1;
           
           Dimensions= GridType::getMeshDimension();
           spaceSteps=globalGrid.getSpaceSteps();

           isDistributed=false;
           if(comm.IsInitialized())
           {
               rank=comm.GetRank();
               this->nproc=comm.GetSize();
               //use only if have more than one process
               if(this->nproc>1)
               {
                   isDistributed=true;
               }
           }
           
           if(!isDistributed)
           {
               //Without distribution

               std::cout <<"BEZ Distribuce"<<std::endl;
               rank=0;
               localorigin=globalGrid.getOrigin();
               localsize=globalGrid.getDimensions();
               localgridsize=globalGrid.getDimensions();
               
               localbegin=CoordinatesType(0);
               return;
           }
           else
           { 
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
           grid.SetDistMesh(this);
       };
       
       void printcoords(std::ostream& out)
       {
           out<<rank<<":";
       };

       void printdistr(std::ostream& out)
       {
           out<<"("<<nproc<<"):";
       };       

       bool IsDistributed(void)
       {
           return this->isDistributed;
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



} // namespace DistributedMeshes
} // namespace Meshes
} // namespace TNL
