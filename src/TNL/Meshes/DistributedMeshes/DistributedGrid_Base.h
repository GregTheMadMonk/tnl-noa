/***************************************************************************
                          DistributedGrid_Base.h  -  part common for all dimensions
                             -------------------
    begin                : July 07, 2018
    copyright            : (C) 2018 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/


/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Meshes/Grid.h>
#include <TNL/Logger.h>
#include <TNL/Meshes/DistributedMeshes/Directions.h>


namespace TNL {
namespace Meshes { 
namespace DistributedMeshes {



template<int dim, typename RealType, typename Device, typename Index >     
class DistributedGrid_Base
{
  public:

      typedef Index IndexType;
      typedef Grid< dim, RealType, Device, IndexType > GridType;
      typedef typename GridType::PointType PointType;
      typedef Containers::StaticVector< dim, IndexType > CoordinatesType;

      static constexpr int getMeshDimension() { return dim; };  

      static constexpr int getNeighborsCount() { return DirectionCount<dim>::get(); } //c++14 may use Directions::pow3(dim)-1 

      DistributedGrid_Base();

      ~DistributedGrid_Base();
    
      void setDomainDecomposition( const CoordinatesType& domainDecomposition );      
      const CoordinatesType& getDomainDecomposition() const;

      bool isDistributed() const;
           
      const CoordinatesType& getOverlap() const;

      //number of elements of local sub domain WITHOUT overlap
      const CoordinatesType& getLocalSize() const;

      //dimensions of global grid
      const CoordinatesType& getGlobalSize() const;

      const GridType& getGlobalGrid() const;

      //coordinates of begin of local subdomain without overlaps in global grid
      const CoordinatesType& getGlobalBegin() const;

      //number of elements of local sub domain WITH overlap
      const CoordinatesType& getLocalGridSize() const;
       
      //coordinates of begin of local subdomain without overlaps in local grid       
      const CoordinatesType& getLocalBegin() const;

      const CoordinatesType& getSubdomainCoordinates() const;

      const PointType& getLocalOrigin() const;
      const PointType& getSpaceSteps() const;

      //aka MPI-communcicator  
      void setCommunicationGroup(void * group);
      void * getCommunicationGroup() const;

      template< int EntityDimension >
      IndexType getEntitiesCount() const;

      template< typename Entity >
      IndexType getEntitiesCount() const; 

      const int* getNeighbors() const; 

      template<typename CommunicatorType, typename DistributedGridType>
      bool SetupByCut(DistributedGridType &inputDistributedGrid, 
                 Containers::StaticVector<dim, int> savedDimensions, 
                 Containers::StaticVector<DistributedGridType::getMeshDimension()-dim,int> reducedDimensions, 
                 Containers::StaticVector<DistributedGridType::getMeshDimension()-dim,IndexType> fixedIndexs);

    int getRankOfProcCoord(const CoordinatesType &nodeCoordinates) const;

   public: 
      bool isThereNeighbor(const CoordinatesType &direction) const;

      void setupNeighbors();

      GridType globalGrid;
      PointType localOrigin;
      CoordinatesType localBegin;
      CoordinatesType localSize;
      CoordinatesType localGridSize;
      CoordinatesType overlap;
      //CoordinatesType globalDimensions;
      CoordinatesType globalBegin;
      PointType spaceSteps;

      CoordinatesType domainDecomposition;
      CoordinatesType subdomainCoordinates;   

      int neighbors[getNeighborsCount()];
      
      int periodicNeighbors[getNeighborsCount()];

      IndexType Dimensions;        
      bool distributed;
        
      int rank;
      int nproc;

      bool isSet;

      //aka MPI-communcicator 
      void * communicationGroup;

};

} // namespace DistributedMeshes
} // namespace Meshes
} // namespace TNL

#include <TNL/Meshes/DistributedMeshes/DistributedGrid_Base.hpp>
