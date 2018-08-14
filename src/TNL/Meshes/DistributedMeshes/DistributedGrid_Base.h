/***************************************************************************
                          DistributedGrid_Base.h  -  part common for all Dimensionensions
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



template< int Dimension,
          typename RealType,
          typename Device,
          typename Index >     
class DistributedGrid_Base
{
  public:

      typedef Index IndexType;
      typedef Grid< Dimension, RealType, Device, IndexType > GridType;
      typedef typename GridType::PointType PointType;
      typedef Containers::StaticVector< Dimension, IndexType > CoordinatesType;
      typedef Containers::StaticVector< Dimension, IndexType > SubdomainOverlapsType;

      static constexpr int getMeshDimension() { return Dimension; };  

      static constexpr int getNeighborsCount() { return DirectionCount<Dimension>::get(); } //c++14 may use Directions::pow3(Dimension)-1 

      DistributedGrid_Base();

      ~DistributedGrid_Base();
    
      void setDomainDecomposition( const CoordinatesType& domainDecomposition );
      
      const CoordinatesType& getDomainDecomposition() const;
      
      template< typename CommunicatorType >
      void setGlobalGrid( const GridType& globalGrid );
      
      void setOverlaps( const SubdomainOverlapsType& lower,
                        const SubdomainOverlapsType& upper );

      bool isDistributed() const;
           
      // TODO: replace it with getLowerOverlap() and getUpperOverlap()
      const CoordinatesType& getOverlap() const;
      
      const CoordinatesType& getLowerOverlap() const;
      
      const CoordinatesType& getUpperOverlap() const;

      //number of elements of local sub domain WITHOUT overlap
      const CoordinatesType& getLocalSize() const;

      //Dimensionensions of global grid
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
      
      const int* getPeriodicNeighbors() const; 

      template<typename CommunicatorType, typename DistributedGridType>
      bool SetupByCut(DistributedGridType &inputDistributedGrid, 
                 Containers::StaticVector<Dimension, int> savedDimensions, 
                 Containers::StaticVector<DistributedGridType::getMeshDimension()-Dimension,int> reducedDimensions, 
                 Containers::StaticVector<DistributedGridType::getMeshDimension()-Dimension,IndexType> fixedIndexs);

      int getRankOfProcCoord(const CoordinatesType &nodeCoordinates) const;

   public: 
      
      bool isThereNeighbor(const CoordinatesType &direction) const;

      void setupNeighbors();
      
      void print( ostream& str ) const;

      GridType globalGrid;
      PointType localOrigin;
      CoordinatesType localBegin;
      CoordinatesType localSize;
      CoordinatesType localGridSize;
      CoordinatesType overlap;
      //CoordinatesType globalDimensions;
      CoordinatesType globalBegin;
      PointType spaceSteps;
      
      SubdomainOverlapsType lowerOverlap, upperOverlap;

      CoordinatesType domainDecomposition;
      CoordinatesType subdomainCoordinates;   

      int neighbors[ getNeighborsCount() ];
      
      int periodicNeighbors[ getNeighborsCount() ];

      IndexType Dimensions;        
      bool distributed;
        
      int rank;
      int nproc;

      bool isSet;

      //aka MPI-communicator 
      void * communicationGroup;

};

} // namespace DistributedMeshes
} // namespace Meshes
} // namespace TNL

#include <TNL/Meshes/DistributedMeshes/DistributedGrid_Base.hpp>
