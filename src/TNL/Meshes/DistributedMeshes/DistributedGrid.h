/***************************************************************************
                          DistributedGrid.h  -  description
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
#include <TNL/Meshes/DistributedMeshes/DistributedMesh.h>

namespace TNL {
namespace Meshes {
namespace DistributedMeshes {

template< int Dimension,
          typename Real,
          typename Device,
          typename Index >
class DistributedMesh< Grid< Dimension, Real, Device, Index > >
{
   public:
      using RealType = Real;
      using DeviceType = Device;
      using IndexType = Index;
      using GridType = Grid< Dimension, Real, Device, IndexType >;
      using PointType = typename GridType::PointType;
      using CoordinatesType = Containers::StaticVector< Dimension, IndexType >;
      using SubdomainOverlapsType = Containers::StaticVector< Dimension, IndexType >;

      static constexpr int getMeshDimension() { return Dimension; };

      static constexpr int getNeighborsCount() { return Directions::i3pow(Dimension)-1; }

      DistributedMesh();

      ~DistributedMesh();

      void setDomainDecomposition( const CoordinatesType& domainDecomposition );

      const CoordinatesType& getDomainDecomposition() const;

      void setGlobalGrid( const GridType& globalGrid );

      const GridType& getGlobalGrid() const;

      void setOverlaps( const SubdomainOverlapsType& lower,
                        const SubdomainOverlapsType& upper);

      void setupGrid( GridType& grid);

      bool isDistributed() const;

      bool isBoundarySubdomain() const;

      // TODO: replace it with getLowerOverlap() and getUpperOverlap()
      // It is still being used in cuts set-up
      const CoordinatesType& getOverlap() const { return this->overlap;};

      //currently used overlaps at this subdomain
      const SubdomainOverlapsType& getLowerOverlap() const;

      const SubdomainOverlapsType& getUpperOverlap() const;

      //number of elements of local sub domain WITHOUT overlap
      // TODO: getSubdomainDimensions
      const CoordinatesType& getLocalSize() const;

      // TODO: delete
      //Dimensions of global grid
      const CoordinatesType& getGlobalSize() const;

      //coordinates of begin of local subdomain without overlaps in global grid
      const CoordinatesType& getGlobalBegin() const;

      //number of elements of local sub domain WITH overlap
      // TODO: replace with localGrid
      const CoordinatesType& getLocalGridSize() const;

      //coordinates of begin of local subdomain without overlaps in local grid
      const CoordinatesType& getLocalBegin() const;

      const CoordinatesType& getSubdomainCoordinates() const;

      const PointType& getLocalOrigin() const;
      const PointType& getSpaceSteps() const;

      //aka MPI-communcicator
      void setCommunicationGroup(MPI_Comm group);
      MPI_Comm getCommunicationGroup() const;

      template< int EntityDimension >
      IndexType getEntitiesCount() const;

      template< typename Entity >
      IndexType getEntitiesCount() const;

      const int* getNeighbors() const;

      const int* getPeriodicNeighbors() const;

      template<typename DistributedGridType>
      bool SetupByCut(DistributedGridType &inputDistributedGrid,
                 Containers::StaticVector<Dimension, int> savedDimensions,
                 Containers::StaticVector<DistributedGridType::getMeshDimension()-Dimension,int> reducedDimensions,
                 Containers::StaticVector<DistributedGridType::getMeshDimension()-Dimension,IndexType> fixedIndexs);

      int getRankOfProcCoord(const CoordinatesType &nodeCoordinates) const;

      String printProcessCoords() const;

      String printProcessDistr() const;

      void writeProlog( Logger& logger );

   public:

      bool isThereNeighbor(const CoordinatesType &direction) const;

      void setupNeighbors();

      void print( std::ostream& str ) const;

      GridType globalGrid;
      PointType localOrigin;
      CoordinatesType localBegin;
      CoordinatesType localSize;
      CoordinatesType localGridSize;
      CoordinatesType overlap;
      CoordinatesType globalBegin;
      PointType spaceSteps;

      SubdomainOverlapsType lowerOverlap, upperOverlap;

      CoordinatesType domainDecomposition;
      CoordinatesType subdomainCoordinates;

      // TODO: static arrays
      int neighbors[ getNeighborsCount() ];
      int periodicNeighbors[ getNeighborsCount() ];

      IndexType Dimensions;
      bool distributed;

      int rank;
      int nproc;

      bool isSet;

      //aka MPI-communicator
      MPI_Comm group;
};

} // namespace DistributedMeshes
} // namespace Meshes
} // namespace TNL

#include <TNL/Meshes/DistributedMeshes/DistributedGrid.hpp>
