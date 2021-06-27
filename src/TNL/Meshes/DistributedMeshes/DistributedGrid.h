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
      using GlobalIndexType = Index;
      using GridType = Grid< Dimension, Real, Device, IndexType >;
      using PointType = typename GridType::PointType;
      using CoordinatesType = Containers::StaticVector< Dimension, IndexType >;
      using SubdomainOverlapsType = Containers::StaticVector< Dimension, IndexType >;

      static constexpr int getMeshDimension() { return Dimension; };

      static constexpr int getNeighborsCount() { return Directions::i3pow(Dimension)-1; }

      DistributedMesh() = default;

      ~DistributedMesh() = default;

      void setDomainDecomposition( const CoordinatesType& domainDecomposition );

      const CoordinatesType& getDomainDecomposition() const;

      void setGlobalGrid( const GridType& globalGrid );

      const GridType& getGlobalGrid() const;

      void setOverlaps( const SubdomainOverlapsType& lower,
                        const SubdomainOverlapsType& upper);

      // for compatibility with DistributedMesh
      void setGhostLevels( int levels );
      int getGhostLevels() const;

      bool isDistributed() const;

      bool isBoundarySubdomain() const;

      //currently used overlaps at this subdomain
      const SubdomainOverlapsType& getLowerOverlap() const;

      const SubdomainOverlapsType& getUpperOverlap() const;

      // returns the local grid WITH overlap
      const GridType& getLocalMesh() const;

      //number of elements of local sub domain WITHOUT overlap
      // TODO: getSubdomainDimensions
      const CoordinatesType& getLocalSize() const;

      // TODO: delete
      //Dimensions of global grid
      const CoordinatesType& getGlobalSize() const;

      //coordinates of begin of local subdomain without overlaps in global grid
      const CoordinatesType& getGlobalBegin() const;

      const CoordinatesType& getSubdomainCoordinates() const;

      //aka MPI-communcicator
      void setCommunicationGroup( MPI_Comm group );
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

      bool operator==( const DistributedMesh& other ) const;

      bool operator!=( const DistributedMesh& other ) const;

   public:

      bool isThereNeighbor(const CoordinatesType &direction) const;

      void setupNeighbors();

      GridType globalGrid, localGrid;
      CoordinatesType localSize = 0;
      CoordinatesType globalBegin = 0;

      SubdomainOverlapsType lowerOverlap = 0;
      SubdomainOverlapsType upperOverlap = 0;

      CoordinatesType domainDecomposition = 0;
      CoordinatesType subdomainCoordinates = 0;

      // TODO: static arrays
      int neighbors[ getNeighborsCount() ];
      int periodicNeighbors[ getNeighborsCount() ];

      bool distributed = false;

      bool isSet = false;

      //aka MPI-communicator
      MPI_Comm group = MPI::AllGroup();
};

template< int Dimension, typename Real, typename Device, typename Index >
std::ostream& operator<<( std::ostream& str, const DistributedMesh< Grid< Dimension, Real, Device, Index > >& grid );

} // namespace DistributedMeshes
} // namespace Meshes
} // namespace TNL

#include <TNL/Meshes/DistributedMeshes/DistributedGrid.hpp>
