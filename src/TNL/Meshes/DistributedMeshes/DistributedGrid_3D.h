/***************************************************************************
                          DistributedGrid_3D.h  -  description
                             -------------------
    begin                : January 15, 2018
    copyright            : (C) 2018 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Config/ConfigDescription.h>
#include <TNL/Meshes/Grid.h>

namespace TNL {
namespace Meshes { 
namespace DistributedMeshes {

enum Directions3D { West = 0 , East = 1 , North = 2, South=3, Top =4, Bottom=5, 
                  NorthWest=6, NorthEast=7, SouthWest=8, SouthEast=9,
                  BottomWest=10,BottomEast=11,BottomNorth=12,BottomSouth=13,
                  TopWest=14,TopEast=15,TopNorth=16,TopSouth=17,
                  BottomNorthWest=18,BottomNorthEast=19,BottomSouthWest=20,BottomSouthEast=21,
                  TopNorthWest=22,TopNorthEast=23,TopSouthWest=24,TopSouthEast=25
                  };


template< typename RealType, typename Device, typename Index >
class DistributedMesh<Grid< 3, RealType, Device, Index >>
{

    public:

      typedef Index IndexType;
      typedef Grid< 3, RealType, Device, IndexType > GridType;
      typedef typename GridType::PointType PointType;
      typedef Containers::StaticVector< 3, IndexType > CoordinatesType;

      static constexpr int getMeshDimension() { return 3; };    

      DistributedMesh();
      
      static void configSetup( Config::ConfigDescription& config );
      
      bool setup( const Config::ParameterContainer& parameters,
                  const String& prefix );
      
      void setDomainDecomposition( const CoordinatesType& domainDecomposition );
      
      const CoordinatesType& getDomainDecomposition() const;
            
      template< int EntityDimension >
      IndexType getEntitiesCount() const;

      template< typename Entity >
      IndexType getEntitiesCount() const;      

      template< typename CommunicatorType > 
      void setGlobalGrid( const GridType& globalGrid,
                          const CoordinatesType& overlap );
       
      void setupGrid( GridType& grid);
       
      String printProcessCoords() const;

      String printProcessDistr() const;

      bool isDistributed() const;
       
      const CoordinatesType& getOverlap() const;
       
      const int* getNeighbors() const;
       
      const CoordinatesType& getLocalSize() const;
       
      const CoordinatesType& getLocalGridSize() const;
       
      const CoordinatesType& getLocalBegin() const;

      //number of elements of global grid
      const CoordinatesType& getGlobalSize() const;

      //coordinates of begin of local subdomain without overlaps in global grid
      const CoordinatesType& getGlobalBegin() const;
       
      void writeProlog( Logger& logger );

   private:

      int getRankOfProcCoord(int x, int y, int z) const;
        
      GridType globalGrid;
      
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
        
      CoordinatesType domainDecomposition;
      CoordinatesType subdomainCoordinates;
      int numberOfLarger[3];
        
      int neighbors[26];

      bool isSet;
};

} // namespace DistributedMeshes
} // namespace Meshes
} // namespace TNL

#include <TNL/Meshes/DistributedMeshes/DistributedGrid_3D.hpp>
