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

template< typename RealType, typename Device, typename Index >
class DistributedMesh< Grid< 2, RealType, Device, Index > >
{
   public:

      typedef Index IndexType;
      typedef Grid< 2, RealType, Device, IndexType > GridType;
      typedef typename GridType::PointType PointType;
      typedef Containers::StaticVector< 2, IndexType > CoordinatesType;

      static constexpr int getMeshDimension() { return 2; };
    
     
   public:
     
      DistributedMesh();
      
      void setDomainDecomposition( const CoordinatesType& domainDecomposition );
      
      const CoordinatesType& getDomainDecomposition() const;
      
      template< int EntityDimension >
      IndexType getEntitiesCount() const;

      template< typename Entity >
      IndexType getEntitiesCount() const;            

      bool setup( const Config::ParameterContainer& parameters,
                  const String& prefix );
      
      template< typename CommunicatorType >
      void setGlobalGrid( const GridType &globalGrid,
                          const CoordinatesType& overlap );
       
      void setupGrid( GridType& grid);
       
      String printProcessCoords() const;

      String printProcessDistr() const;
       
      bool isDistributed() const;
       
      const CoordinatesType& getOverlap() const;
       
      const int* getNeighbors() const;
       
      const CoordinatesType& getLocalSize() const;

      //number of elements of global grid
      const CoordinatesType& getGlobalSize() const;

      //coordinates of begin of local subdomain without overlaps in global grid
      const CoordinatesType& getGlobalBegin() const;

      const CoordinatesType& getLocalGridSize() const;
       
      const CoordinatesType& getLocalBegin() const;
       
      void writeProlog( Logger& logger ) const;
               
   private : 
       
      int getRankOfProcCoord(int x, int y) const;
        
      GridType globalGrid;
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
        
      CoordinatesType domainDecomposition;
      CoordinatesType subdomainCoordinates;
      int numberOfLarger[2];
        
      int neighbors[8];
      bool isSet;
};

} // namespace DistributedMeshes
} // namespace Meshes
} // namespace TNL

#include <TNL/Meshes/DistributedMeshes/DistributedGrid_2D.hpp>
