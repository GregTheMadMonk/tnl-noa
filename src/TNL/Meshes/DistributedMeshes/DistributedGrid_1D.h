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

template< typename RealType, typename Device, typename Index >     
class DistributedMesh< Grid< 1, RealType, Device, Index > >
{

    public:
    
      typedef Index IndexType;
      typedef Grid< 1, RealType, Device, IndexType > GridType;
      typedef typename GridType::PointType PointType;
      typedef Containers::StaticVector< 1, IndexType > CoordinatesType;

      static constexpr int getMeshDimension() { return 1; };    

      DistributedMesh();

      bool setup( const Config::ParameterContainer& parameters,
                  const String& prefix );
      
      void setDomainDecomposition( const CoordinatesType& domainDecomposition );
      
      const CoordinatesType& getDomainDecomposition() const;
      
      template< int EntityDimension >
      IndexType getEntitiesCount() const;

      template< typename Entity >
      IndexType getEntitiesCount() const;      
      
      template<typename CommunicatorType>
      void setGlobalGrid( const GridType& globalGrid, const CoordinatesType& overlap );
       
      void setupGrid( GridType& grid );
       
      String printProcessCoords() const;

      String printProcessDistr() const;

      bool isDistributed() const;
       
      int getLeft() const;
       
      int getRight() const;
       
      const CoordinatesType& getOverlap() const;

      //number of elements of local sub domain WITHOUT overlap
      const CoordinatesType& getLocalSize() const;

      //dimensions of global grid
      const CoordinatesType& getGlobalSize() const;

      //coordinates of begin of local subdomain without overlaps in global grid
      const CoordinatesType& getGlobalBegin() const;

      //number of elements of local sub domain WITH overlap
      const CoordinatesType& getLocalGridSize() const;
       
      //coordinates of begin of local subdomain without overlaps in local grid       
      const CoordinatesType& getLocalBegin() const;
      
      void writeProlog( Logger& logger ) const;       
       
   private : 

      GridType globalGrid;
      PointType localOrigin;
      CoordinatesType localBegin;
      CoordinatesType localSize;
      CoordinatesType localGridSize;
      CoordinatesType overlap;
      CoordinatesType globalDimensions;
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

#include <TNL/Meshes/DistributedMeshes/DistributedGrid_1D.hpp>