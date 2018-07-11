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

      DistributedGrid_Base();
    
      void setDomainDecomposition( const CoordinatesType& domainDecomposition );      
      const CoordinatesType& getDomainDecomposition() const;

      bool isDistributed() const;
           
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

      //aka MPI-communcicator  
      void setCommunicationGroup(void * group);
      void * getCommunicationGroup() const;

      template< int EntityDimension >
      IndexType getEntitiesCount() const;

      template< typename Entity >
      IndexType getEntitiesCount() const; 

   public: 

      GridType globalGrid;
      PointType localOrigin;
      CoordinatesType localBegin;
      CoordinatesType localSize;
      CoordinatesType localGridSize;
      CoordinatesType overlap;
      CoordinatesType globalDimensions;
      CoordinatesType globalBegin;
      PointType spaceSteps;

      CoordinatesType domainDecomposition;
      CoordinatesType subdomainCoordinates;    

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
