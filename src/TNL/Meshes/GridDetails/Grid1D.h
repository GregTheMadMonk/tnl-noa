/***************************************************************************
                          Grid1D.h  -  description
                             -------------------
    begin                : Feb 13, 2015
    copyright            : (C) 2015 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Meshes/Grid.h>
#include <TNL/Logger.h>
#include <TNL/Meshes/GridDetails/GridEntityTopology.h>
#include <TNL/Meshes/GridDetails/GridEntityGetter.h>
#include <TNL/Meshes/GridDetails/NeighborGridEntityGetter.h>
#include <TNL/Meshes/GridEntity.h>
#include <TNL/Meshes/GridEntityConfig.h>
#include <TNL/Meshes/DistributedMeshes/DistributedMesh.h>

namespace TNL {
namespace Meshes {

template< typename Real,
          typename Device,
          typename Index >
class Grid< 1, Real, Device, Index > : public Object
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index GlobalIndexType;
   typedef Containers::StaticVector< 1, Real > PointType;
   typedef Containers::StaticVector< 1, Index > CoordinatesType;
   typedef Grid< 1, Real, Devices::Host, Index > HostType;
   typedef Grid< 1, Real, Devices::Cuda, Index > CudaType;
   typedef Grid< 1, Real, Device, Index > ThisType;

   typedef DistributedMeshes::DistributedMesh <ThisType> DistributedMeshType;

   // TODO: deprecated and to be removed (GlobalIndexType shall be used instead)
   typedef Index IndexType;
 
   static constexpr int getMeshDimension() { return 1; };
 
   template< int EntityDimension,
             typename Config = GridEntityCrossStencilStorage< 1 > >
   using EntityType = GridEntity< ThisType, EntityDimension, Config >;
 
   typedef EntityType< getMeshDimension(), GridEntityCrossStencilStorage< 1 > > Cell;
   typedef EntityType< 0 > Face;
   typedef EntityType< 0 > Vertex;

   Grid();

   static String getType();

   String getTypeVirtual() const;

   static String getSerializationType();

   virtual String getSerializationTypeVirtual() const;

   void setDimensions( const Index xSize );

   void setDimensions( const CoordinatesType& dimensions );

   __cuda_callable__
   const CoordinatesType& getDimensions() const;
 
   void setOrigin( const PointType& origin);

   void setDomain( const PointType& origin,
                   const PointType& proportions );


   __cuda_callable__
   inline const PointType& getOrigin() const;

   __cuda_callable__
   inline const PointType& getProportions() const;
 

   template< int EntiytDimension >
   __cuda_callable__
   IndexType getEntitiesCount() const;

   template< typename Entity >
   __cuda_callable__
   IndexType getEntitiesCount() const;
   
   template< typename Entity >
   __cuda_callable__
   inline Entity getEntity( const IndexType& entityIndex ) const;
 
   template< typename Entity >
   __cuda_callable__
   inline Index getEntityIndex( const Entity& entity ) const;
 
   __cuda_callable__
   
   inline void setSpaceSteps(const PointType& steps);

   inline const PointType& getSpaceSteps() const;


   template< int xPow >
   __cuda_callable__
   const RealType& getSpaceStepsProducts() const;
   
   __cuda_callable__
   inline const RealType& getCellMeasure() const;
 
   __cuda_callable__
   inline RealType getSmallestSpaceStep() const;


   template< typename GridFunction >
   typename GridFunction::RealType getDifferenceAbsMax( const GridFunction& f1,
                                                        const GridFunction& f2 ) const;

   template< typename GridFunction >
   typename GridFunction::RealType getDifferenceLpNorm( const GridFunction& f1,
                                                        const GridFunction& f2,
                                                        const typename GridFunction::RealType& p ) const;
   
   void SetDistMesh(DistributedMeshType * distMesh);
   
   DistributedMeshType * getDistributedMesh() const;

   /****
    *  Method for saving the object to a file as a binary data
    */
   bool save( File& file ) const;

   /****
    *  Method for restoring the object from a file
    */
   bool load( File& file );

   bool save( const String& fileName ) const;

   bool load( const String& fileName );

   void writeProlog( Logger& logger ) const;

   protected:

   void computeProportions();
       
   void computeSpaceStepPowers();

   void computeSpaceSteps();

   CoordinatesType dimensions;
 
   IndexType numberOfCells, numberOfVertices;

   PointType origin, proportions;
 
   PointType spaceSteps;
 
   RealType spaceStepsProducts[ 5 ];
   
   DistributedMeshType *distGrid;
};

} // namespace Meshes
} // namespace TNL

#include <TNL/Meshes/GridDetails/Grid1D_impl.h>
