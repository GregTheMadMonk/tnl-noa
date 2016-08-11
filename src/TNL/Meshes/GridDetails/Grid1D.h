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
#include <TNL/Meshes/GridDetails/NeighbourGridEntityGetter.h>
#include <TNL/Meshes/GridEntity.h>
#include <TNL/Meshes/GridEntityConfig.h>

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
   typedef Index IndexType;
   typedef Containers::StaticVector< 1, Real > VertexType;
   typedef Containers::StaticVector< 1, Index > CoordinatesType;
   typedef Grid< 1, Real, Devices::Host, Index > HostType;
   typedef Grid< 1, Real, Devices::Cuda, Index > CudaType;
   typedef Grid< 1, Real, Device, Index > ThisType;
 
   static const int meshDimensions = 1;
 
   template< int EntityDimensions,
             typename Config = GridEntityCrossStencilStorage< 1 > >
   using MeshEntity = GridEntity< ThisType, EntityDimensions, Config >;
 
   typedef MeshEntity< meshDimensions, GridEntityCrossStencilStorage< 1 > > Cell;
   typedef MeshEntity< 0 > Face;
   typedef MeshEntity< 0 > Vertex;

   static constexpr int getMeshDimensions() { return meshDimensions; };
 
   Grid();

   static String getType();

   String getTypeVirtual() const;

   static String getSerializationType();

   virtual String getSerializationTypeVirtual() const;

   void setDimensions( const Index xSize );

   void setDimensions( const CoordinatesType& dimensions );

   __cuda_callable__ inline
   const CoordinatesType& getDimensions() const;

   void setDomain( const VertexType& origin,
                   const VertexType& proportions );

   __cuda_callable__
   inline const VertexType& getOrigin() const;

   __cuda_callable__
   inline const VertexType& getProportions() const;
 
   template< typename EntityType >
   __cuda_callable__
   inline IndexType getEntitiesCount() const;
 
   template< typename EntityType >
   __cuda_callable__
   inline EntityType getEntity( const IndexType& entityIndex ) const;
 
   template< typename EntityType >
   __cuda_callable__
   inline Index getEntityIndex( const EntityType& entity ) const;
 
   template< typename EntityType >
   __cuda_callable__
   RealType getEntityMeasure( const EntityType& entity ) const;
 
   __cuda_callable__
   RealType getCellMeasure() const;
 
   __cuda_callable__
   inline VertexType getSpaceSteps() const;

   template< int xPow >
   __cuda_callable__
   inline const RealType& getSpaceStepsProducts() const;
 
   __cuda_callable__
   inline RealType getSmallestSpaceStep() const;


   template< typename GridFunction >
   typename GridFunction::RealType getDifferenceAbsMax( const GridFunction& f1,
                                                        const GridFunction& f2 ) const;

   template< typename GridFunction >
   typename GridFunction::RealType getDifferenceLpNorm( const GridFunction& f1,
                                                        const GridFunction& f2,
                                                        const typename GridFunction::RealType& p ) const;

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

   bool writeMesh( const String& fileName,
                   const String& format ) const;

   template< typename MeshFunction >
   bool write( const MeshFunction& function,
               const String& fileName,
               const String& format ) const;

   void writeProlog( Logger& logger );

   protected:

   void computeSpaceSteps();

   CoordinatesType dimensions;
 
   IndexType numberOfCells, numberOfVertices;

   VertexType origin, proportions;
 
   VertexType spaceSteps;
 
   RealType spaceStepsProducts[ 5 ];
};

} // namespace Meshes
} // namespace TNL

#include <TNL/Meshes/GridDetails/Grid1D_impl.h>
