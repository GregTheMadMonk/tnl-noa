/***************************************************************************
                          Grid3D.h  -  description
                             -------------------
    begin                : Feb 13, 2015
    copyright            : (C) 2015 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Logger.h>
#include <TNL/Meshes/Grid.h>
#include <TNL/Meshes/GridDetails/GridEntityTopology.h>
#include <TNL/Meshes/GridDetails/GridEntityGetter.h>
#include <TNL/Meshes/GridDetails/NeighborGridEntityGetter.h>
#include <TNL/Meshes/GridEntity.h>
#include <TNL/Meshes/GridEntityConfig.h>

namespace TNL {
namespace Meshes {

template< typename Real,
          typename Device,
          typename Index >
class Grid< 3, Real, Device, Index > : public Object
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index GlobalIndexType;
   typedef Containers::StaticVector< 3, Real > PointType;
   typedef Containers::StaticVector< 3, Index > CoordinatesType;
   typedef Grid< 3, Real, Devices::Host, Index > HostType;
   typedef Grid< 3, Real, Devices::Cuda, Index > CudaType;
   typedef Grid< 3, Real, Device, Index > ThisType;

   typedef DistributedMeshes::DistributedMesh <ThisType> DistributedMeshType;
 
   // TODO: deprecated and to be removed (GlobalIndexType shall be used instead)
   typedef Index IndexType;

   static constexpr int getMeshDimension() { return 3; };

   template< int EntityDimension,
             typename Config = GridEntityCrossStencilStorage< 1 > >
   using EntityType = GridEntity< ThisType, EntityDimension, Config >;

   typedef EntityType< getMeshDimension(), GridEntityCrossStencilStorage< 1 > > Cell;
   typedef EntityType< getMeshDimension() - 1 > Face;
   typedef EntityType< 1 > Edge;
   typedef EntityType< 0 > Vertex;

   Grid();

   static String getType();

   String getTypeVirtual() const;

   static String getSerializationType();

   virtual String getSerializationTypeVirtual() const;

   void setDimensions( const Index xSize, const Index ySize, const Index zSize );

   void setDimensions( const CoordinatesType& );

   __cuda_callable__
   const CoordinatesType& getDimensions() const;

   void setDomain( const PointType& origin,
                   const PointType& proportions );
   
   void setOrigin( const PointType& origin);
   
   __cuda_callable__
   inline const PointType& getOrigin() const;

   __cuda_callable__
   inline const PointType& getProportions() const;


   template< int EntityDimension >
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
   inline const PointType& getSpaceSteps() const;

   inline void setSpaceSteps(const PointType& steps);
   
   void SetDistMesh(DistributedMeshType * distGrid);
   
   DistributedMeshType * GetDistMesh(void) const;
   
   template< int xPow, int yPow, int zPow >
   __cuda_callable__
   const RealType& getSpaceStepsProducts() const;

   __cuda_callable__
   inline const RealType& getCellMeasure() const;


   __cuda_callable__
   RealType getSmallestSpaceStep() const;

   template< typename GridFunction >
   typename GridFunction::RealType getAbsMax( const GridFunction& f ) const;

   template< typename GridFunction >
   typename GridFunction::RealType getLpNorm( const GridFunction& f,
                                              const typename GridFunction::RealType& p ) const;

   template< typename GridFunction >
   typename GridFunction::RealType getDifferenceAbsMax( const GridFunction& f1,
                                                        const GridFunction& f2 ) const;

   template< typename GridFunction >
   typename GridFunction::RealType getDifferenceLpNorm( const GridFunction& f1,
                                                        const GridFunction& f2,
                                                        const typename GridFunction::RealType& p ) const;

   //! Method for saving the object to a file as a binary data
   bool save( File& file ) const;

   //! Method for restoring the object from a file
   bool load( File& file );

   bool save( const String& fileName ) const;

   bool load( const String& fileName );

   void writeProlog( Logger& logger ) const;

   protected:

   void computeProportions();
       
   void computeSpaceStepPowers();    
       
   void computeSpaceSteps();

   CoordinatesType dimensions;

   IndexType numberOfCells,
          numberOfNxFaces, numberOfNyFaces, numberOfNzFaces, numberOfNxAndNyFaces, numberOfFaces,
          numberOfDxEdges, numberOfDyEdges, numberOfDzEdges, numberOfDxAndDyEdges, numberOfEdges,
          numberOfVertices;

   PointType origin, proportions;

   IndexType cellZNeighborsStep;

   PointType spaceSteps;

   RealType spaceStepsProducts[ 5 ][ 5 ][ 5 ];
   
   DistributedMeshType *distGrid;

   template< typename, typename, int >
   friend class GridEntityGetter;

   template< typename, int, typename >
   friend class NeighborGridEntityGetter;
};

} // namespace Meshes
} // namespace TNL

#include <TNL/Meshes/GridDetails/Grid3D_impl.h>
