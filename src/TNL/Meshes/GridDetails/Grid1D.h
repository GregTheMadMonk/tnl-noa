/***************************************************************************
                          Grid1D.h  -  description
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

   /**
    * \brief Sets the number of dimensions.
    * \param xSize Number of dimensions.
    */
   void setDimensions( const Index xSize );

   /**
    * \brief Sets the number of dimensions.
    * \param xSize Number of dimensions.
    */
   void setDimensions( const CoordinatesType& dimensions );

   /**
    * \brief Returns number of dimensions of entities in this grid.
    */
   __cuda_callable__
   const CoordinatesType& getDimensions() const;

   /**
    * \brief Sets the origin.
    * \param origin Starting point of this grid.
    */
   void setOrigin( const PointType& origin);

   /**
    * \brief Sets the origin and proportions of this grid.
    * \param origin Point where this grid starts.
    * \param proportions Total length of this grid.
    */
   void setDomain( const PointType& origin,
                   const PointType& proportions );

   /**
    * \brief Gets the origin.
    * \param origin Starting point of this grid.
    */
   __cuda_callable__
   inline const PointType& getOrigin() const;

   /**
    * \brief Gets length of one entity of this grid.
    */
   __cuda_callable__
   inline const PointType& getProportions() const;

   /**
    * \brief Gets number of entities in this grid.
    * \tparam EntityDimension Integer specifying dimension of the entity.
    */
   template< int EntityDimension >
   __cuda_callable__
   IndexType getEntitiesCount() const;

   /**
    * \brief Gets number of entities in this grid.
    * \tparam Entity Type of the entity.
    */
   template< typename Entity >
   __cuda_callable__
   IndexType getEntitiesCount() const;

   /**
    * \brief Gets entity type using entity index.
    * \param entityIndex Index of entity.
    * \tparam Entity Type of the entity.
    */
   template< typename Entity >
   __cuda_callable__
   inline Entity getEntity( const IndexType& entityIndex ) const;

    /**
    * \brief Gets entity index using entity type.
    * \param entity Type of entity.
    * \tparam Entity Type of the entity.
    */
   template< typename Entity >
   __cuda_callable__
   inline Index getEntityIndex( const Entity& entity ) const;

   /**
    * \brief Gets length of one step.
    */
   __cuda_callable__
   inline const PointType& getSpaceSteps() const;

   /**
    * \brief Sets the length of steps.
    * \param steps Length of one step.
    */
   inline void setSpaceSteps(const PointType& steps);

   template< int xPow >
   __cuda_callable__
   const RealType& getSpaceStepsProducts() const;

   __cuda_callable__
   inline const RealType& getCellMeasure() const;

   /**
    * \brief Gets the smallest length of step out of all coordinates.
    */
   __cuda_callable__
   inline RealType getSmallestSpaceStep() const;


   template< typename GridFunction >
   typename GridFunction::RealType getDifferenceAbsMax( const GridFunction& f1,
                                                        const GridFunction& f2 ) const;

   template< typename GridFunction >
   typename GridFunction::RealType getDifferenceLpNorm( const GridFunction& f1,
                                                        const GridFunction& f2,
                                                        const typename GridFunction::RealType& p ) const;
   
   void setDistMesh(DistributedMeshType * distMesh);
   
   DistributedMeshType * getDistributedMesh() const;

   /****
    *  Method for saving the object to a file as a binary data
    */
   bool save( File& file ) const;

   /**
    * \brief Method for restoring the object from a file.
    */
   bool load( File& file );

   /**
    * \brief Method for saving the object to a file.
    */
   bool save( const String& fileName ) const;

   /**
    * \brief Method for restoring the object from a file.
    */
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
