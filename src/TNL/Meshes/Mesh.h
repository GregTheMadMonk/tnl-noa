/***************************************************************************
                          Mesh.h  -  description
                             -------------------
    begin                : Feb 16, 2014
    copyright            : (C) 2014 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

/***
 * Authors:
 * Oberhuber Tomas, tomas.oberhuber@fjfi.cvut.cz
 * Zabka Vitezslav, zabkav@gmail.com
 */

#pragma once

#include <ostream>
#include <TNL/Object.h>
#include <TNL/Logger.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Meshes/MeshDetails/ConfigValidator.h>
#include <TNL/Meshes/MeshDetails/traits/MeshTraits.h>
#include <TNL/Meshes/MeshDetails/layers/StorageLayer.h>
#include <TNL/Meshes/MeshDetails/layers/EntityTags/LayerFamily.h>

#include <TNL/Meshes/DistributedMeshes/DistributedMesh.h>

namespace TNL {
/**
 * \brief Namespace for numerical meshes and related objects.
 */
namespace Meshes {

template< typename MeshConfig,
          typename Device,
          typename EntityTopology_ >
class MeshEntity;

template< typename MeshConfig > class Initializer;
template< typename Mesh > class EntityStorageRebinder;
template< typename Mesh, int Dimension > struct IndexPermutationApplier;


template< typename MeshConfig, typename Device, typename MeshType >
class MeshInitializableBase
{
   public:
      using MeshTraitsType = MeshTraits< MeshConfig, Device >;

      // The points and cellSeeds arrays will be reset when not needed to save memory.
      void init( typename MeshTraitsType::PointArrayType& points,
                 typename MeshTraitsType::CellSeedArrayType& cellSeeds );
};

// The mesh cannot be initialized on CUDA GPU, so this specialization is empty.
template< typename MeshConfig, typename MeshType >
class MeshInitializableBase< MeshConfig, Devices::Cuda, MeshType >
{
};


template< typename MeshConfig,
          typename Device = Devices::Host >
class Mesh
   : public Object,
     public ConfigValidator< MeshConfig >,
     public MeshInitializableBase< MeshConfig, Device, Mesh< MeshConfig, Device > >,
     public StorageLayerFamily< MeshConfig, Device >,
     public EntityTags::LayerFamily< MeshConfig, Device, Mesh< MeshConfig, Device > >
{
      using StorageBaseType = StorageLayerFamily< MeshConfig, Device >;
      using EntityTagsLayerFamily = EntityTags::LayerFamily< MeshConfig, Device, Mesh >;

   public:
      using Config          = MeshConfig;
      using MeshTraitsType  = MeshTraits< MeshConfig, Device >;
      using DeviceType      = typename MeshTraitsType::DeviceType;
      using GlobalIndexType = typename MeshTraitsType::GlobalIndexType;
      using LocalIndexType  = typename MeshTraitsType::LocalIndexType;
      using PointType       = typename MeshTraitsType::PointType;
      using RealType        = typename PointType::RealType;
      using GlobalIndexArray = Containers::Array< GlobalIndexType, DeviceType, GlobalIndexType >;

      template< int Dimension >
      using EntityTraits = typename MeshTraitsType::template EntityTraits< Dimension >;

      template< int Dimension >
      using EntityType = typename EntityTraits< Dimension >::EntityType;

      // constructors
      Mesh() = default;

      Mesh( const Mesh& mesh );

      template< typename Device_ >
      Mesh( const Mesh< MeshConfig, Device_ >& mesh );

      Mesh& operator=( const Mesh& mesh );

      template< typename Device_ >
      Mesh& operator=( const Mesh< MeshConfig, Device_ >& mesh );


      static constexpr int getMeshDimension();

      // types of common entities
      using Cell = EntityType< getMeshDimension() >;
      using Face = EntityType< getMeshDimension() - 1 >;
      using Vertex = EntityType< 0 >;

      static String getSerializationType();

      virtual String getSerializationTypeVirtual() const;

      /**
       * Entities
       */
      template< int Dimension >
      __cuda_callable__
      GlobalIndexType getEntitiesCount() const;

      template< int Dimension >
      __cuda_callable__
      EntityType< Dimension > getEntity( const GlobalIndexType entityIndex ) const;

      // duplicated for compatibility with grids
      template< typename EntityType >
      __cuda_callable__
      GlobalIndexType getEntitiesCount() const;

      template< typename EntityType >
      __cuda_callable__
      EntityType getEntity( const GlobalIndexType entityIndex ) const;

      /**
       * Points
       */
      __cuda_callable__
      const PointType& getPoint( const GlobalIndexType vertexIndex ) const;

      __cuda_callable__
      PointType& getPoint( const GlobalIndexType vertexIndex );

      /**
       * Subentities
       */
      template< int EntityDimension, int SubentityDimension >
      __cuda_callable__
      constexpr LocalIndexType getSubentitiesCount( const GlobalIndexType entityIndex ) const;

      template< int EntityDimension, int SubentityDimension >
      __cuda_callable__
      GlobalIndexType getSubentityIndex( const GlobalIndexType entityIndex, const LocalIndexType subentityIndex ) const;

      /**
       * Superentities
       */
      template< int EntityDimension, int SuperentityDimension >
      __cuda_callable__
      LocalIndexType getSuperentitiesCount( const GlobalIndexType entityIndex ) const;

      template< int EntityDimension, int SuperentityDimension >
      __cuda_callable__
      GlobalIndexType getSuperentityIndex( const GlobalIndexType entityIndex, const LocalIndexType superentityIndex ) const;

      /**
       * Cell neighbors - access the dual graph
       */
      __cuda_callable__
      LocalIndexType getCellNeighborsCount( const GlobalIndexType cellIndex ) const;

      __cuda_callable__
      GlobalIndexType getCellNeighborIndex( const GlobalIndexType cellIndex, const LocalIndexType neighborIndex ) const;


      /*
       * The permutations follow the definition used in the Metis library: Let M
       * be the original mesh and M' the permuted mesh. Then entity with index i
       * in M' is the entity with index perm[i] in M and entity with index j in
       * M is the entity with index iperm[j] in M'.
       */
      template< int Dimension >
      void reorderEntities( const GlobalIndexArray& perm,
                            const GlobalIndexArray& iperm );


      void save( File& file ) const;

      void load( File& file );

      using Object::load;
      using Object::save;

      void print( std::ostream& str ) const;

      bool operator==( const Mesh& mesh ) const;

      bool operator!=( const Mesh& mesh ) const;

      void writeProlog( Logger& logger ) const;

      DistributedMeshes::DistributedMesh< Mesh<MeshConfig,Device> >* getDistributedMesh(void) const
      {
         return nullptr;
      }

   protected:
      // Methods for the mesh initializer
      using StorageBaseType::getPoints;
      using StorageBaseType::setEntitiesCount;
      using StorageBaseType::getSubentitiesMatrix;
      using StorageBaseType::getSuperentitiesMatrix;

      friend Initializer< MeshConfig >;

      template< typename Mesh, int Dimension >
      friend struct IndexPermutationApplier;
};

template< typename MeshConfig, typename Device >
std::ostream& operator<<( std::ostream& str, const Mesh< MeshConfig, Device >& mesh );

} // namespace Meshes
} // namespace TNL

#include <TNL/Meshes/MeshEntity.h>

#include <TNL/Meshes/Mesh.hpp>
