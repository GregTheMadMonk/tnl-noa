/***************************************************************************
                          MeshEntity.h  -  description
                             -------------------
    begin                : Feb 11, 2014
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

#include <TNL/File.h>
#include <TNL/Meshes/MeshDetails/traits/MeshTraits.h>
#include <TNL/Meshes/Topologies/Vertex.h>
#include <TNL/Meshes/MeshDetails/MeshEntityIndex.h>
#include <TNL/Meshes/MeshDetails/EntityLayers/SubentityAccess.h>
#include <TNL/Meshes/MeshDetails/EntityLayers/SuperentityAccess.h>

namespace TNL {
namespace Meshes {

template< typename MeshConfig, typename Device > class Mesh;
template< typename MeshConfig > class Initializer;
template< typename Mesh > class EntityStorageRebinder;
template< typename Mesh, int Dimension > struct IndexPermutationApplier;

template< typename MeshConfig,
          typename Device,
          typename EntityTopology_ >
class MeshEntity
   : protected SubentityAccessLayerFamily< MeshConfig, Device, EntityTopology_ >,
     protected SuperentityAccessLayerFamily< MeshConfig, Device, EntityTopology_ >,
     public MeshEntityIndex< typename MeshConfig::IdType >
{
   static_assert( std::is_same< EntityTopology_, typename MeshTraits< MeshConfig, Device >::template EntityTraits< EntityTopology_::dimension >::EntityTopology >::value,
                  "Specified entity topology is not compatible with the MeshConfig." );

   public:
      using MeshTraitsType  = MeshTraits< MeshConfig, Device >;
      using DeviceType      = Device;
      using EntityTopology  = EntityTopology_;
      using GlobalIndexType = typename MeshTraitsType::GlobalIndexType;
      using LocalIndexType  = typename MeshTraitsType::LocalIndexType;

      template< int Subdimension >
      using SubentityTraits = typename MeshTraitsType::template SubentityTraits< EntityTopology, Subdimension >;

      template< int Superdimension >
      using SuperentityTraits = typename MeshTraitsType::template SuperentityTraits< EntityTopology, Superdimension >;

      // constructors
      MeshEntity() = default;

      __cuda_callable__
      MeshEntity( const MeshEntity& entity );

      template< typename Device_ >
      MeshEntity( const MeshEntity< MeshConfig, Device_, EntityTopology >& entity );

      __cuda_callable__
      MeshEntity& operator=( const MeshEntity& entity );

      template< typename Device_ >
      __cuda_callable__
      MeshEntity& operator=( const MeshEntity< MeshConfig, Device_, EntityTopology >& entity );


      static String getSerializationType();

      String getSerializationTypeVirtual() const;

      void save( File& file ) const;

      void load( File& file );

      void print( std::ostream& str ) const;

      __cuda_callable__
      bool operator==( const MeshEntity& entity ) const;

      __cuda_callable__
      bool operator!=( const MeshEntity& entity ) const;

      static constexpr int getEntityDimension();

      /****
       * Subentities
       */
      using SubentityAccessLayerFamily< MeshConfig, Device, EntityTopology_ >::getSubentitiesCount;
      using SubentityAccessLayerFamily< MeshConfig, Device, EntityTopology_ >::getSubentityIndex;
      using SubentityAccessLayerFamily< MeshConfig, Device, EntityTopology_ >::getSubentityOrientation;

      /****
       * Superentities
       */
      using SuperentityAccessLayerFamily< MeshConfig, Device, EntityTopology_ >::getSuperentitiesCount;
      using SuperentityAccessLayerFamily< MeshConfig, Device, EntityTopology_ >::getSuperentityIndex;

      /****
       * Vertices
       */
      static constexpr LocalIndexType getVerticesCount();

      GlobalIndexType getVertexIndex( const LocalIndexType localIndex ) const;

   protected:
      /****
       * Methods for the mesh initialization
       */
      using SubentityAccessLayerFamily< MeshConfig, Device, EntityTopology_ >::bindSubentitiesStorageNetwork;
      using SubentityAccessLayerFamily< MeshConfig, Device, EntityTopology_ >::setSubentityIndex;
      using SubentityAccessLayerFamily< MeshConfig, Device, EntityTopology_ >::subentityOrientationsArray;

      using SuperentityAccessLayerFamily< MeshConfig, Device, EntityTopology_ >::bindSuperentitiesStorageNetwork;
      using SuperentityAccessLayerFamily< MeshConfig, Device, EntityTopology_ >::setNumberOfSuperentities;
      using SuperentityAccessLayerFamily< MeshConfig, Device, EntityTopology_ >::setSuperentityIndex;

   friend Initializer< MeshConfig >;

   friend EntityStorageRebinder< Mesh< MeshConfig, DeviceType > >;

   template< typename Mesh, int Dimension >
   friend struct IndexPermutationApplier;
};

/****
 * Vertex entity specialization
 */
template< typename MeshConfig, typename Device >
class MeshEntity< MeshConfig, Device, Topologies::Vertex >
   : protected SuperentityAccessLayerFamily< MeshConfig, Device, Topologies::Vertex >,
     public MeshEntityIndex< typename MeshConfig::IdType >
{
   public:
      using MeshTraitsType  = MeshTraits< MeshConfig, Device >;
      using DeviceType      = Device;
      using EntityTopology  = Topologies::Vertex;
      using GlobalIndexType = typename MeshTraitsType::GlobalIndexType;
      using LocalIndexType  = typename MeshTraitsType::LocalIndexType;
      using PointType       = typename MeshTraitsType::PointType;

      template< int Superdimension >
      using SuperentityTraits = typename MeshTraitsType::template SuperentityTraits< EntityTopology, Superdimension >;

      // constructors
      MeshEntity() = default;

      __cuda_callable__
      MeshEntity( const MeshEntity& entity );

      template< typename Device_ >
      MeshEntity( const MeshEntity< MeshConfig, Device_, EntityTopology >& entity );

      __cuda_callable__
      MeshEntity& operator=( const MeshEntity& entity );

      template< typename Device_ >
      __cuda_callable__
      MeshEntity& operator=( const MeshEntity< MeshConfig, Device_, EntityTopology >& entity );


      static String getSerializationType();

      String getSerializationTypeVirtual() const;

      void save( File& file ) const;

      void load( File& file );

      void print( std::ostream& str ) const;

      __cuda_callable__
      bool operator==( const MeshEntity& entity ) const;

      __cuda_callable__
      bool operator!=( const MeshEntity& entity ) const;

      static constexpr int getEntityDimension();

      /****
       * Superentities
       */
      using SuperentityAccessLayerFamily< MeshConfig, Device, Topologies::Vertex >::getSuperentitiesCount;
      using SuperentityAccessLayerFamily< MeshConfig, Device, Topologies::Vertex >::getSuperentityIndex;

      /****
       * Points
       */
      __cuda_callable__
      PointType getPoint() const;

      __cuda_callable__
      void setPoint( const PointType& point );

   protected:
      using SuperentityAccessLayerFamily< MeshConfig, Device, Topologies::Vertex >::bindSuperentitiesStorageNetwork;
      using SuperentityAccessLayerFamily< MeshConfig, Device, Topologies::Vertex >::setNumberOfSuperentities;
      using SuperentityAccessLayerFamily< MeshConfig, Device, Topologies::Vertex >::setSuperentityIndex;

      PointType point;

   friend Initializer< MeshConfig >;

   friend EntityStorageRebinder< Mesh< MeshConfig, DeviceType > >;

   template< typename Mesh, int Dimension >
   friend struct IndexPermutationApplier;
};

template< typename MeshConfig,
          typename Device,
          typename EntityTopology >
std::ostream& operator<<( std::ostream& str, const MeshEntity< MeshConfig, Device, EntityTopology >& entity );

} // namespace Meshes
} // namespace TNL

#include <TNL/Meshes/MeshDetails/MeshEntity_impl.h>
