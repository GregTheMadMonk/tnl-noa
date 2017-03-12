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
#include <TNL/Meshes/Topologies/MeshVertexTopology.h>
#include <TNL/Meshes/MeshDetails/MeshEntityIndex.h>
#include <TNL/Meshes/MeshDetails/layers/MeshSubentityAccess.h>
#include <TNL/Meshes/MeshDetails/layers/MeshSuperentityAccess.h>

namespace TNL {
namespace Meshes {

template< typename MeshConfig >
class MeshInitializer;
template< typename DimensionTag, typename SuperdimensionTag >
struct MeshEntityStorageRebinderWorker;
template< typename Mesh, int Dimension >
struct IndexPermutationApplier;

template< typename MeshConfig,
          typename Device,
          typename EntityTopology_ >
class MeshEntity
   : protected MeshSubentityAccess< MeshConfig, Device, EntityTopology_ >,
     protected MeshSuperentityAccess< MeshConfig, Device, EntityTopology_ >,
     public MeshEntityIndex< typename MeshConfig::IdType >
{
   static_assert( is_compatible_topology< typename MeshConfig::CellTopology, EntityTopology_ >::value,
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

      MeshEntity( const MeshEntity& entity );

      template< typename Device_ >
      MeshEntity( const MeshEntity< MeshConfig, Device_, EntityTopology >& entity );

      __cuda_callable__
      MeshEntity& operator=( const MeshEntity& entity );

      template< typename Device_ >
      __cuda_callable__
      MeshEntity& operator=( const MeshEntity< MeshConfig, Device_, EntityTopology >& entity );


      static String getType();

      String getTypeVirtual() const;

      bool save( File& file ) const;

      bool load( File& file );

      void print( std::ostream& str ) const;

      __cuda_callable__
      bool operator==( const MeshEntity& entity ) const;

      __cuda_callable__
      bool operator!=( const MeshEntity& entity ) const;

      static constexpr int getEntityDimension();

      /****
       * Subentities
       */
      using MeshSubentityAccess< MeshConfig, Device, EntityTopology_ >::getSubentitiesCount;
      using MeshSubentityAccess< MeshConfig, Device, EntityTopology_ >::getSubentityIndex;
      using MeshSubentityAccess< MeshConfig, Device, EntityTopology_ >::getSubentityOrientation;

      /****
       * Superentities
       */
      using MeshSuperentityAccess< MeshConfig, Device, EntityTopology_ >::getSuperentitiesCount;
      using MeshSuperentityAccess< MeshConfig, Device, EntityTopology_ >::getSuperentityIndex;

      /****
       * Vertices
       */
      static constexpr LocalIndexType getVerticesCount();

      GlobalIndexType getVertexIndex( const LocalIndexType localIndex ) const;

   protected:
      /****
       * Methods for the mesh initialization
       */
      using MeshSubentityAccess< MeshConfig, Device, EntityTopology_ >::bindSubentitiesStorageNetwork;
      using MeshSubentityAccess< MeshConfig, Device, EntityTopology_ >::setSubentityIndex;
      using MeshSubentityAccess< MeshConfig, Device, EntityTopology_ >::subentityOrientationsArray;

      using MeshSuperentityAccess< MeshConfig, Device, EntityTopology_ >::bindSuperentitiesStorageNetwork;
      using MeshSuperentityAccess< MeshConfig, Device, EntityTopology_ >::setNumberOfSuperentities;
      using MeshSuperentityAccess< MeshConfig, Device, EntityTopology_ >::setSuperentityIndex;

   friend MeshInitializer< MeshConfig >;

   template< typename DimensionTag, typename SuperdimensionTag >
   friend struct MeshEntityStorageRebinderWorker;

   template< typename Mesh, int Dimension >
   friend struct IndexPermutationApplier;
};

/****
 * Vertex entity specialization
 */
template< typename MeshConfig, typename Device >
class MeshEntity< MeshConfig, Device, MeshVertexTopology >
   : protected MeshSuperentityAccess< MeshConfig, Device, MeshVertexTopology >,
     public MeshEntityIndex< typename MeshConfig::IdType >
{
   public:
      using MeshTraitsType  = MeshTraits< MeshConfig, Device >;
      using DeviceType      = Device;
      using EntityTopology  = MeshVertexTopology;
      using GlobalIndexType = typename MeshTraitsType::GlobalIndexType;
      using LocalIndexType  = typename MeshTraitsType::LocalIndexType;
      using PointType       = typename MeshTraitsType::PointType;

      template< int Superdimension >
      using SuperentityTraits = typename MeshTraitsType::template SuperentityTraits< EntityTopology, Superdimension >;

      // constructors
      MeshEntity() = default;

      MeshEntity( const MeshEntity& entity );

      template< typename Device_ >
      MeshEntity( const MeshEntity< MeshConfig, Device_, EntityTopology >& entity );

      __cuda_callable__
      MeshEntity& operator=( const MeshEntity& entity );

      template< typename Device_ >
      __cuda_callable__
      MeshEntity& operator=( const MeshEntity< MeshConfig, Device_, EntityTopology >& entity );


      static String getType();

      String getTypeVirtual() const;

      bool save( File& file ) const;

      bool load( File& file );

      void print( std::ostream& str ) const;

      __cuda_callable__
      bool operator==( const MeshEntity& entity ) const;

      __cuda_callable__
      bool operator!=( const MeshEntity& entity ) const;

      static constexpr int getEntityDimension();

      /****
       * Superentities
       */
      using MeshSuperentityAccess< MeshConfig, Device, MeshVertexTopology >::getSuperentitiesCount;
      using MeshSuperentityAccess< MeshConfig, Device, MeshVertexTopology >::getSuperentityIndex;

      /****
       * Points
       */
      __cuda_callable__
      PointType getPoint() const;

      __cuda_callable__
      void setPoint( const PointType& point );

   protected:
      using MeshSuperentityAccess< MeshConfig, Device, MeshVertexTopology >::bindSuperentitiesStorageNetwork;
      using MeshSuperentityAccess< MeshConfig, Device, MeshVertexTopology >::setNumberOfSuperentities;
      using MeshSuperentityAccess< MeshConfig, Device, MeshVertexTopology >::setSuperentityIndex;

      PointType point;

   friend MeshInitializer< MeshConfig >;

   template< typename DimensionTag, typename SuperdimensionTag >
   friend struct MeshEntityStorageRebinderWorker;

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
