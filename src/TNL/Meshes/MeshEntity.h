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
#include <TNL/Meshes/MeshDetails/layers/MeshEntityStorageRebinder.h>

namespace TNL {
namespace Meshes {

template< typename MeshConfig >
class MeshInitializer;

template< typename MeshConfig,
          typename Device,
          typename EntityTopology_ >
class MeshEntity
   : protected MeshSubentityAccess< MeshConfig, EntityTopology_ >,
     protected MeshSuperentityAccess< MeshConfig, EntityTopology_ >,
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

      static String getType();

      String getTypeVirtual() const;

      bool save( File& file ) const;

      bool load( File& file );

      void print( std::ostream& str ) const;

      bool operator==( const MeshEntity& entity ) const;

      bool operator!=( const MeshEntity& entity ) const;

      static constexpr int getEntityDimension();

      /****
       * Subentities
       */
      using MeshSubentityAccess< MeshConfig, EntityTopology_ >::getSubentitiesCount;
      using MeshSubentityAccess< MeshConfig, EntityTopology_ >::getSubentityIndex;
      using MeshSubentityAccess< MeshConfig, EntityTopology_ >::getSubentityOrientation;

      /****
       * Superentities
       */
      using MeshSuperentityAccess< MeshConfig, EntityTopology_ >::getSuperentitiesCount;
      using MeshSuperentityAccess< MeshConfig, EntityTopology_ >::getSuperentityIndex;

      /****
       * Vertices
       */
      static constexpr LocalIndexType getVerticesCount();

      GlobalIndexType getVertexIndex( const LocalIndexType localIndex ) const;

   protected:
      /****
       * Methods for the mesh initialization
       */
      using MeshSubentityAccess< MeshConfig, EntityTopology_ >::bindSubentitiesStorageNetwork;
      using MeshSubentityAccess< MeshConfig, EntityTopology_ >::setSubentityIndex;
      using MeshSubentityAccess< MeshConfig, EntityTopology_ >::subentityOrientationsArray;

      using MeshSuperentityAccess< MeshConfig, EntityTopology_ >::bindSuperentitiesStorageNetwork;
      using MeshSuperentityAccess< MeshConfig, EntityTopology_ >::setNumberOfSuperentities;
      using MeshSuperentityAccess< MeshConfig, EntityTopology_ >::setSuperentityIndex;

   friend MeshInitializer< MeshConfig >;

   template< typename Mesh, typename DimensionTag, typename SuperdimensionTag >
   friend struct MeshEntityStorageRebinderWorker;
};

/****
 * Vertex entity specialization
 */
template< typename MeshConfig, typename Device >
class MeshEntity< MeshConfig, Device, MeshVertexTopology >
   : protected MeshSuperentityAccess< MeshConfig, MeshVertexTopology >,
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

      static String getType();

      String getTypeVirtual() const;

      bool save( File& file ) const;

      bool load( File& file );

      void print( std::ostream& str ) const;

      bool operator==( const MeshEntity& entity ) const;

      bool operator!=( const MeshEntity& entity ) const;

      static constexpr int getEntityDimension();

      /****
       * Superentities
       */
      using MeshSuperentityAccess< MeshConfig, MeshVertexTopology >::getSuperentitiesCount;
      using MeshSuperentityAccess< MeshConfig, MeshVertexTopology >::getSuperentityIndex;

      /****
       * Points
       */
      PointType getPoint() const;

      void setPoint( const PointType& point );

   protected:
      using MeshSuperentityAccess< MeshConfig, MeshVertexTopology >::bindSuperentitiesStorageNetwork;
      using MeshSuperentityAccess< MeshConfig, MeshVertexTopology >::setNumberOfSuperentities;
      using MeshSuperentityAccess< MeshConfig, MeshVertexTopology >::setSuperentityIndex;

      PointType point;

   friend MeshInitializer< MeshConfig >;

   template< typename Mesh, typename DimensionTag, typename SuperdimensionTag >
   friend struct MeshEntityStorageRebinderWorker;
};

template< typename MeshConfig,
          typename Device,
          typename EntityTopology >
std::ostream& operator<<( std::ostream& str, const MeshEntity< MeshConfig, Device, EntityTopology >& entity );

} // namespace Meshes
} // namespace TNL

#include <TNL/Meshes/MeshDetails/MeshEntity_impl.h>
