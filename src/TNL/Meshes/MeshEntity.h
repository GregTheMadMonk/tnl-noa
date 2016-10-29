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
#include <TNL/Meshes/MeshDetails/MeshEntityId.h>
#include <TNL/Meshes/MeshDetails/traits/MeshTraits.h>
#include <TNL/Meshes/MeshDimensionTag.h>
#include <TNL/Meshes/Topologies/MeshVertexTopology.h>
#include <TNL/Meshes/MeshDetails/layers/MeshSubentityAccess.h>
#include <TNL/Meshes/MeshDetails/layers/MeshSuperentityAccess.h>
#include <TNL/Meshes/MeshDetails/layers/MeshEntityStorageRebinder.h>
#include <TNL/Meshes/MeshDetails/initializer/MeshEntitySeed.h>

namespace TNL {
namespace Meshes {

template< typename MeshConfig >
class MeshInitializer;

template< typename MeshConfig,
          typename EntityTopology_ >
class MeshEntity
   : protected MeshSubentityAccess< MeshConfig, EntityTopology_ >,
     protected MeshSuperentityAccess< MeshConfig, EntityTopology_ >,
     public MeshEntityId< typename MeshConfig::IdType,
                          typename MeshConfig::GlobalIndexType >
{
   public:
      using MeshTraitsType  = MeshTraits< MeshConfig >;
      using EntityTopology  = EntityTopology_;
      using GlobalIndexType = typename MeshTraitsType::GlobalIndexType;
      using LocalIndexType  = typename MeshTraitsType::LocalIndexType;

      template< int Subdimensions >
      using SubentityTraits = typename MeshTraitsType::template SubentityTraits< EntityTopology, Subdimensions >;

      template< int SuperDimensions >
      using SuperentityTraits = typename MeshTraitsType::template SuperentityTraits< EntityTopology, SuperDimensions >;

      static String getType();

      String getTypeVirtual() const;

      bool save( File& file ) const;

      bool load( File& file );

      void print( std::ostream& str ) const;

      bool operator==( const MeshEntity& entity ) const;

      bool operator!=( const MeshEntity& entity ) const;

      constexpr int getEntityDimensions() const;

      /****
       * Subentities
       */
      using MeshSubentityAccess< MeshConfig, EntityTopology_ >::getNumberOfSubentities;
      using MeshSubentityAccess< MeshConfig, EntityTopology_ >::getSubentityIndex;
      using MeshSubentityAccess< MeshConfig, EntityTopology_ >::getSubentityOrientation;

      /****
       * Superentities
       */
      using MeshSuperentityAccess< MeshConfig, EntityTopology_ >::getNumberOfSuperentities;
      using MeshSuperentityAccess< MeshConfig, EntityTopology_ >::getSuperentityIndex;

      /****
       * Vertices
       */
      constexpr LocalIndexType getNumberOfVertices() const;

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

   template< typename Mesh, typename DimensionsTag, typename SuperdimensionsTag >
   friend struct MeshEntityStorageRebinderWorker;
};

/****
 * Vertex entity specialization
 */
template< typename MeshConfig >
class MeshEntity< MeshConfig, MeshVertexTopology >
   : protected MeshSuperentityAccess< MeshConfig, MeshVertexTopology >,
     public MeshEntityId< typename MeshConfig::IdType,
                          typename MeshConfig::GlobalIndexType >
{
   public:
      using MeshTraitsType  = MeshTraits< MeshConfig >;
      using EntityTopology  = MeshVertexTopology;
      using GlobalIndexType = typename MeshTraitsType::GlobalIndexType;
      using LocalIndexType  = typename MeshTraitsType::LocalIndexType;
      using PointType       = typename MeshTraitsType::PointType;

      template< int SuperDimensions >
      using SuperentityTraits = typename MeshTraitsType::template SuperentityTraits< EntityTopology, SuperDimensions >;

      static String getType();

      String getTypeVirtual() const;

      bool save( File& file ) const;

      bool load( File& file );

      void print( std::ostream& str ) const;

      bool operator==( const MeshEntity& entity ) const;

      bool operator!=( const MeshEntity& entity ) const;

      constexpr int getEntityDimensions() const;

      /****
       * Superentities
       */
      using MeshSuperentityAccess< MeshConfig, MeshVertexTopology >::getNumberOfSuperentities;
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

   template< typename Mesh, typename DimensionsTag, typename SuperdimensionsTag >
   friend struct MeshEntityStorageRebinderWorker;
};

template< typename MeshConfig,
          typename EntityTopology >
std::ostream& operator<<( std::ostream& str, const MeshEntity< MeshConfig, EntityTopology >& entity );

} // namespace Meshes
} // namespace TNL

#include <TNL/Meshes/MeshDetails/MeshEntity_impl.h>
