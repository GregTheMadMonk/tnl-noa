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
#include <TNL/Containers/DynamicTypeTag.h>
#include <TNL/Meshes/MeshDetails/MeshEntityId.h>
#include <TNL/Meshes/MeshDetails/traits/MeshTraits.h>
#include <TNL/Meshes/MeshDimensionTag.h>
#include <TNL/Meshes/Topologies/MeshVertexTopology.h>
#include <TNL/Meshes/MeshDetails/layers/MeshSubentityStorageLayer.h>
#include <TNL/Meshes/MeshDetails/layers/MeshSuperentityAccess.h>
#include <TNL/Meshes/MeshDetails/initializer/MeshEntitySeed.h>

namespace TNL {
namespace Meshes {

template< typename MeshConfig >
class MeshInitializer;

template< typename MeshConfig,
          typename EntityTopology_ >
class MeshEntity
   : public MeshSubentityStorageLayers< MeshConfig, EntityTopology_ >,
     protected MeshSuperentityAccess< MeshConfig, EntityTopology_ >,
     public MeshEntityId< typename MeshConfig::IdType,
                          typename MeshConfig::GlobalIndexType >
{
   public:

      typedef MeshTraits< MeshConfig >                                MeshTraitsType;
      typedef EntityTopology_                                         EntityTopology;
      typedef typename MeshTraitsType::GlobalIndexType                GlobalIndexType;
      typedef typename MeshTraitsType::LocalIndexType                 LocalIndexType;
      typedef typename MeshTraitsType::IdPermutationArrayAccessorType IdPermutationArrayAccessorType;

      template< int Subdimensions >
      using SubentityTraits = typename MeshTraitsType::template SubentityTraits< EntityTopology, Subdimensions >;
 
      template< int SuperDimensions >
      using SuperentityTraits = typename MeshTraitsType::template SuperentityTraits< EntityTopology, SuperDimensions >;
 
      MeshEntity();
 
      ~MeshEntity();

      static String getType();

      String getTypeVirtual() const;

      bool save( File& file ) const;

      bool load( File& file );

      void print( std::ostream& str ) const;

      bool operator==( const MeshEntity& entity ) const;

      bool operator!=( const MeshEntity& entity ) const;
 
      constexpr int getEntityDimension() const;

      /****
       * Subentities
       */
      template< int Subdimensions >
      constexpr bool subentitiesAvailable() const;

      template< int Subdimensions >
      constexpr LocalIndexType getNumberOfSubentities() const;

      template< int Subdimensions >
      GlobalIndexType getSubentityIndex( const LocalIndexType localIndex) const;

      template< int Subdimensions >
      typename SubentityTraits< Subdimensions >::AccessArrayType& getSubentityIndices();

      template< int Subdimensions >
      const typename SubentityTraits< Subdimensions >::AccessArrayType& getSubentityIndices() const;

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

      typename SubentityTraits< 0 >::AccessArrayType& getVerticesIndices();

      const typename SubentityTraits< 0 >::AccessArrayType& getVerticesIndices() const;

      template< int Dimension >
      IdPermutationArrayAccessorType subentityOrientation( LocalIndexType index ) const;
 
   protected:
      /****
       * Methods for the mesh initialization
       */
      using MeshSuperentityAccess< MeshConfig, EntityTopology_ >::bindSuperentitiesStorageNetwork;
      using MeshSuperentityAccess< MeshConfig, EntityTopology_ >::setNumberOfSuperentities;
      using MeshSuperentityAccess< MeshConfig, EntityTopology_ >::setSuperentityIndex;

      template< int Subdimensions >
      void setSubentityIndex( const LocalIndexType& localIndex,
                              const GlobalIndexType& globalIndex );
 
      template< int Subdimensions >
      typename SubentityTraits< Subdimensions >::IdArrayType& subentityIdsArray();

      template< int Subdimensions >
      typename SubentityTraits< Subdimensions >::OrientationArrayType& subentityOrientationsArray();
 
   friend MeshInitializer< MeshConfig >;
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

      typedef MeshTraits< MeshConfig >                                MeshTraitsType;
      typedef MeshVertexTopology                                      EntityTopology;
      typedef typename MeshTraitsType::GlobalIndexType                GlobalIndexType;
      typedef typename MeshTraitsType::LocalIndexType                 LocalIndexType;
      typedef typename MeshTraitsType::PointType                      PointType;
      typedef typename MeshTraitsType::IdPermutationArrayAccessorType IdPermutationArrayAccessorType;
 
      template< int SuperDimensions >
      using SuperentityTraits = typename MeshTraitsType::template SuperentityTraits< EntityTopology, SuperDimensions >;

      static String getType();

      String getTypeVirtual() const;

      ~MeshEntity();

      bool save( File& file ) const;

      bool load( File& file );

      void print( std::ostream& str ) const;

      bool operator==( const MeshEntity& entity ) const;

      bool operator!=( const MeshEntity& entity ) const;
 
      constexpr int getEntityDimension() const;

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
};

template< typename MeshConfig,
          typename EntityTopology >
std::ostream& operator <<( std::ostream& str, const MeshEntity< MeshConfig, EntityTopology >& entity );
} // namespace Meshes

/****
 * This tells the compiler that theMeshEntity is a type with a dynamic memory allocation.
 * It is necessary for the loading and the saving of the mesh entities arrays.
 */
namespace Containers {
template< typename MeshConfig,
          typename EntityTopology >
struct DynamicTypeTag< Meshes::MeshEntity< MeshConfig, EntityTopology > >
{
   enum { value = true };
};
}

} // namespace TNL

#include <TNL/Meshes/MeshDetails/MeshEntity_impl.h>
