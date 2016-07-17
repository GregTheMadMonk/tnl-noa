/***************************************************************************
                          tnlMeshEntity.h  -  description
                             -------------------
    begin                : Feb 11, 2014
    copyright            : (C) 2014 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <core/tnlFile.h>
#include <core/tnlDynamicTypeTag.h>
#include <mesh/tnlMeshEntityId.h>
#include <mesh/traits/tnlMeshTraits.h>
#include <mesh/tnlDimensionsTag.h>
#include <mesh/topologies/tnlMeshVertexTopology.h>
#include <mesh/layers/tnlMeshSubentityStorageLayer.h>
#include <mesh/layers/tnlMeshSuperentityStorageLayer.h>
#include <mesh/layers/tnlMeshSuperentityAccess.h>
#include <mesh/initializer/tnlMeshEntitySeed.h>

namespace TNL {

template< typename MeshConfig >
class tnlMeshInitializer;

template< typename MeshConfig,
          typename EntityTopology_ >
class tnlMeshEntity
   : public tnlMeshSubentityStorageLayers< MeshConfig, EntityTopology_ >,
     public tnlMeshSuperentityAccess< MeshConfig, EntityTopology_ >,
     public tnlMeshEntityId< typename MeshConfig::IdType,
                             typename MeshConfig::GlobalIndexType >
{
   public:

      typedef tnlMeshTraits< MeshConfig >                         MeshTraits;
      typedef EntityTopology_                                     EntityTopology;
      typedef typename MeshTraits::GlobalIndexType                GlobalIndexType;
      typedef typename MeshTraits::LocalIndexType                 LocalIndexType;
      typedef typename MeshTraits::IdPermutationArrayAccessorType IdPermutationArrayAccessorType;
      typedef tnlMeshEntitySeed< MeshConfig, EntityTopology >     SeedType;

      template< int Subdimensions > using SubentityTraits =
      typename MeshTraits::template SubentityTraits< EntityTopology, Subdimensions >;
 
      template< int SuperDimensions > using SuperentityTraits =
      typename MeshTraits::template SuperentityTraits< EntityTopology, SuperDimensions >;
 
      tnlMeshEntity( const SeedType& entitySeed );

      tnlMeshEntity();
 
      ~tnlMeshEntity();

      static tnlString getType();

      tnlString getTypeVirtual() const;

      bool save( tnlFile& file ) const;

      bool load( tnlFile& file );

      void print( ostream& str ) const;

      bool operator==( const tnlMeshEntity& entity ) const;
 
      constexpr int getEntityDimensions() const;

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
      typename SubentityTraits< Subdimensions >::AccessArrayType& getSubentitiesIndices();

      template< int Subdimensions >
      const typename SubentityTraits< Subdimensions >::AccessArrayType& getSubentitiesIndices() const;

      /****
       * Superentities
       */
      template< int SuperDimensions >
      LocalIndexType getNumberOfSuperentities() const;

      template< int SuperDimensions >
      GlobalIndexType getSuperentityIndex( const LocalIndexType localIndex ) const;

      template< int SuperDimensions >
         typename SuperentityTraits< SuperDimensions >::AccessArrayType& getSuperentitiesIndices();

      template< int SuperDimensions >
         const typename SuperentityTraits< SuperDimensions >::AccessArrayType& getSuperentitiesIndices() const;

      /****
       * Vertices
       */
      constexpr LocalIndexType getNumberOfVertices() const;

      GlobalIndexType getVertexIndex( const LocalIndexType localIndex ) const;

      typename SubentityTraits< 0 >::AccessArrayType& getVerticesIndices();

      const typename SubentityTraits< 0 >::AccessArrayType& getVerticesIndices() const;

      template< int Dimensions >
      IdPermutationArrayAccessorType subentityOrientation( LocalIndexType index ) const;
 
   protected:

      /****
       * Methods for the mesh initialization
       */
      typedef tnlMeshSuperentityAccess< MeshConfig, EntityTopology >            SuperentityAccessBase;
      typedef typename MeshTraits::IdArrayAccessorType                          IdArrayAccessorType;
      typedef tnlMeshSubentityStorageLayers< MeshConfig, EntityTopology >       SubentityStorageLayers;

      template< int Subdimensions >
      void setSubentityIndex( const LocalIndexType localIndex,
                              const GlobalIndexType globalIndex );
 
      template< int Subdimensions >
      typename SubentityTraits< Subdimensions >::IdArrayType& subentityIdsArray();

      template< int Superdimensions >
      IdArrayAccessorType& superentityIdsArray();

      template< int Subdimensions >
      typename SubentityTraits< Subdimensions >::OrientationArrayType& subentityOrientationsArray();
 
   friend tnlMeshInitializer< MeshConfig >;
 
};

/****
 * Vertex entity specialization
 */
template< typename MeshConfig >
class tnlMeshEntity< MeshConfig, tnlMeshVertexTopology >
   : public tnlMeshSuperentityAccess< MeshConfig, tnlMeshVertexTopology >,
     public tnlMeshEntityId< typename MeshConfig::IdType,
                             typename MeshConfig::GlobalIndexType >
{
   public:

      typedef tnlMeshTraits< MeshConfig >                         MeshTraits;
      typedef tnlMeshVertexTopology                               EntityTopology;
      typedef typename MeshTraits::GlobalIndexType                GlobalIndexType;
      typedef typename MeshTraits::LocalIndexType                 LocalIndexType;
      typedef typename MeshTraits::PointType                      PointType;
      typedef typename MeshTraits::IdPermutationArrayAccessorType IdPermutationArrayAccessorType;
      typedef tnlMeshEntitySeed< MeshConfig, EntityTopology >     SeedType;
 
      template< int SuperDimensions > using SuperentityTraits =
      typename MeshTraits::template SuperentityTraits< EntityTopology, SuperDimensions >;

      static tnlString getType();

      tnlString getTypeVirtual() const;

      ~tnlMeshEntity();

      bool save( tnlFile& file ) const;

      bool load( tnlFile& file );

      void print( ostream& str ) const;

      bool operator==( const tnlMeshEntity& entity ) const;
 
      constexpr int getEntityDimensions() const;

      template< int Superdimensions > LocalIndexType getNumberOfSuperentities() const;

      template< int Superdimensions >
         typename SuperentityTraits< Superdimensions >::AccessArrayType& getSuperentitiesIndices();

      template< int Superdimensions >
         const typename SuperentityTraits< Superdimensions >::AccessArrayType& getSuperentitiesIndeces() const;

      template< int Dimensions >
      GlobalIndexType getSuperentityIndex( const LocalIndexType localIndex ) const;

      /****
       * Points
       */
      PointType getPoint() const;

      void setPoint( const PointType& point );

   protected:
 
      typedef typename MeshTraits::IdArrayAccessorType                          IdArrayAccessorType;
      typedef tnlMeshSuperentityAccess< MeshConfig, tnlMeshVertexTopology >     SuperentityAccessBase;
 
      template< int Superdimensions >
      IdArrayAccessorType& superentityIdsArray();

      PointType point;
 
   friend tnlMeshInitializer< MeshConfig >;
};

template< typename MeshConfig,
          typename EntityTopology >
ostream& operator <<( ostream& str, const tnlMeshEntity< MeshConfig, EntityTopology >& entity );

/****
 * This tells the compiler that theMeshEntity is a type with a dynamic memory allocation.
 * It is necessary for the loading and the saving of the mesh entities arrays.
 */
template< typename MeshConfig,
          typename EntityTopology >
struct tnlDynamicTypeTag< tnlMeshEntity< MeshConfig, EntityTopology > >
{
   enum { value = true };
};

} // namespace TNL

#include <mesh/tnlMeshEntity_impl.h>
