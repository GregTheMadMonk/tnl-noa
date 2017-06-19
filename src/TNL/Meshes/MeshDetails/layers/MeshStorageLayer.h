/***************************************************************************
                          MeshStorageLayer.h  -  description
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
#include <TNL/Meshes/MeshDetails/layers/MeshSubentityStorageLayer.h>
#include <TNL/Meshes/MeshDetails/layers/MeshSuperentityStorageLayer.h>
#include <TNL/Meshes/MeshDetails/layers/MeshBoundaryTagsLayer.h>

namespace TNL {
namespace Meshes {

template< typename MeshConfig,
          typename Device,
          typename DimensionTag,
          bool EntityStorage = MeshTraits< MeshConfig, Device >::template EntityTraits< DimensionTag::value >::storageEnabled >
class MeshStorageLayer;


template< typename MeshConfig, typename Device >
class MeshStorageLayers
   : public MeshStorageLayer< MeshConfig, Device, typename MeshTraits< MeshConfig, Device >::DimensionTag >
{
   using MeshTraitsType = MeshTraits< MeshConfig, Device >;
   using BaseType       = MeshStorageLayer< MeshConfig, Device, typename MeshTraitsType::DimensionTag >;
   template< int Dimension >
   using EntityTraits = typename MeshTraitsType::template EntityTraits< Dimension >;

public:
   MeshStorageLayers() = default;
   explicit MeshStorageLayers( const MeshStorageLayers& other ) : BaseType( other ) {}
   template< typename Device_ >
   MeshStorageLayers( const MeshStorageLayers< MeshConfig, Device_ >& other ) : BaseType( other ) {}

protected:
   template< int Dimension >
   void setNumberOfEntities( const typename EntityTraits< Dimension >::GlobalIndexType& entitiesCount )
   {
      static_assert( EntityTraits< Dimension >::storageEnabled, "You try to set number of entities which are not configured for storage." );
      BaseType::setNumberOfEntities( DimensionTag< Dimension >(), entitiesCount );
   }

   template< int Dimension, int Subdimension >
   typename MeshTraitsType::template SubentityTraits< typename EntityTraits< Dimension >::EntityTopology, Subdimension >::StorageNetworkType&
   getSubentityStorageNetwork()
   {
      static_assert( EntityTraits< Dimension >::storageEnabled, "You try to get subentity storage of entities which are not configured for storage." );
      static_assert( Dimension > Subdimension, "Invalid combination of Dimension and Subdimension." );
      using BaseType = MeshSubentityStorageLayers< MeshConfig,
                                                   Device,
                                                   typename EntityTraits< Dimension >::EntityTopology >;
      return BaseType::template getSubentityStorageNetwork< Subdimension >();
   }

   template< int Dimension, int Superdimension >
   typename MeshTraitsType::template SuperentityTraits< typename EntityTraits< Dimension >::EntityTopology, Superdimension >::StorageNetworkType&
   getSuperentityStorageNetwork()
   {
      static_assert( EntityTraits< Dimension >::storageEnabled, "You try to get superentity storage of entities which are not configured for storage." );
      static_assert( Dimension < Superdimension, "Invalid combination of Dimension and Superdimension." );
      using BaseType = MeshSuperentityStorageLayers< MeshConfig,
                                                     Device,
                                                     typename EntityTraits< Dimension >::EntityTopology >;
      return BaseType::template getSuperentityStorageNetwork< Superdimension >();
   }


   // The following methods are implemented in the BoundaryTags layers. They are
   // needed for the mesh traverser.
   template< int Dimension >
   __cuda_callable__
   bool isBoundaryEntity( const typename EntityTraits< Dimension >::GlobalIndexType& entityIndex ) const
   {
      static_assert( EntityTraits< Dimension >::boundaryTagsEnabled, "You try to access boundary tags which are not configured for storage." );
      return BaseType::isBoundaryEntity( DimensionTag< Dimension >(), entityIndex );
   }

   template< int Dimension >
   __cuda_callable__
   typename EntityTraits< Dimension >::GlobalIndexType getBoundaryEntitiesCount() const
   {
      static_assert( EntityTraits< Dimension >::boundaryTagsEnabled, "You try to access boundary tags which are not configured for storage." );
      return BaseType::getBoundaryEntitiesCount( DimensionTag< Dimension >() );
   }

   template< int Dimension >
   __cuda_callable__
   typename EntityTraits< Dimension >::GlobalIndexType getBoundaryEntityIndex( const typename EntityTraits< Dimension >::GlobalIndexType& i ) const
   {
      static_assert( EntityTraits< Dimension >::boundaryTagsEnabled, "You try to access boundary tags which are not configured for storage." );
      return BaseType::getBoundaryEntityIndex( DimensionTag< Dimension >(), i );
   }

   template< int Dimension >
   __cuda_callable__
   typename EntityTraits< Dimension >::GlobalIndexType getInteriorEntitiesCount() const
   {
      static_assert( EntityTraits< Dimension >::boundaryTagsEnabled, "You try to access boundary tags which are not configured for storage." );
      return BaseType::getInteriorEntitiesCount( DimensionTag< Dimension >() );
   }

   template< int Dimension >
   __cuda_callable__
   typename EntityTraits< Dimension >::GlobalIndexType getInteriorEntityIndex( const typename EntityTraits< Dimension >::GlobalIndexType& i ) const
   {
      static_assert( EntityTraits< Dimension >::boundaryTagsEnabled, "You try to access boundary tags which are not configured for storage." );
      return BaseType::getInteriorEntityIndex( DimensionTag< Dimension >(), i );
   }

   // setters for boundary tags
   template< int Dimension >
   void resetBoundaryTags()
   {
      BaseType::resetBoundaryTags( DimensionTag< Dimension >() );
   }

   template< int Dimension >
   void updateBoundaryIndices()
   {
      BaseType::updateBoundaryIndices( DimensionTag< Dimension >() );
   }

   template< int Dimension >
   __cuda_callable__
   void setIsBoundaryEntity( const typename EntityTraits< Dimension >::GlobalIndexType& entityIndex, bool isBoundary )
   {
      static_assert( EntityTraits< Dimension >::boundaryTagsEnabled, "You try to access boundary tags which are not configured for storage." );
      BaseType::setIsBoundaryEntity( DimensionTag< Dimension >(), entityIndex, isBoundary );
   }
};


template< typename MeshConfig,
          typename Device,
          typename DimensionTag >
class MeshStorageLayer< MeshConfig,
                        Device,
                        DimensionTag,
                        true >
   : public MeshStorageLayer< MeshConfig, Device, typename DimensionTag::Decrement >,
     public MeshSubentityStorageLayers< MeshConfig,
                                        Device,
                                        typename MeshTraits< MeshConfig, Device >::template EntityTraits< DimensionTag::value >::EntityTopology >,
     public MeshSuperentityStorageLayers< MeshConfig,
                                          Device,
                                          typename MeshTraits< MeshConfig, Device >::template EntityTraits< DimensionTag::value >::EntityTopology >,
     public MeshBoundaryTagsLayer< MeshConfig, Device, DimensionTag >
{
public:
   using BaseType = MeshStorageLayer< MeshConfig, Device, typename DimensionTag::Decrement >;
   using MeshTraitsType   = MeshTraits< MeshConfig, Device >;
   using EntityTraitsType = typename MeshTraitsType::template EntityTraits< DimensionTag::value >;
   using StorageArrayType = typename EntityTraitsType::StorageArrayType;
   using GlobalIndexType  = typename EntityTraitsType::GlobalIndexType;
   using EntityType       = typename EntityTraitsType::EntityType;
   using EntityTopology   = typename EntityTraitsType::EntityTopology;
   using SubentityStorageBaseType = MeshSubentityStorageLayers< MeshConfig, Device, EntityTopology >;
   using SuperentityStorageBaseType = MeshSuperentityStorageLayers< MeshConfig, Device, EntityTopology >;
   using BoundaryTagsBaseType = MeshBoundaryTagsLayer< MeshConfig, Device, DimensionTag >;

   /****
     * Make visible getters of the lower layer
     */
   using BaseType::setNumberOfEntities;
   using BaseType::getEntitiesCount;
   using BaseType::getEntity;

   using BaseType::resetBoundaryTags;
   using BaseType::isBoundaryEntity;
   using BaseType::setIsBoundaryEntity;
   using BaseType::updateBoundaryIndices;
   using BaseType::getBoundaryEntitiesCount;
   using BaseType::getBoundaryEntityIndex;
   using BaseType::getInteriorEntitiesCount;
   using BaseType::getInteriorEntityIndex;

   using BoundaryTagsBaseType::resetBoundaryTags;
   using BoundaryTagsBaseType::isBoundaryEntity;
   using BoundaryTagsBaseType::setIsBoundaryEntity;
   using BoundaryTagsBaseType::updateBoundaryIndices;
   using BoundaryTagsBaseType::getBoundaryEntitiesCount;
   using BoundaryTagsBaseType::getBoundaryEntityIndex;
   using BoundaryTagsBaseType::getInteriorEntitiesCount;
   using BoundaryTagsBaseType::getInteriorEntityIndex;

   MeshStorageLayer() = default;

   explicit MeshStorageLayer( const MeshStorageLayer& other )
   {
      operator=( other );
   }

   template< typename Device_ >
   MeshStorageLayer( const MeshStorageLayer< MeshConfig, Device_, DimensionTag >& other )
   {
      operator=( other );
   }

   MeshStorageLayer& operator=( const MeshStorageLayer& other )
   {
      BaseType::operator=( other );
      SubentityStorageBaseType::operator=( other );
      SuperentityStorageBaseType::operator=( other );
      BoundaryTagsBaseType::operator=( other );
      // TODO: throw exception if allocation fails
      entities.setLike( other.entities);
      entities = other.entities;
      return *this;
   }

   template< typename Device_ >
   MeshStorageLayer& operator=( const MeshStorageLayer< MeshConfig, Device_, DimensionTag >& other )
   {
      BaseType::operator=( other );
      SubentityStorageBaseType::operator=( other );
      SuperentityStorageBaseType::operator=( other );
      BoundaryTagsBaseType::operator=( other );
      // TODO: throw exception if allocation fails
      entities.setLike( other.entities);
      entities = other.entities;
      return *this;
   }


   void setNumberOfEntities( DimensionTag, const GlobalIndexType& entitiesCount )
   {
      this->entities.setSize( entitiesCount );
      SubentityStorageBaseType::setNumberOfEntities( entitiesCount );
      SuperentityStorageBaseType::setNumberOfEntities( entitiesCount );
      BoundaryTagsBaseType::setNumberOfEntities( entitiesCount );
   }

   __cuda_callable__
   GlobalIndexType getEntitiesCount( DimensionTag ) const
   {
      return this->entities.getSize();
   }

   __cuda_callable__
   EntityType& getEntity( DimensionTag,
                          const GlobalIndexType entityIndex )
   {
      return this->entities[ entityIndex ];
   }

   __cuda_callable__
   const EntityType& getEntity( DimensionTag,
                                const GlobalIndexType entityIndex ) const
   {
      return this->entities[ entityIndex ];
   }

   bool save( File& file ) const
   {
      if( ! BaseType::save( file ) ||
          ! SubentityStorageBaseType::save( file ) ||
          ! SuperentityStorageBaseType::save( file ) ||
          ! BoundaryTagsBaseType::save( file ) ||
          ! this->entities.save( file ) )
      {
         std::cerr << "Saving of the mesh entities with dimension " << DimensionTag::value << " failed." << std::endl;
         return false;
      }
      return true;
   }

   bool load( File& file )
   {
      if( ! BaseType::load( file ) ||
          ! SubentityStorageBaseType::load( file ) ||
          ! SuperentityStorageBaseType::load( file ) ||
          ! BoundaryTagsBaseType::load( file ) ||
          ! this->entities.load( file ) )
      {
         std::cerr << "Loading of the mesh entities with dimension " << DimensionTag::value << " failed." << std::endl;
         return false;
      }
      return true;
   }

   void print( std::ostream& str ) const
   {
      BaseType::print( str );
      str << "The entities with dimension " << DimensionTag::value << " are: " << std::endl;
      for( GlobalIndexType i = 0; i < entities.getSize(); i++ )
         str << i << " " << entities[ i ] << std::endl;
      SubentityStorageBaseType::print( str );
      SuperentityStorageBaseType::print( str );
      BoundaryTagsBaseType::print( str );
      str << std::endl;
   }

   bool operator==( const MeshStorageLayer& meshLayer ) const
   {
      return ( BaseType::operator==( meshLayer ) &&
               SubentityStorageBaseType::operator==( meshLayer ) &&
               SuperentityStorageBaseType::operator==( meshLayer ) &&
               BoundaryTagsBaseType::operator==( meshLayer ) &&
               entities == meshLayer.entities );
   }

protected:
   StorageArrayType entities;

   // friend class is needed for templated assignment operators
   template< typename MeshConfig_, typename Device_, typename DimensionTag_, bool Storage_ >
   friend class MeshStorageLayer;
};

template< typename MeshConfig,
          typename Device,
          typename DimensionTag >
class MeshStorageLayer< MeshConfig, Device, DimensionTag, false >
   : public MeshStorageLayer< MeshConfig, Device, typename DimensionTag::Decrement  >
{
public:
   using BaseType = MeshStorageLayer< MeshConfig, Device, typename DimensionTag::Decrement >;

   MeshStorageLayer() = default;
   explicit MeshStorageLayer( const MeshStorageLayer& other )
      : BaseType( other )
   {}
};

template< typename MeshConfig,
          typename Device >
class MeshStorageLayer< MeshConfig, Device, Meshes::DimensionTag< 0 >, true >
   : public MeshSuperentityStorageLayers< MeshConfig, Device, MeshVertexTopology >,
     public MeshBoundaryTagsLayer< MeshConfig, Device, Meshes::DimensionTag< 0 > >
{
public:
   using DimensionTag               = Meshes::DimensionTag< 0 >;
   using SuperentityStorageBaseType = MeshSuperentityStorageLayers< MeshConfig, Device, MeshVertexTopology >;
   using BoundaryTagsBaseType       = MeshBoundaryTagsLayer< MeshConfig, Device, Meshes::DimensionTag< 0 > >;

   using MeshTraitsType             = MeshTraits< MeshConfig, Device >;
   using EntityTraitsType           = typename MeshTraitsType::template EntityTraits< 0 >;
   using StorageArrayType           = typename EntityTraitsType::StorageArrayType;
   using GlobalIndexType            = typename EntityTraitsType::GlobalIndexType;
   using VertexType                 = typename EntityTraitsType::EntityType;
   using PointType                  = typename VertexType::PointType;
   using EntityTopology             = MeshVertexTopology;

   using BoundaryTagsBaseType::resetBoundaryTags;
   using BoundaryTagsBaseType::isBoundaryEntity;
   using BoundaryTagsBaseType::setIsBoundaryEntity;
   using BoundaryTagsBaseType::updateBoundaryIndices;
   using BoundaryTagsBaseType::getBoundaryEntitiesCount;
   using BoundaryTagsBaseType::getBoundaryEntityIndex;
   using BoundaryTagsBaseType::getInteriorEntitiesCount;
   using BoundaryTagsBaseType::getInteriorEntityIndex;

   MeshStorageLayer() = default;

   explicit MeshStorageLayer( const MeshStorageLayer& other )
   {
      operator=( other );
   }

   template< typename Device_ >
   MeshStorageLayer( const MeshStorageLayer< MeshConfig, Device_, DimensionTag >& other )
   {
      operator=( other );
   }

   MeshStorageLayer& operator=( const MeshStorageLayer& other )
   {
      SuperentityStorageBaseType::operator=( other );
      BoundaryTagsBaseType::operator=( other );
      // TODO: throw exception if allocation fails
      vertices.setLike( other.vertices);
      vertices = other.vertices;
      return *this;
   }

   template< typename Device_ >
   MeshStorageLayer& operator=( const MeshStorageLayer< MeshConfig, Device_, DimensionTag >& other )
   {
      SuperentityStorageBaseType::operator=( other );
      BoundaryTagsBaseType::operator=( other );
      // TODO: throw exception if allocation fails
      vertices.setLike( other.vertices);
      vertices = other.vertices;
      return *this;
   }


   GlobalIndexType getVerticesCount() const
   {
      return this->vertices.getSize();
   }

   void setVertex( const GlobalIndexType vertexIndex,
                   const VertexType& vertex ) const
   {
      this->vertices.setElement( vertexIndex, vertex );
   }

   VertexType& getVertex( const GlobalIndexType vertexIndex )
   {
      return this->vertices[ vertexIndex ];
   }

   const VertexType& getVertex( const GlobalIndexType vertexIndex ) const
   {
      return this->vertices[ vertexIndex ];
   }

   void setVertex( const GlobalIndexType vertexIndex,
                   const PointType& point )
   {
      this->vertices[ vertexIndex ].setPoint( point );
   }

   /****
    * This is only for the completeness and compatibility
    * with higher dimensions entities storage layers.
    */

   void setNumberOfEntities( DimensionTag, const GlobalIndexType& entitiesCount )
   {
      this->vertices.setSize( entitiesCount );
      SuperentityStorageBaseType::setNumberOfEntities( entitiesCount );
      BoundaryTagsBaseType::setNumberOfEntities( entitiesCount );
   }

   __cuda_callable__
   GlobalIndexType getEntitiesCount( DimensionTag ) const
   {
      return this->vertices.getSize();
   }

   __cuda_callable__
   VertexType& getEntity( DimensionTag,
                          const GlobalIndexType entityIndex )
   {
      return this->vertices[ entityIndex ];
   }

   __cuda_callable__
   const VertexType& getEntity( DimensionTag,
                                const GlobalIndexType entityIndex ) const
   {
      return this->vertices[ entityIndex ];
   }

   bool save( File& file ) const
   {
      if( ! SuperentityStorageBaseType::save( file ) ||
          ! BoundaryTagsBaseType::save( file ) ||
          ! this->vertices.save( file ) )
      {
         std::cerr << "Saving of the mesh entities with dimension " << DimensionTag::value << " failed." << std::endl;
         return false;
      }
      return true;
   }

   bool load( File& file )
   {
      if( ! SuperentityStorageBaseType::load( file ) ||
          ! BoundaryTagsBaseType::load( file ) ||
          ! this->vertices.load( file ) )
      {
         std::cerr << "Loading of the mesh entities with dimension " << DimensionTag::value << " failed." << std::endl;
         return false;
      }
      return true;
   }

   void print( std::ostream& str ) const
   {
      str << "The mesh vertices are: " << std::endl;
      for( GlobalIndexType i = 0; i < vertices.getSize(); i++ )
         str << i << " " << vertices[ i ] << std::endl;
      SuperentityStorageBaseType::print( str );
      BoundaryTagsBaseType::print( str );
      str << std::endl;
   }

   bool operator==( const MeshStorageLayer& meshLayer ) const
   {
      return ( SuperentityStorageBaseType::operator==( meshLayer ) &&
               BoundaryTagsBaseType::operator==( meshLayer ) &&
               vertices == meshLayer.vertices );
   }

protected:
   StorageArrayType vertices;

   // friend class is needed for templated assignment operators
   template< typename MeshConfig_, typename Device_, typename DimensionTag_, bool Storage_ >
   friend class MeshStorageLayer;
};

/****
 * Forces termination of recursive inheritance (prevents compiler from generating huge error logs)
 */
template< typename MeshConfig,
          typename Device >
class MeshStorageLayer< MeshConfig, Device, DimensionTag< 0 >, false >
{
   MeshStorageLayer() = default;
   explicit MeshStorageLayer( const MeshStorageLayer& other ) {}
};

} // namespace Meshes
} // namespace TNL
