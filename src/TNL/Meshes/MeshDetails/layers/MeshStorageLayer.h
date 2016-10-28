/***************************************************************************
                          MeshStorageLayer.h  -  description
                             -------------------
    begin                : Feb 11, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
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
#include <TNL/Meshes/MeshDetails/traits/MeshEntityTraits.h>
#include <TNL/Meshes/MeshDetails/layers/MeshSubentityStorageLayer.h>
#include <TNL/Meshes/MeshDetails/layers/MeshSuperentityStorageLayer.h>

namespace TNL {
namespace Meshes {

template< typename MeshConfig,
          typename DimensionTag,
          bool EntityStorage = MeshEntityTraits< MeshConfig, DimensionTag::value >::storageEnabled >
class MeshStorageLayer;


template< typename MeshConfig >
class MeshStorageLayers
   : public MeshStorageLayer< MeshConfig, typename MeshTraits< MeshConfig >::DimensionTag >
{
   using MeshTraitsType   = MeshTraits< MeshConfig >;
   using BaseType         = MeshStorageLayer< MeshConfig, typename MeshTraitsType::DimensionsTag >;
   template< int Dimensions >
   using EntityTraits = typename MeshTraitsType::template EntityTraits< Dimensions >;

protected:
   template< int Dimensions >
   bool setNumberOfEntities( const typename EntityTraits< Dimensions >::GlobalIndexType& entitiesCount )
   {
      return BaseType::setNumberOfEntities( MeshDimensionsTag< Dimensions >(), entitiesCount );
   }

   template< int Dimensions, int Subdimensions >
   typename MeshTraitsType::template SubentityTraits< typename EntityTraits< Dimensions >::EntityTopology, Subdimensions >::StorageNetworkType&
   getSubentityStorageNetwork()
   {
      static_assert( Dimensions > Subdimensions, "Invalid combination of Dimensions and Subdimensions." );
      using BaseType = MeshSubentityStorageLayers< MeshConfig,
                                                   typename MeshTraits< MeshConfig >::template EntityTraits< Dimensions >::EntityTopology >;
      return BaseType::template getSubentityStorageNetwork< Subdimensions >();
   }

   template< int Dimensions, int Superdimensions >
   typename MeshTraitsType::template SuperentityTraits< typename EntityTraits< Dimensions >::EntityTopology, Superdimensions >::StorageNetworkType&
   getSuperentityStorageNetwork()
   {
      static_assert( Dimensions < Superdimensions, "Invalid combination of Dimensions and Superdimensions." );
      using BaseType = MeshSuperentityStorageLayers< MeshConfig,
                                                     typename MeshTraits< MeshConfig >::template EntityTraits< Dimensions >::EntityTopology >;
      return BaseType::template getSuperentityStorageNetwork< Superdimensions >( MeshDimensionsTag< Dimensions >() );
   }
};


template< typename MeshConfig,
          typename DimensionTag >
class MeshStorageLayer< MeshConfig,
                        DimensionsTag,
                        true >
   : public MeshStorageLayer< MeshConfig, typename DimensionsTag::Decrement >,
     public MeshSubentityStorageLayers< MeshConfig,
                                        typename MeshTraits< MeshConfig >::template EntityTraits< DimensionsTag::value >::EntityTopology >,
     public MeshSuperentityStorageLayers< MeshConfig,
                                          typename MeshTraits< MeshConfig >::template EntityTraits< DimensionsTag::value >::EntityTopology >
{
public:
   using BaseType = MeshStorageLayer< MeshConfig, typename DimensionsTag::Decrement >;
   using MeshTraitsType   = MeshTraits< MeshConfig >;
   using EntityTraitsType = typename MeshTraitsType::template EntityTraits< DimensionsTag::value >;
   using StorageArrayType = typename EntityTraitsType::StorageArrayType;
   using GlobalIndexType  = typename EntityTraitsType::GlobalIndexType;
   using EntityType       = typename EntityTraitsType::EntityType;
   using EntityTopology   = typename EntityTraitsType::EntityTopology;
   using SubentityStorageBaseType = MeshSubentityStorageLayers< MeshConfig, EntityTopology >;
   using SuperentityStorageBaseType = MeshSuperentityStorageLayers< MeshConfig, EntityTopology >;

   /****
     * Make visible getters of the lower layer
     */
   using BaseType::setNumberOfEntities;
   using BaseType::getNumberOfEntities;
   using BaseType::getEntity;

   MeshStorageLayer()
   {
   }

   bool setNumberOfEntities( DimensionsTag, const GlobalIndexType& entitiesCount )
   {
      if( ! this->entities.setSize( entitiesCount ) )
         return false;
      if( ! SubentityStorageBaseType::setNumberOfEntities( entitiesCount ) )
         return false;
      if( ! SuperentityStorageBaseType::setNumberOfEntities( entitiesCount ) )
         return false;
      return true;
   }

   GlobalIndexType getNumberOfEntities( DimensionsTag ) const
   {
      return this->entities.getSize();
   }

   EntityType& getEntity( DimensionsTag,
                          const GlobalIndexType entityIndex )
   {
      return this->entities[ entityIndex ];
   }

   const EntityType& getEntity( DimensionsTag,
                                const GlobalIndexType entityIndex ) const
   {
      return this->entities[ entityIndex ];
   }

   bool save( File& file ) const
   {
      if( ! BaseType::save( file ) ||
          ! SubentityStorageBaseType::save( file ) ||
          ! SuperentityStorageBaseType::save( file ) ||
          ! this->entities.save( file ) )
      {
         std::cerr << "Saving of the mesh entities with " << DimensionsTag::value << " dimensions failed." << std::endl;
         return false;
      }
      return true;
   }

   bool load( File& file )
   {
      if( ! BaseType::load( file ) ||
          ! SubentityStorageBaseType::load( file ) ||
          ! SuperentityStorageBaseType::load( file ) ||
          ! this->entities.load( file ) )
      {
         std::cerr << "Loading of the mesh entities with " << DimensionsTag::value << " dimensions failed." << std::endl;
         return false;
      }
      return true;
   }

   void print( std::ostream& str ) const
   {
      BaseType::print( str );
      str << "The entities with " << DimensionsTag::value << " dimensions are: " << std::endl;
      for( GlobalIndexType i = 0; i < entities.getSize();i ++ )
         str << i << " " << entities[ i ] << std::endl;
      SubentityStorageBaseType::print( str );
      SuperentityStorageBaseType::print( str );
      str << std::endl;
   }

   bool operator==( const MeshStorageLayer& meshLayer ) const
   {
      return ( BaseType::operator==( meshLayer ) &&
               SubentityStorageBaseType::operator==( meshLayer ) &&
               SuperentityStorageBaseType::operator==( meshLayer ) &&
               entities == meshLayer.entities );
   }

protected:
   StorageArrayType entities;
};

template< typename MeshConfig,
          typename DimensionTag >
class MeshStorageLayer< MeshConfig, DimensionTag, false >
   : public MeshStorageLayer< MeshConfig, typename DimensionTag::Decrement  >
{
};

template< typename MeshConfig >
class MeshStorageLayer< MeshConfig, MeshDimensionsTag< 0 >, true >
   : public MeshSuperentityStorageLayers< MeshConfig,
                                          MeshVertexTopology >
{
public:
   using DimensionsTag              = MeshDimensionsTag< 0 >;
   using SuperentityStorageBaseType = MeshSuperentityStorageLayers< MeshConfig, MeshVertexTopology >;

   using MeshTraitsType             = MeshTraits< MeshConfig >;
   using EntityTraitsType           = typename MeshTraitsType::template EntityTraits< 0 >;
   using StorageArrayType           = typename EntityTraitsType::StorageArrayType;
   using GlobalIndexType            = typename EntityTraitsType::GlobalIndexType;
   using VertexType                 = typename EntityTraitsType::EntityType;
   using PointType                  = typename VertexType::PointType;
   using EntityTopology             = MeshVertexTopology;

   MeshStorageLayer()
   {
   }

   GlobalIndexType getNumberOfVertices() const
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

   bool setNumberOfEntities( DimensionsTag, const GlobalIndexType& entitiesCount )
   {
      if( ! this->vertices.setSize( entitiesCount ) )
         return false;
      if( ! SuperentityStorageBaseType::setNumberOfEntities( entitiesCount ) )
         return false;
      return true;
   }

   GlobalIndexType getNumberOfEntities( DimensionsTag ) const
   {
      return this->vertices.getSize();
   }

   VertexType& getEntity( DimensionTag,
                          const GlobalIndexType entityIndex )
   {
      return this->vertices[ entityIndex ];
   }

   const VertexType& getEntity( DimensionTag,
                                const GlobalIndexType entityIndex ) const
   {
      return this->vertices.getElement( entityIndex );
   }

   bool save( File& file ) const
   {
      if( ! SuperentityStorageBaseType::save( file ) ||
          ! this->vertices.save( file ) )
      {
         std::cerr << "Saving of the mesh entities with " << DimensionTag::value << " dimensions failed." << std::endl;
         return false;
      }
      return true;
   }

   bool load( File& file )
   {
      if( ! SuperentityStorageBaseType::load( file ) ||
          ! this->vertices.load( file ) )
      {
         std::cerr << "Loading of the mesh entities with " << DimensionTag::value << " dimensions failed." << std::endl;
         return false;
      }
      return true;
   }

   void print( std::ostream& str ) const
   {
      str << "The mesh vertices are: " << std::endl;
      for( GlobalIndexType i = 0; i < vertices.getSize(); i++ )
         str << i << vertices[ i ] << std::endl;
      SuperentityStorageBaseType::print( str );
      str << std::endl;
   }

   bool operator==( const MeshStorageLayer& meshLayer ) const
   {
      return ( SuperentityStorageBaseType::operator==( meshLayer ) && vertices == meshLayer.vertices );
   }

protected:
   StorageArrayType vertices;
};

/****
 * Forces termination of recursive inheritance (prevents compiler from generating huge error logs)
 */
template< typename MeshConfig >
class MeshStorageLayer< MeshConfig, MeshDimensionTag< 0 >, false >
{
};

} // namespace Meshes
} // namespace TNL
