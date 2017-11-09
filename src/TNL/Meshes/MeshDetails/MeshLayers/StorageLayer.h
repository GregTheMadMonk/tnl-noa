/***************************************************************************
                          StorageLayer.h  -  description
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
#include <TNL/Meshes/MeshDetails/traits/WeakStorageTraits.h>
#include <TNL/Meshes/MeshDetails/MeshLayers/SubentityStorageLayer.h>
#include <TNL/Meshes/MeshDetails/MeshLayers/SuperentityStorageLayer.h>

namespace TNL {
namespace Meshes {

template< typename MeshConfig,
          typename Device,
          typename DimensionTag,
          bool EntityStorage = WeakEntityStorageTrait< MeshConfig, Device, DimensionTag >::storageEnabled >
class StorageLayer;


template< typename MeshConfig, typename Device >
class StorageLayerFamily
   : public StorageLayer< MeshConfig, Device, DimensionTag< 0 > >
{
   using MeshTraitsType = MeshTraits< MeshConfig, Device >;
   using BaseType       = StorageLayer< MeshConfig, Device, DimensionTag< 0 > >;
   template< int Dimension >
   using EntityTraits = typename MeshTraitsType::template EntityTraits< Dimension >;

public:
   // inherit constructors and assignment operators (including templated versions)
   using BaseType::BaseType;
   using BaseType::operator=;

protected:
   template< int Dimension >
   void setEntitiesCount( const typename EntityTraits< Dimension >::GlobalIndexType& entitiesCount )
   {
      static_assert( EntityTraits< Dimension >::storageEnabled, "You try to set number of entities which are not configured for storage." );
      BaseType::setEntitiesCount( DimensionTag< Dimension >(), entitiesCount );
   }

   template< int Dimension, int Subdimension >
   typename MeshTraitsType::template SubentityTraits< typename EntityTraits< Dimension >::EntityTopology, Subdimension >::StorageNetworkType&
   getSubentityStorageNetwork()
   {
      static_assert( EntityTraits< Dimension >::storageEnabled, "You try to get subentity storage of entities which are not configured for storage." );
      static_assert( Dimension > Subdimension, "Invalid combination of Dimension and Subdimension." );
      using BaseType = SubentityStorageLayerFamily< MeshConfig,
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
      using BaseType = SuperentityStorageLayerFamily< MeshConfig,
                                                     Device,
                                                     typename EntityTraits< Dimension >::EntityTopology >;
      return BaseType::template getSuperentityStorageNetwork< Superdimension >();
   }
};


template< typename MeshConfig,
          typename Device,
          typename DimensionTag >
class StorageLayer< MeshConfig,
                    Device,
                    DimensionTag,
                    true >
   : public SubentityStorageLayerFamily< MeshConfig,
                                         Device,
                                         typename MeshTraits< MeshConfig, Device >::template EntityTraits< DimensionTag::value >::EntityTopology >,
     public SuperentityStorageLayerFamily< MeshConfig,
                                           Device,
                                           typename MeshTraits< MeshConfig, Device >::template EntityTraits< DimensionTag::value >::EntityTopology >,
     public StorageLayer< MeshConfig, Device, typename DimensionTag::Increment >
{
public:
   using BaseType = StorageLayer< MeshConfig, Device, typename DimensionTag::Increment >;
   using MeshTraitsType   = MeshTraits< MeshConfig, Device >;
   using EntityTraitsType = typename MeshTraitsType::template EntityTraits< DimensionTag::value >;
   using StorageArrayType = typename EntityTraitsType::StorageArrayType;
   using GlobalIndexType  = typename EntityTraitsType::GlobalIndexType;
   using EntityType       = typename EntityTraitsType::EntityType;
   using EntityTopology   = typename EntityTraitsType::EntityTopology;
   using SubentityStorageBaseType = SubentityStorageLayerFamily< MeshConfig, Device, EntityTopology >;
   using SuperentityStorageBaseType = SuperentityStorageLayerFamily< MeshConfig, Device, EntityTopology >;

   StorageLayer() = default;

   explicit StorageLayer( const StorageLayer& other )
   {
      operator=( other );
   }

   template< typename Device_ >
   StorageLayer( const StorageLayer< MeshConfig, Device_, DimensionTag >& other )
   {
      operator=( other );
   }

   StorageLayer& operator=( const StorageLayer& other )
   {
      entities.setLike( other.entities );
      entities = other.entities;
      SubentityStorageBaseType::operator=( other );
      SuperentityStorageBaseType::operator=( other );
      BaseType::operator=( other );
      return *this;
   }

   template< typename Device_ >
   StorageLayer& operator=( const StorageLayer< MeshConfig, Device_, DimensionTag >& other )
   {
      entities.setLike( other.entities );
      entities = other.entities;
      SubentityStorageBaseType::operator=( other );
      SuperentityStorageBaseType::operator=( other );
      BaseType::operator=( other );
      return *this;
   }

   bool save( File& file ) const
   {
      if( ! SubentityStorageBaseType::save( file ) ||
          ! SuperentityStorageBaseType::save( file ) ||
          ! this->entities.save( file ) ||
          ! BaseType::save( file ) )
      {
         std::cerr << "Saving of the mesh entities with dimension " << DimensionTag::value << " failed." << std::endl;
         return false;
      }
      return true;
   }

   bool load( File& file )
   {
      if( ! SubentityStorageBaseType::load( file ) ||
          ! SuperentityStorageBaseType::load( file ) ||
          ! this->entities.load( file ) ||
          ! BaseType::load( file ) )
      {
         std::cerr << "Loading of the mesh entities with dimension " << DimensionTag::value << " failed." << std::endl;
         return false;
      }
      return true;
   }

   void print( std::ostream& str ) const
   {
      str << "The entities with dimension " << DimensionTag::value << " are: " << std::endl;
      for( GlobalIndexType i = 0; i < entities.getSize(); i++ )
         str << i << " " << entities[ i ] << std::endl;
      SubentityStorageBaseType::print( str );
      SuperentityStorageBaseType::print( str );
      str << std::endl;
      BaseType::print( str );
   }

   bool operator==( const StorageLayer& meshLayer ) const
   {
      return ( entities == meshLayer.entities &&
               SubentityStorageBaseType::operator==( meshLayer ) &&
               SuperentityStorageBaseType::operator==( meshLayer ) &&
               BaseType::operator==( meshLayer ) );
   }


   using BaseType::getEntitiesCount;
   __cuda_callable__
   GlobalIndexType getEntitiesCount( DimensionTag ) const
   {
      return this->entities.getSize();
   }

   using BaseType::getEntity;
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

protected:
   using BaseType::setEntitiesCount;
   void setEntitiesCount( DimensionTag, const GlobalIndexType& entitiesCount )
   {
      this->entities.setSize( entitiesCount );
      SubentityStorageBaseType::setEntitiesCount( entitiesCount );
      SuperentityStorageBaseType::setEntitiesCount( entitiesCount );
   }

   StorageArrayType entities;

   // friend class is needed for templated assignment operators
   template< typename MeshConfig_, typename Device_, typename DimensionTag_, bool Storage_ >
   friend class StorageLayer;
};

template< typename MeshConfig,
          typename Device,
          typename DimensionTag >
class StorageLayer< MeshConfig, Device, DimensionTag, false >
   : public StorageLayer< MeshConfig, Device, typename DimensionTag::Decrement  >
{
   using BaseType = StorageLayer< MeshConfig, Device, typename DimensionTag::Decrement >;
public:
   // inherit constructors and assignment operators (including templated versions)
   using BaseType::BaseType;
   using BaseType::operator=;
};

// termination of recursive inheritance (everything is reduced to EntityStorage == false thanks to the WeakEntityStorageTrait)
template< typename MeshConfig,
          typename Device >
class StorageLayer< MeshConfig, Device, DimensionTag< MeshConfig::meshDimension + 1 >, false >
{
protected:
   using DimensionTag     = Meshes::DimensionTag< MeshConfig::meshDimension >;
   using GlobalIndexType  = typename MeshConfig::GlobalIndexType;

   StorageLayer() = default;

   explicit StorageLayer( const StorageLayer& other ) {}

   template< typename Device_ >
   StorageLayer( const StorageLayer< MeshConfig, Device_, DimensionTag >& other ) {}

   StorageLayer& operator=( const StorageLayer& other )
   {
      return *this;
   }

   template< typename Device_ >
   StorageLayer& operator=( const StorageLayer< MeshConfig, Device_, DimensionTag >& other )
   {
      return *this;
   }


   void setEntitiesCount() {}
   void getEntitiesCount() const {}
   void getEntity() const {}

   bool save( File& file ) const
   {
      return true;
   }

   bool load( File& file )
   {
      return true;
   }

   void print( std::ostream& str ) const {}

   bool operator==( const StorageLayer& meshLayer ) const
   {
      return true;
   }
};

} // namespace Meshes
} // namespace TNL
