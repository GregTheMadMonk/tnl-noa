/***************************************************************************
                          MeshSuperentityStorageLayer.h  -  description
                             -------------------
    begin                : Feb 13, 2014
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
#include <TNL/Meshes/DimensionTag.h>
#include <TNL/Meshes/MeshDetails/traits/MeshTraits.h>
#include <TNL/Meshes/MeshDetails/traits/WeakStorageTraits.h>

namespace TNL {
namespace Meshes {

template< typename MeshConfig,
          typename Device,
          typename EntityTopology,
          typename SuperdimensionTag,
          bool SuperentityStorage = WeakSuperentityStorageTrait< MeshConfig, Device, EntityTopology, SuperdimensionTag >::storageEnabled >
class MeshSuperentityStorageLayer;

template< typename MeshConfig,
          typename Device,
          typename EntityTopology >
class MeshSuperentityStorageLayers
   : public MeshSuperentityStorageLayer< MeshConfig,
                                         Device,
                                         EntityTopology,
                                         DimensionTag< MeshTraits< MeshConfig, Device >::meshDimension > >
{
   using BaseType = MeshSuperentityStorageLayer< MeshConfig,
                                                 Device,
                                                 EntityTopology,
                                                 DimensionTag< MeshTraits< MeshConfig, Device >::meshDimension > >;
   using MeshTraitsType = MeshTraits< MeshConfig, Device >;

public:
   MeshSuperentityStorageLayers() = default;
   explicit MeshSuperentityStorageLayers( const MeshSuperentityStorageLayers& other )
      : BaseType( other )
   {}
   template< typename Device_ >
   MeshSuperentityStorageLayers( const MeshSuperentityStorageLayers< MeshConfig, Device_, EntityTopology >& other )
      : BaseType( other )
   {}

   template< int Superdimension >
   typename MeshTraitsType::template SuperentityTraits< EntityTopology, Superdimension >::StorageNetworkType&
   getSuperentityStorageNetwork()
   {
      static_assert( EntityTopology::dimension < Superdimension, "Invalid combination of Dimension and Superdimension." );
      return BaseType::getSuperentityStorageNetwork( DimensionTag< Superdimension >() );
   }
};

template< typename MeshConfig,
          typename Device,
          typename EntityTopology,
          typename SuperdimensionTag >
class MeshSuperentityStorageLayer< MeshConfig, Device, EntityTopology, SuperdimensionTag, true >
   : public MeshSuperentityStorageLayer< MeshConfig, Device, EntityTopology, typename SuperdimensionTag::Decrement >
{
   using BaseType = MeshSuperentityStorageLayer< MeshConfig, Device, EntityTopology, typename SuperdimensionTag::Decrement >;
   using MeshTraitsType        = MeshTraits< MeshConfig, Device >;
   using SuperentityTraitsType = typename MeshTraitsType::template SuperentityTraits< EntityTopology, SuperdimensionTag::value >;

protected:
   using GlobalIndexType    = typename SuperentityTraitsType::GlobalIndexType;
   using StorageNetworkType = typename SuperentityTraitsType::StorageNetworkType;
 
   MeshSuperentityStorageLayer() = default;

   explicit MeshSuperentityStorageLayer( const MeshSuperentityStorageLayer& other )
   {
      operator=( other );
   }

   template< typename Device_ >
   MeshSuperentityStorageLayer( const MeshSuperentityStorageLayer< MeshConfig, Device_, EntityTopology, SuperdimensionTag >& other )
   {
      operator=( other );
   }

   MeshSuperentityStorageLayer& operator=( const MeshSuperentityStorageLayer& other )
   {
      BaseType::operator=( other );
      // TODO: throw exception if allocation fails
      storageNetwork.setLike( other.storageNetwork );
      storageNetwork = other.storageNetwork;
      return *this;
   }

   template< typename Device_ >
   MeshSuperentityStorageLayer& operator=( const MeshSuperentityStorageLayer< MeshConfig, Device_, EntityTopology, SuperdimensionTag >& other )
   {
      BaseType::operator=( other );
      // TODO: throw exception if allocation fails
      storageNetwork.setLike( other.storageNetwork );
      storageNetwork = other.storageNetwork;
      return *this;
   }


   void setEntitiesCount( const GlobalIndexType& entitiesCount )
   {
      BaseType::setEntitiesCount( entitiesCount );
      this->storageNetwork.setKeysRange( entitiesCount );
   }

   bool save( File& file ) const
   {
      if( ! BaseType::save( file ) ||
          ! this->storageNetwork.save( file ) )
      {
         std::cerr << "Saving of the entity superentities layer with " << SuperdimensionTag::value << " dimension failed." << std::endl;
         return false;
      }
      return true;
   }

   bool load( File& file )
   {
      if( ! BaseType::load( file ) ||
          ! this->storageNetwork.load( file ) )
      {
         std::cerr << "Loading of the entity superentities layer with " << SuperdimensionTag::value << " dimension failed." << std::endl;
         return false;
      }
      return true;
   }

   void print( std::ostream& str ) const
   {
      BaseType::print( str );
      str << "Storage network for superentities with dimension " << SuperdimensionTag::value << " of entities with dimension " << EntityTopology::dimension << " is: " << std::endl;
      str << this->storageNetwork << std::endl;
   }

   bool operator==( const MeshSuperentityStorageLayer& layer ) const
   {
      return ( BaseType::operator==( layer ) &&
               storageNetwork == layer.storageNetwork );
   }

   using BaseType::getSuperentityStorageNetwork;
   StorageNetworkType& getSuperentityStorageNetwork( SuperdimensionTag )
   {
      return this->storageNetwork;
   }

private:
   StorageNetworkType storageNetwork;

   // friend class is needed for templated assignment operators
   template< typename MeshConfig_, typename Device_, typename EntityTopology_, typename SuperdimensionTag_, bool Storage_ >
   friend class MeshSuperentityStorageLayer;
};

template< typename MeshConfig,
          typename Device,
          typename EntityTopology,
          typename SuperdimensionTag >
class MeshSuperentityStorageLayer< MeshConfig, Device, EntityTopology, SuperdimensionTag, false >
   : public MeshSuperentityStorageLayer< MeshConfig, Device, EntityTopology, typename SuperdimensionTag::Decrement >
{
public:
   using BaseType = MeshSuperentityStorageLayer< MeshConfig, Device, EntityTopology, typename SuperdimensionTag::Decrement >;

   MeshSuperentityStorageLayer() = default;
   explicit MeshSuperentityStorageLayer( const MeshSuperentityStorageLayer& other )
      : BaseType( other )
   {}
   template< typename Device_ >
   MeshSuperentityStorageLayer( const MeshSuperentityStorageLayer< MeshConfig, Device_, EntityTopology, SuperdimensionTag >& other )
      : BaseType( other )
   {}
   template< typename Device_ >
   MeshSuperentityStorageLayer& operator=( const MeshSuperentityStorageLayer< MeshConfig, Device_, EntityTopology, SuperdimensionTag >& other )
   { return *this; }
};

template< typename MeshConfig,
          typename Device,
          typename EntityTopology >
class MeshSuperentityStorageLayer< MeshConfig, Device, EntityTopology, DimensionTag< EntityTopology::dimension >, false >
{
   using SuperdimensionTag = DimensionTag< EntityTopology::dimension >;

protected:
   using GlobalIndexType = typename MeshConfig::GlobalIndexType;
 
   MeshSuperentityStorageLayer() = default;
   explicit MeshSuperentityStorageLayer( const MeshSuperentityStorageLayer& other ) {}
   template< typename Device_ >
   MeshSuperentityStorageLayer( const MeshSuperentityStorageLayer< MeshConfig, Device_, EntityTopology, SuperdimensionTag >& other ) {}
   template< typename Device_ >
   MeshSuperentityStorageLayer& operator=( const MeshSuperentityStorageLayer< MeshConfig, Device_, EntityTopology, SuperdimensionTag >& other ) { return *this; }

   /****
    * These methods are due to 'using BaseType::...;' in the derived classes.
    */
   void setEntitiesCount( const GlobalIndexType& entitiesCount ) {}

   void print( std::ostream& str ) const {}

   bool operator==( const MeshSuperentityStorageLayer& layer ) const
   {
      return true;
   }

   bool save( File& file ) const
   {
      return true;
   }

   bool load( File& file )
   {
      return true;
   }
 
   void getSuperentityStorageNetwork( SuperdimensionTag ) {}
};

template< typename MeshConfig,
          typename Device,
          typename EntityTopology >
class MeshSuperentityStorageLayer< MeshConfig,
                                   Device,
                                   EntityTopology,
                                   DimensionTag< EntityTopology::dimension >,
                                   true >
{
   using SuperdimensionTag = DimensionTag< EntityTopology::dimension >;

protected:
   using GlobalIndexType = typename MeshConfig::GlobalIndexType;
 
   MeshSuperentityStorageLayer() = default;
   explicit MeshSuperentityStorageLayer( const MeshSuperentityStorageLayer& other ) {}
   template< typename Device_ >
   MeshSuperentityStorageLayer( const MeshSuperentityStorageLayer< MeshConfig, Device_, EntityTopology, SuperdimensionTag >& other ) {}
   template< typename Device_ >
   MeshSuperentityStorageLayer& operator=( const MeshSuperentityStorageLayer< MeshConfig, Device_, EntityTopology, SuperdimensionTag >& other ) { return *this; }

   /****
    * These methods are due to 'using BaseType::...;' in the derived classes.
    */
   void setEntitiesCount( const GlobalIndexType& entitiesCount ) {}

   void print( std::ostream& str ) const {}

   bool operator==( const MeshSuperentityStorageLayer& layer ) const
   {
      return true;
   }

   bool save( File& file ) const
   {
      return true;
   }

   bool load( File& file )
   {
      return true;
   }
 
   void getSuperentityStorageNetwork( SuperdimensionTag ) {}
};

} // namespace Meshes
} // namespace TNL
