/***************************************************************************
                          SuperentityStorageLayer.h  -  description
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
class SuperentityStorageLayer;

template< typename MeshConfig,
          typename Device,
          typename EntityTopology >
class SuperentityStorageLayerFamily
   : public SuperentityStorageLayer< MeshConfig,
                                     Device,
                                     EntityTopology,
                                     DimensionTag< MeshTraits< MeshConfig, Device >::meshDimension > >
{
   using BaseType = SuperentityStorageLayer< MeshConfig,
                                             Device,
                                             EntityTopology,
                                             DimensionTag< MeshTraits< MeshConfig, Device >::meshDimension > >;
   using MeshTraitsType = MeshTraits< MeshConfig, Device >;

public:
   // inherit constructors and assignment operators (including templated versions)
   using BaseType::BaseType;
   using BaseType::operator=;

protected:
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
class SuperentityStorageLayer< MeshConfig, Device, EntityTopology, SuperdimensionTag, true >
   : public SuperentityStorageLayer< MeshConfig, Device, EntityTopology, typename SuperdimensionTag::Decrement >
{
   using BaseType = SuperentityStorageLayer< MeshConfig, Device, EntityTopology, typename SuperdimensionTag::Decrement >;
   using MeshTraitsType        = MeshTraits< MeshConfig, Device >;
   using SuperentityTraitsType = typename MeshTraitsType::template SuperentityTraits< EntityTopology, SuperdimensionTag::value >;

protected:
   using GlobalIndexType    = typename MeshTraitsType::GlobalIndexType;
   using StorageNetworkType = typename SuperentityTraitsType::StorageNetworkType;
 
   SuperentityStorageLayer() = default;

   explicit SuperentityStorageLayer( const SuperentityStorageLayer& other )
   {
      operator=( other );
   }

   template< typename Device_ >
   SuperentityStorageLayer( const SuperentityStorageLayer< MeshConfig, Device_, EntityTopology, SuperdimensionTag >& other )
   {
      operator=( other );
   }

   SuperentityStorageLayer& operator=( const SuperentityStorageLayer& other )
   {
      BaseType::operator=( other );
      storageNetwork.setLike( other.storageNetwork );
      storageNetwork = other.storageNetwork;
      return *this;
   }

   template< typename Device_ >
   SuperentityStorageLayer& operator=( const SuperentityStorageLayer< MeshConfig, Device_, EntityTopology, SuperdimensionTag >& other )
   {
      BaseType::operator=( other );
      storageNetwork.setLike( other.storageNetwork );
      storageNetwork = other.storageNetwork;
      return *this;
   }


   bool save( File& file ) const
   {
      try
      {
         BaseType::save( file );
         this->storageNetwork.save( file );
      }
      catch(...)
      {
         std::cerr << "Saving of the entity superentities layer with " << SuperdimensionTag::value << " dimension failed." << std::endl;
         return false;
      }
      return true;
   }

   bool load( File& file )
   {
      try
      {
         BaseType::load( file );
         this->storageNetwork.load( file );
      }
      catch(...)
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

   bool operator==( const SuperentityStorageLayer& layer ) const
   {
      return ( BaseType::operator==( layer ) &&
               storageNetwork == layer.storageNetwork );
   }

protected:
   void setEntitiesCount( const GlobalIndexType& entitiesCount )
   {
      BaseType::setEntitiesCount( entitiesCount );
      this->storageNetwork.setKeysRange( entitiesCount );
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
   friend class SuperentityStorageLayer;
};

template< typename MeshConfig,
          typename Device,
          typename EntityTopology,
          typename SuperdimensionTag >
class SuperentityStorageLayer< MeshConfig, Device, EntityTopology, SuperdimensionTag, false >
   : public SuperentityStorageLayer< MeshConfig, Device, EntityTopology, typename SuperdimensionTag::Decrement >
{
   using BaseType = SuperentityStorageLayer< MeshConfig, Device, EntityTopology, typename SuperdimensionTag::Decrement >;
public:
   // inherit constructors and assignment operators (including templated versions)
   using BaseType::BaseType;
   using BaseType::operator=;
};

// termination of recursive inheritance (everything is reduced to EntityStorage == false thanks to the WeakSuperentityStorageTrait)
template< typename MeshConfig,
          typename Device,
          typename EntityTopology >
class SuperentityStorageLayer< MeshConfig, Device, EntityTopology, DimensionTag< EntityTopology::dimension >, false >
{
   using SuperdimensionTag = DimensionTag< EntityTopology::dimension >;

protected:
   using GlobalIndexType = typename MeshConfig::GlobalIndexType;
 
   SuperentityStorageLayer() = default;
   explicit SuperentityStorageLayer( const SuperentityStorageLayer& other ) {}
   template< typename Device_ >
   SuperentityStorageLayer( const SuperentityStorageLayer< MeshConfig, Device_, EntityTopology, SuperdimensionTag >& other ) {}
   template< typename Device_ >
   SuperentityStorageLayer& operator=( const SuperentityStorageLayer< MeshConfig, Device_, EntityTopology, SuperdimensionTag >& other ) { return *this; }

   void setEntitiesCount( GlobalIndexType entitiesCount ) {}

   void print( std::ostream& str ) const {}

   bool operator==( const SuperentityStorageLayer& layer ) const
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
