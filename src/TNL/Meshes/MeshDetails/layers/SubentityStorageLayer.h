/***************************************************************************
                          SubentityStorageLayer.h  -  description
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
#include <TNL/Meshes/DimensionTag.h>
#include <TNL/Meshes/MeshDetails/traits/MeshTraits.h>
#include <TNL/Meshes/MeshDetails/traits/WeakStorageTraits.h>

namespace TNL {
namespace Meshes {

template< typename MeshConfig,
          typename Device,
          typename EntityTopology,
          typename SubdimensionTag,
          bool SubentityStorage = WeakSubentityStorageTrait< MeshConfig, Device, EntityTopology, SubdimensionTag >::storageEnabled >
class SubentityStorageLayer;

template< typename MeshConfig,
          typename Device,
          typename EntityTopology >
class SubentityStorageLayerFamily
   : public SubentityStorageLayer< MeshConfig, Device, EntityTopology, DimensionTag< 0 > >
{
   using BaseType = SubentityStorageLayer< MeshConfig, Device, EntityTopology, DimensionTag< 0 > >;
   using MeshTraitsType = MeshTraits< MeshConfig, Device >;

public:
   // inherit constructors and assignment operators (including templated versions)
   using BaseType::BaseType;
   using BaseType::operator=;

protected:
   template< int Subdimension >
   __cuda_callable__
   typename MeshTraitsType::template SubentityTraits< EntityTopology, Subdimension >::StorageNetworkType&
   getSubentityStorageNetwork()
   {
      static_assert( EntityTopology::dimension > Subdimension, "Invalid combination of Dimension and Subdimension." );
      return BaseType::getSubentityStorageNetwork( DimensionTag< Subdimension >() );
   }

   template< int Subdimension >
   __cuda_callable__
   const typename MeshTraitsType::template SubentityTraits< EntityTopology, Subdimension >::StorageNetworkType&
   getSubentityStorageNetwork() const
   {
      static_assert( EntityTopology::dimension > Subdimension, "Invalid combination of Dimension and Subdimension." );
      return BaseType::getSubentityStorageNetwork( DimensionTag< Subdimension >() );
   }
};

template< typename MeshConfig,
          typename Device,
          typename EntityTopology,
          typename SubdimensionTag >
class SubentityStorageLayer< MeshConfig,
                             Device,
                             EntityTopology,
                             SubdimensionTag,
                             true >
   : public SubentityStorageLayer< MeshConfig, Device, EntityTopology, typename SubdimensionTag::Increment >
{
   using BaseType = SubentityStorageLayer< MeshConfig, Device, EntityTopology, typename SubdimensionTag::Increment >;
   using MeshTraitsType      = MeshTraits< MeshConfig, Device >;
   using SubentityTraitsType = typename MeshTraitsType::template SubentityTraits< EntityTopology, SubdimensionTag::value >;

protected:
   using GlobalIndexType    = typename MeshTraitsType::GlobalIndexType;
   using StorageNetworkType = typename SubentityTraitsType::StorageNetworkType;

   SubentityStorageLayer() = default;

   explicit SubentityStorageLayer( const SubentityStorageLayer& other )
   {
      operator=( other );
   }

   template< typename Device_ >
   SubentityStorageLayer( const SubentityStorageLayer< MeshConfig, Device_, EntityTopology, SubdimensionTag >& other )
   {
      operator=( other );
   }

   SubentityStorageLayer& operator=( const SubentityStorageLayer& other )
   {
      BaseType::operator=( other );
      storageNetwork.setLike( other.storageNetwork );
      storageNetwork = other.storageNetwork;
      return *this;
   }

   template< typename Device_ >
   SubentityStorageLayer& operator=( const SubentityStorageLayer< MeshConfig, Device_, EntityTopology, SubdimensionTag >& other )
   {
      BaseType::operator=( other );
      storageNetwork.setLike( other.storageNetwork );
      storageNetwork = other.storageNetwork;
      return *this;
   }


   void save( File& file ) const
   {
      BaseType::save( file );
      this->storageNetwork.save( file );
   }

   void load( File& file )
   {
      BaseType::load( file );
      this->storageNetwork.load( file );
   }

   void print( std::ostream& str ) const
   {
      BaseType::print( str );
      str << "Storage network for subentities with dimension " << SubdimensionTag::value << " of entities with dimension " << EntityTopology::dimension << " is: " << std::endl;
      str << this->storageNetwork << std::endl;
   }

   bool operator==( const SubentityStorageLayer& layer ) const
   {
      return ( BaseType::operator==( layer ) &&
               storageNetwork == layer.storageNetwork );
   }

protected:
   void setEntitiesCount( const GlobalIndexType& entitiesCount )
   {
      BaseType::setEntitiesCount( entitiesCount );
      this->storageNetwork.setKeysRange( entitiesCount );
      this->storageNetwork.allocate();
   }

   using BaseType::getSubentityStorageNetwork;
   __cuda_callable__
   StorageNetworkType& getSubentityStorageNetwork( SubdimensionTag )
   {
      return this->storageNetwork;
   }

   __cuda_callable__
   const StorageNetworkType& getSubentityStorageNetwork( SubdimensionTag ) const
   {
      return this->storageNetwork;
   }

private:
   StorageNetworkType storageNetwork;

   // friend class is needed for templated assignment operators
   template< typename MeshConfig_, typename Device_, typename EntityTopology_, typename SubdimensionTag_, bool Storage_ >
   friend class SubentityStorageLayer;
};

template< typename MeshConfig,
          typename Device,
          typename EntityTopology,
          typename SubdimensionTag >
class SubentityStorageLayer< MeshConfig,
                             Device,
                             EntityTopology,
                             SubdimensionTag,
                             false >
   : public SubentityStorageLayer< MeshConfig, Device, EntityTopology, typename SubdimensionTag::Increment >
{
   using BaseType = SubentityStorageLayer< MeshConfig, Device, EntityTopology, typename SubdimensionTag::Increment >;
public:
   // inherit constructors and assignment operators (including templated versions)
   using BaseType::BaseType;
   using BaseType::operator=;
};

// termination of recursive inheritance (everything is reduced to EntityStorage == false thanks to the WeakSubentityStorageTrait)
template< typename MeshConfig,
          typename Device,
          typename EntityTopology >
class SubentityStorageLayer< MeshConfig,
                             Device,
                             EntityTopology,
                             DimensionTag< EntityTopology::dimension >,
                             false >
{
   using SubdimensionTag = DimensionTag< EntityTopology::dimension >;

protected:
   using GlobalIndexType = typename MeshConfig::GlobalIndexType;

   SubentityStorageLayer() = default;
   explicit SubentityStorageLayer( const SubentityStorageLayer& other ) {}
   template< typename Device_ >
   SubentityStorageLayer( const SubentityStorageLayer< MeshConfig, Device_, EntityTopology, SubdimensionTag >& other ) {}
   template< typename Device_ >
   SubentityStorageLayer& operator=( const SubentityStorageLayer< MeshConfig, Device_, EntityTopology, SubdimensionTag >& other ) { return *this; }

   void setEntitiesCount( GlobalIndexType entitiesCount ) {}

   void print( std::ostream& str ) const {}

   bool operator==( const SubentityStorageLayer& layer ) const
   {
      return true;
   }

   void save( File& file ) const {}
   void load( File& file ) {}

   void getSubentityStorageNetwork( SubdimensionTag ) {}
};

} // namespace Meshes
} // namespace TNL
