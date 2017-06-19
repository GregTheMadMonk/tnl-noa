/***************************************************************************
                          MeshSubentityStorageLayer.h  -  description
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
#include <TNL/Meshes/MeshDetails/traits/MeshSubentityTraits.h>

namespace TNL {
namespace Meshes {

template< typename MeshConfig,
          typename Device,
          typename EntityTopology,
          typename SubdimensionTag,
          bool SubentityStorage =
               MeshConfig::subentityStorage( EntityTopology(), SubdimensionTag::value ) >
class MeshSubentityStorageLayer;

template< typename MeshConfig,
          typename Device,
          typename EntityTopology >
class MeshSubentityStorageLayers
   : public MeshSubentityStorageLayer< MeshConfig, Device, EntityTopology, DimensionTag< 0 > >
{
   using BaseType = MeshSubentityStorageLayer< MeshConfig, Device, EntityTopology, DimensionTag< 0 > >;
   using MeshTraitsType = MeshTraits< MeshConfig, Device >;

public:
   MeshSubentityStorageLayers() = default;
   explicit MeshSubentityStorageLayers( const MeshSubentityStorageLayers& other )
      : BaseType( other )
   {}
   template< typename Device_ >
   MeshSubentityStorageLayers( const MeshSubentityStorageLayers< MeshConfig, Device_, EntityTopology >& other )
      : BaseType( other )
   {}

   template< int Subdimension >
   typename MeshTraitsType::template SubentityTraits< EntityTopology, Subdimension >::StorageNetworkType&
   getSubentityStorageNetwork()
   {
      static_assert( EntityTopology::dimension > Subdimension, "Invalid combination of Dimension and Subdimension." );
      return BaseType::getSubentityStorageNetwork( DimensionTag< Subdimension >() );
   }
};

template< typename MeshConfig,
          typename Device,
          typename EntityTopology,
          typename SubdimensionTag >
class MeshSubentityStorageLayer< MeshConfig,
                                 Device,
                                 EntityTopology,
                                 SubdimensionTag,
                                 true >
   : public MeshSubentityStorageLayer< MeshConfig, Device, EntityTopology, typename SubdimensionTag::Increment >
{
   using BaseType = MeshSubentityStorageLayer< MeshConfig, Device, EntityTopology, typename SubdimensionTag::Increment >;
   using MeshTraitsType      = MeshTraits< MeshConfig, Device >;
   using SubentityTraitsType = typename MeshTraitsType::template SubentityTraits< EntityTopology, SubdimensionTag::value >;

protected:
   using GlobalIndexType    = typename SubentityTraitsType::GlobalIndexType;
   using LocalIndexType     = typename SubentityTraitsType::LocalIndexType;
   using StorageNetworkType = typename SubentityTraitsType::StorageNetworkType;

   MeshSubentityStorageLayer() = default;

   explicit MeshSubentityStorageLayer( const MeshSubentityStorageLayer& other )
   {
      operator=( other );
   }

   template< typename Device_ >
   MeshSubentityStorageLayer( const MeshSubentityStorageLayer< MeshConfig, Device_, EntityTopology, SubdimensionTag >& other )
   {
      operator=( other );
   }

   MeshSubentityStorageLayer& operator=( const MeshSubentityStorageLayer& other )
   {
      BaseType::operator=( other );
      // TODO: throw exception if allocation fails
      storageNetwork.setLike( other.storageNetwork );
      storageNetwork = other.storageNetwork;
      return *this;
   }

   template< typename Device_ >
   MeshSubentityStorageLayer& operator=( const MeshSubentityStorageLayer< MeshConfig, Device_, EntityTopology, SubdimensionTag >& other )
   {
      BaseType::operator=( other );
      // TODO: throw exception if allocation fails
      storageNetwork.setLike( other.storageNetwork );
      storageNetwork = other.storageNetwork;
      return *this;
   }


   void setNumberOfEntities( const GlobalIndexType& entitiesCount )
   {
      BaseType::setNumberOfEntities( entitiesCount );
      this->storageNetwork.setKeysRange( entitiesCount );
      this->storageNetwork.allocate();
   }

   bool save( File& file ) const
   {
      if( ! BaseType::save( file ) ||
          ! this->storageNetwork.save( file ) )
      {
         std::cerr << "Saving of the entity subentities layer with " << SubdimensionTag::value << " dimension failed." << std::endl;
         return false;
      }
      return true;
   }

   bool load( File& file )
   {
      if( ! BaseType::load( file ) ||
          ! this->storageNetwork.load( file ) )
      {
         std::cerr << "Loading of the entity subentities layer with " << SubdimensionTag::value << " dimension failed." << std::endl;
         return false;
      }
      return true;
   }

   void print( std::ostream& str ) const
   {
      BaseType::print( str );
      str << "Storage network for subentities with dimension " << SubdimensionTag::value << " of entities with dimension " << EntityTopology::dimension << " is: " << std::endl;
      str << this->storageNetwork << std::endl;
   }

   bool operator==( const MeshSubentityStorageLayer& layer ) const
   {
      return ( BaseType::operator==( layer ) &&
               storageNetwork == layer.storageNetwork );
   }

   using BaseType::getSubentityStorageNetwork;
   StorageNetworkType& getSubentityStorageNetwork( SubdimensionTag )
   {
      return this->storageNetwork;
   }

private:
   StorageNetworkType storageNetwork;

   // friend class is needed for templated assignment operators
   template< typename MeshConfig_, typename Device_, typename EntityTopology_, typename SubdimensionTag_, bool Storage_ >
   friend class MeshSubentityStorageLayer;
};

template< typename MeshConfig,
          typename Device,
          typename EntityTopology,
          typename SubdimensionTag >
class MeshSubentityStorageLayer< MeshConfig,
                                 Device,
                                 EntityTopology,
                                 SubdimensionTag,
                                 false >
   : public MeshSubentityStorageLayer< MeshConfig, Device, EntityTopology, typename SubdimensionTag::Increment >
{
public:
   using BaseType = MeshSubentityStorageLayer< MeshConfig, Device, EntityTopology, typename SubdimensionTag::Increment >;

   MeshSubentityStorageLayer() = default;
   explicit MeshSubentityStorageLayer( const MeshSubentityStorageLayer& other )
      : BaseType( other )
   {}
   template< typename Device_ >
   MeshSubentityStorageLayer( const MeshSubentityStorageLayer< MeshConfig, Device_, EntityTopology, SubdimensionTag >& other )
      : BaseType( other )
   {}
   template< typename Device_ >
   MeshSubentityStorageLayer& operator=( const MeshSubentityStorageLayer< MeshConfig, Device_, EntityTopology, SubdimensionTag >& other )
   { return *this; }
};


template< typename MeshConfig,
          typename Device,
          typename EntityTopology >
class MeshSubentityStorageLayer< MeshConfig,
                                 Device,
                                 EntityTopology,
                                 DimensionTag< EntityTopology::dimension >,
                                 true >
{
   using SubdimensionTag = DimensionTag< EntityTopology::dimension >;

protected:
   MeshSubentityStorageLayer() = default;
   explicit MeshSubentityStorageLayer( const MeshSubentityStorageLayer& other ) {}
   template< typename Device_ >
   MeshSubentityStorageLayer( const MeshSubentityStorageLayer< MeshConfig, Device_, EntityTopology, SubdimensionTag >& other ) {}
   template< typename Device_ >
   MeshSubentityStorageLayer& operator=( const MeshSubentityStorageLayer< MeshConfig, Device_, EntityTopology, SubdimensionTag >& other ) { return *this; }

   /****
    * These methods are due to 'using BaseType::...;' in the derived classes.
    */
   template< typename GlobalIndexType >
   void setNumberOfEntities( const GlobalIndexType& entitiesCount ) {}

   void print( std::ostream& str ) const {}

   bool operator==( const MeshSubentityStorageLayer& layer ) const
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
 
   void getSubentityStorageNetwork( SubdimensionTag ) {}
};

template< typename MeshConfig,
          typename Device,
          typename EntityTopology >
class MeshSubentityStorageLayer< MeshConfig,
                                 Device,
                                 EntityTopology,
                                 DimensionTag< EntityTopology::dimension >,
                                 false >
{
   using SubdimensionTag = DimensionTag< EntityTopology::dimension >;

protected:
   MeshSubentityStorageLayer() = default;
   explicit MeshSubentityStorageLayer( const MeshSubentityStorageLayer& other ) {}
   template< typename Device_ >
   MeshSubentityStorageLayer( const MeshSubentityStorageLayer< MeshConfig, Device_, EntityTopology, SubdimensionTag >& other ) {}
   template< typename Device_ >
   MeshSubentityStorageLayer& operator=( const MeshSubentityStorageLayer< MeshConfig, Device_, EntityTopology, SubdimensionTag >& other ) { return *this; }

   /****
    * These methods are due to 'using BaseType::...;' in the derived classes.
    */
   template< typename GlobalIndexType >
   void setNumberOfEntities( const GlobalIndexType& entitiesCount ) {}

   void print( std::ostream& str ) const {}

   bool operator==( const MeshSubentityStorageLayer& layer ) const
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
 
   void getSubentityStorageNetwork( SubdimensionTag ) {}
};

} // namespace Meshes
} // namespace TNL
