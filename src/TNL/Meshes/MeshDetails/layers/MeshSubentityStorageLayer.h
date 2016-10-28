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
#include <TNL/Meshes/MeshDimensionsTag.h>
#include <TNL/Meshes/MeshDetails/traits/MeshTraits.h>
#include <TNL/Meshes/MeshDetails/traits/MeshSubentityTraits.h>

namespace TNL {
namespace Meshes {

template< typename MeshConfig,
          typename EntityTopology,
          typename SubdimensionsTag,
          bool SubentityStorage =
               MeshConfig::subentityStorage( EntityTopology(), SubdimensionsTag::value ) >
class MeshSubentityStorageLayer;

template< typename MeshConfig,
          typename EntityTopology >
class MeshSubentityStorageLayers
   : public MeshSubentityStorageLayer< MeshConfig,
                                       EntityTopology,
                                       MeshDimensionsTag< 0 > >
{
   using BaseType = MeshSubentityStorageLayer< MeshConfig,
                                               EntityTopology,
                                               MeshDimensionsTag< 0 > >;
   using MeshTraitsType = MeshTraits< MeshConfig >;

public:
   template< int Subdimensions >
   typename MeshTraitsType::template SubentityTraits< EntityTopology, Subdimensions >::StorageNetworkType&
   getSubentityStorageNetwork()
   {
      static_assert( EntityTopology::dimensions > Subdimensions, "Invalid combination of Dimensions and Subdimensions." );
      return BaseType::getSubentityStorageNetwork( MeshDimensionsTag< EntityTopology::dimensions >(), MeshDimensionsTag< Subdimensions >() );
   }
};

template< typename MeshConfig,
          typename EntityTopology,
          typename SubdimensionsTag >
class MeshSubentityStorageLayer< MeshConfig,
                                 EntityTopology,
                                 SubdimensionsTag,
                                 true >
   : public MeshSubentityStorageLayer< MeshConfig,
                                       EntityTopology,
                                       typename SubdimensionsTag::Increment >
{
   using BaseType = MeshSubentityStorageLayer< MeshConfig,
                                               EntityTopology,
                                               typename SubdimensionsTag::Increment >;

   using MeshTraitsType      = MeshTraits< MeshConfig >;
   using SubentityTraitsType = typename MeshTraitsType::template SubentityTraits< EntityTopology, SubdimensionsTag::value >;

protected:
   using GlobalIndexType    = typename SubentityTraitsType::GlobalIndexType;
   using LocalIndexType     = typename SubentityTraitsType::LocalIndexType;
   using StorageNetworkType = typename SubentityTraitsType::StorageNetworkType;

   bool setNumberOfEntities( const GlobalIndexType& entitiesCount )
   {
      if( ! BaseType::setNumberOfEntities( entitiesCount ) )
         return false;
      this->storageNetwork.setKeysRange( entitiesCount );
      return this->storageNetwork.allocate();
   }

   bool save( File& file ) const
   {
      if( ! BaseType::save( file ) ||
          ! this->storageNetwork.save( file ) )
      {
         std::cerr << "Saving of the entity subentities layer with " << SubdimensionsTag::value << " dimensions failed." << std::endl;
         return false;
      }
      return true;
   }

   bool load( File& file )
   {
      if( ! BaseType::load( file ) ||
          ! this->storageNetwork.load( file ) )
      {
         std::cerr << "Loading of the entity subentities layer with " << SubdimensionsTag::value << " dimensions failed." << std::endl;
         return false;
      }
      return true;
   }

   void print( std::ostream& str ) const
   {
      BaseType::print( str );
      str << "Storage network for subentities with " << SubdimensionsTag::value << " dimensions of entities with " << EntityTopology::dimensions << " dimensions is: " << std::endl;
      str << this->storageNetwork << std::endl;
   }

   bool operator==( const MeshSubentityStorageLayer& layer ) const
   {
      return ( BaseType::operator==( layer ) &&
               storageNetwork == layer.storageNetwork );
   }

   using BaseType::getSubentityStorageNetwork;
   StorageNetworkType& getSubentityStorageNetwork( MeshDimensionsTag< EntityTopology::dimensions >, SubdimensionsTag )
   {
      return this->storageNetwork;
   }

private:
   StorageNetworkType storageNetwork;
};

template< typename MeshConfig,
          typename EntityTopology,
          typename SubdimensionsTag >
class MeshSubentityStorageLayer< MeshConfig,
                                 EntityTopology,
                                 SubdimensionsTag,
                                 false >
   : public MeshSubentityStorageLayer< MeshConfig,
                                       EntityTopology,
                                       typename SubdimensionsTag::Increment >
{
};


template< typename MeshConfig,
          typename EntityTopology >
class MeshSubentityStorageLayer< MeshConfig,
                                 EntityTopology,
                                 MeshDimensionsTag< EntityTopology::dimensions >,
                                 true >
{
   using SubdimensionsTag = MeshDimensionsTag< EntityTopology::dimensions >;

protected:
   /****
    * These methods are due to 'using BaseType::...;' in the derived classes.
    */
   template< typename GlobalIndexType >
   bool setNumberOfEntities( const GlobalIndexType& entitiesCount )
   {
      return true;
   }

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
 
   void getSubentityStorageNetwork( MeshDimensionsTag< EntityTopology::dimensions >, SubdimensionsTag ) {}
};

template< typename MeshConfig,
          typename EntityTopology >
class MeshSubentityStorageLayer< MeshConfig,
                                 EntityTopology,
                                 MeshDimensionsTag< EntityTopology::dimensions >,
                                 false >
{
   using SubdimensionsTag = MeshDimensionsTag< EntityTopology::dimensions >;

protected:
   /****
    * These methods are due to 'using BaseType::...;' in the derived classes.
    */
   template< typename GlobalIndexType >
   bool setNumberOfEntities( const GlobalIndexType& entitiesCount )
   {
      return true;
   }

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
 
   void getSubentityStorageNetwork( MeshDimensionsTag< EntityTopology::dimensions >, SubdimensionsTag ) {}
};

} // namespace Meshes
} // namespace TNL
