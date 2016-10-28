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
#include <TNL/Meshes/MeshDimensionTag.h>
#include <TNL/Meshes/MeshDetails/traits/MeshTraits.h>
#include <TNL/Meshes/MeshDetails/traits/MeshSuperentityTraits.h>

namespace TNL {
namespace Meshes {

template< typename MeshConfig,
          typename EntityTopology,
          typename SuperdimensionsTag,
          bool SuperentityStorage =
               MeshTraits< MeshConfig >::template SuperentityTraits< EntityTopology, SuperdimensionsTag::value >::storageEnabled >
class MeshSuperentityStorageLayer;

template< typename MeshConfig,
          typename EntityTopology >
class MeshSuperentityStorageLayers
   : public MeshSuperentityStorageLayer< MeshConfig,
                                         EntityTopology,
                                         MeshDimensionsTag< MeshTraits< MeshConfig >::meshDimensions > >
{
   using BaseType = MeshSuperentityStorageLayer< MeshConfig,
                                                 EntityTopology,
                                                 MeshDimensionsTag< MeshTraits< MeshConfig >::meshDimensions > >;
   using MeshTraitsType = MeshTraits< MeshConfig >;

public:
   template< int Superdimensions >
   typename MeshTraitsType::template SuperentityTraits< EntityTopology, Superdimensions >::StorageNetworkType&
   getSuperentityStorageNetwork( MeshDimensionsTag< EntityTopology::dimensions > )
   {
      return BaseType::getSuperentityStorageNetwork( MeshDimensionsTag< Superdimensions >() );
   }
};

template< typename MeshConfig,
          typename EntityTopology,
          typename SuperdimensionsTag >
class MeshSuperentityStorageLayer< MeshConfig, EntityTopology, SuperdimensionsTag, true >
   : public MeshSuperentityStorageLayer< MeshConfig, EntityTopology, typename SuperdimensionsTag::Decrement >
{
   using BaseType = MeshSuperentityStorageLayer< MeshConfig, EntityTopology, typename SuperdimensionsTag::Decrement >;

   using MeshTraitsType        = MeshTraits< MeshConfig >;
   using SuperentityTraitsType = typename MeshTraitsType::template SuperentityTraits< EntityTopology, SuperdimensionsTag::value >;

protected:
   using GlobalIndexType    = typename SuperentityTraitsType::GlobalIndexType;
   using LocalIndexType     = typename SuperentityTraitsType::LocalIndexType;
   using StorageNetworkType = typename SuperentityTraitsType::StorageNetworkType;
 
   MeshSuperentityStorageLayer& operator=( const MeshSuperentityStorageLayer& layer ) = delete;

   bool setNumberOfEntities( const GlobalIndexType& entitiesCount )
   {
      if( ! BaseType::setNumberOfEntities( entitiesCount ) )
         return false;
      this->storageNetwork.setKeysRange( entitiesCount );
      return true;
   }

   bool save( File& file ) const
   {
      if( ! BaseType::save( file ) ||
          ! this->storageNetwork.save( file ) )
      {
         std::cerr << "Saving of the entity superentities layer with " << SuperdimensionsTag::value << " dimensions failed." << std::endl;
         return false;
      }
      return true;
   }

   bool load( File& file )
   {
      if( ! BaseType::load( file ) ||
          ! this->storageNetwork.load( file ) )
      {
         std::cerr << "Loading of the entity superentities layer with " << SuperdimensionsTag::value << " dimensions failed." << std::endl;
         return false;
      }
      return true;
   }

   void print( std::ostream& str ) const
   {
      BaseType::print( str );
      str << "Storage network for superentities with " << SuperdimensionsTag::value << " dimensions of entities with " << EntityTopology::dimensions << " dimensions is: " << std::endl;
      str << this->storageNetwork << std::endl;
   }

   bool operator==( const MeshSuperentityStorageLayer& layer ) const
   {
      return ( BaseType::operator==( layer ) &&
               storageNetwork == layer.storageNetwork );
   }

   using BaseType::getSuperentityStorageNetwork;
   StorageNetworkType& getSuperentityStorageNetwork( SuperdimensionsTag )
   {
      return this->storageNetwork;
   }

private:
   StorageNetworkType storageNetwork;
};

template< typename MeshConfig,
          typename EntityTopology,
          typename SuperdimensionsTag >
class MeshSuperentityStorageLayer< MeshConfig, EntityTopology, SuperdimensionsTag, false >
   : public MeshSuperentityStorageLayer< MeshConfig, EntityTopology, typename SuperdimensionsTag::Decrement >
{
};

template< typename MeshConfig,
          typename EntityTopology >
class MeshSuperentityStorageLayer< MeshConfig, EntityTopology, MeshDimensionTag< EntityTopology::dimensions >, false >
{
   using SuperdimensionsTag = MeshDimensionsTag< EntityTopology::dimensions >;
   using MeshTraitsType = MeshTraits< MeshConfig >;
   using SuperentityTraitsType = typename MeshTraitsType::template SuperentityTraits< EntityTopology, SuperdimensionsTag::value >;
   using ThisType = MeshSuperentityStorageLayer< MeshConfig, EntityTopology, SuperdimensionsTag, false >;

protected:
   using GlobalIndexType    = typename SuperentityTraitsType::GlobalIndexType;
   using LocalIndexType     = typename SuperentityTraitsType::LocalIndexType;
   using StorageNetworkType = typename SuperentityTraitsType::StorageNetworkType;
 
   /****
    * These methods are due to 'using BaseType::...;' in the derived classes.
    */
   bool setNumberOfEntities( const GlobalIndexType& entitiesCount )
   {
      return true;
   }

   void print( std::ostream& str ) const {}

   bool operator==( const ThisType& layer ) const
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
 
   void getSuperentityStorageNetwork( SuperdimensionsTag ) {}
};

template< typename MeshConfig,
          typename EntityTopology >
class MeshSuperentityStorageLayer< MeshConfig,
                                   EntityTopology,
                                   MeshDimensionsTag< EntityTopology::dimensions >,
                                   true >
{
   using SuperdimensionsTag = MeshDimensionsTag< EntityTopology::dimensions >;
   using MeshTraitsType = MeshTraits< MeshConfig >;
   using SuperentityTraitsType = typename MeshTraitsType::template SuperentityTraits< EntityTopology, SuperdimensionsTag::value >;
   using ThisType = MeshSuperentityStorageLayer< MeshConfig, EntityTopology, SuperdimensionsTag, true >;

protected:
   using GlobalIndexType    = typename SuperentityTraitsType::GlobalIndexType;
   using LocalIndexType     = typename SuperentityTraitsType::LocalIndexType;
   using StorageNetworkType = typename SuperentityTraitsType::StorageNetworkType;
 
   /****
    * These methods are due to 'using BaseType::...;' in the derived classes.
    */
   bool setNumberOfEntities( const GlobalIndexType& entitiesCount )
   {
      return true;
   }

   void print( std::ostream& str ) const {}

   bool operator==( const ThisType& layer ) const
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
 
   void getSuperentityStorageNetwork( SuperdimensionsTag ) {}
};

} // namespace Meshes
} // namespace TNL
