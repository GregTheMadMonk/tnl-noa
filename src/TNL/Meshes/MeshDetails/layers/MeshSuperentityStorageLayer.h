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
#include <TNL/Meshes/MeshDetails/traits/MeshSuperentityTraits.h>

namespace TNL {
namespace Meshes {

template< typename MeshConfig,
          typename EntityTopology,
          typename SuperdimensionTag,
          bool SuperentityStorage =
               MeshTraits< MeshConfig >::template SuperentityTraits< EntityTopology, SuperdimensionTag::value >::storageEnabled >
class MeshSuperentityStorageLayer;

template< typename MeshConfig,
          typename EntityTopology >
class MeshSuperentityStorageLayers
   : public MeshSuperentityStorageLayer< MeshConfig,
                                         EntityTopology,
                                         DimensionTag< MeshTraits< MeshConfig >::meshDimension > >
{
   using BaseType = MeshSuperentityStorageLayer< MeshConfig,
                                                 EntityTopology,
                                                 DimensionTag< MeshTraits< MeshConfig >::meshDimension > >;
   using MeshTraitsType = MeshTraits< MeshConfig >;

public:
   template< int Superdimension >
   typename MeshTraitsType::template SuperentityTraits< EntityTopology, Superdimension >::StorageNetworkType&
   getSuperentityStorageNetwork()
   {
      static_assert( EntityTopology::dimension < Superdimension, "Invalid combination of Dimension and Superdimension." );
      return BaseType::getSuperentityStorageNetwork( DimensionTag< Superdimension >() );
   }
};

template< typename MeshConfig,
          typename EntityTopology,
          typename SuperdimensionTag >
class MeshSuperentityStorageLayer< MeshConfig, EntityTopology, SuperdimensionTag, true >
   : public MeshSuperentityStorageLayer< MeshConfig, EntityTopology, typename SuperdimensionTag::Decrement >
{
   using BaseType = MeshSuperentityStorageLayer< MeshConfig, EntityTopology, typename SuperdimensionTag::Decrement >;

   using MeshTraitsType        = MeshTraits< MeshConfig >;
   using SuperentityTraitsType = typename MeshTraitsType::template SuperentityTraits< EntityTopology, SuperdimensionTag::value >;

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
};

template< typename MeshConfig,
          typename EntityTopology,
          typename SuperdimensionTag >
class MeshSuperentityStorageLayer< MeshConfig, EntityTopology, SuperdimensionTag, false >
   : public MeshSuperentityStorageLayer< MeshConfig, EntityTopology, typename SuperdimensionTag::Decrement >
{
};

template< typename MeshConfig,
          typename EntityTopology >
class MeshSuperentityStorageLayer< MeshConfig, EntityTopology, DimensionTag< EntityTopology::dimension >, false >
{
   using SuperdimensionTag = DimensionTag< EntityTopology::dimension >;
   using MeshTraitsType = MeshTraits< MeshConfig >;
   using SuperentityTraitsType = typename MeshTraitsType::template SuperentityTraits< EntityTopology, SuperdimensionTag::value >;
   using ThisType = MeshSuperentityStorageLayer< MeshConfig, EntityTopology, SuperdimensionTag, false >;

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
 
   void getSuperentityStorageNetwork( SuperdimensionTag ) {}
};

template< typename MeshConfig,
          typename EntityTopology >
class MeshSuperentityStorageLayer< MeshConfig,
                                   EntityTopology,
                                   DimensionTag< EntityTopology::dimension >,
                                   true >
{
   using SuperdimensionTag = DimensionTag< EntityTopology::dimension >;
   using MeshTraitsType = MeshTraits< MeshConfig >;
   using SuperentityTraitsType = typename MeshTraitsType::template SuperentityTraits< EntityTopology, SuperdimensionTag::value >;
   using ThisType = MeshSuperentityStorageLayer< MeshConfig, EntityTopology, SuperdimensionTag, true >;

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
 
   void getSuperentityStorageNetwork( SuperdimensionTag ) {}
};

} // namespace Meshes
} // namespace TNL
