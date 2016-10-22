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
          typename DimensionTag,
          bool SuperentityStorage =
               MeshSuperentityTraits< MeshConfig, EntityTopology, DimensionsTag::value >::storageEnabled >
class MeshSuperentityStorageLayer;

template< typename MeshConfig,
          typename EntityTopology >
class MeshSuperentityStorageLayers
   : public MeshSuperentityStorageLayer< MeshConfig,
                                         EntityTopology,
                                         MeshDimensionsTag< MeshTraits< MeshConfig >::meshDimensions > >
{
};

template< typename MeshConfig,
          typename EntityTopology,
          typename DimensionTag >
class MeshSuperentityStorageLayer< MeshConfig, EntityTopology, DimensionTag, true >
   : public MeshSuperentityStorageLayer< MeshConfig, EntityTopology, typename DimensionTag::Decrement >
{
   using BaseType = MeshSuperentityStorageLayer< MeshConfig, EntityTopology, typename DimensionsTag::Decrement >;

   static constexpr int Dimensions = DimensionsTag::value;
   using MeshTraitsType        = MeshTraits< MeshConfig >;
   using SuperentityTraitsType = typename MeshTraitsType::template SuperentityTraits< EntityTopology, Dimensions >;

protected:
   using GlobalIndexType    = typename SuperentityTraitsType::GlobalIndexType;
   using LocalIndexType     = typename SuperentityTraitsType::LocalIndexType;
   using StorageNetworkType = typename SuperentityTraitsType::StorageNetworkType;
 
   /****
     * Make visible setters and getters of the lower superentities
     */
   using BaseType::setNumberOfSuperentities;
   using BaseType::getNumberOfSuperentities;
   using BaseType::setSuperentityIndex;
   using BaseType::getSuperentityIndex;
   using BaseType::getSuperentityIndices;

   MeshSuperentityStorageLayer& operator = ( const MeshSuperentityStorageLayer& layer ) = delete;

   bool save( File& file ) const
   {
      if( ! BaseType::save( file ) ||
          ! this->storageNetwork.save( file ) )
      {
         std::cerr << "Saving of the entity superentities layer with " << DimensionsTag::value << " dimensions failed." << std::endl;
         return false;
      }
      return true;
   }

   bool load( File& file )
   {
      if( ! BaseType::load( file ) ||
          ! this->storageNetwork.load( file ) )
      {
         std::cerr << "Loading of the entity superentities layer with " << DimensionsTag::value << " dimensions failed." << std::endl;
         return false;
      }
      return true;
   }

   void print( std::ostream& str ) const
   {
      BaseType::print( str );
      str << "Storage network for superentities with " << DimensionsTag::value << " dimensions of entities with " << EntityTopology::dimensions << " dimensions is: " << std::endl;
      str << this->storageNetwork << std::endl;
   }

   bool operator==( const MeshSuperentityStorageLayer& layer ) const
   {
      return ( BaseType::operator==( layer ) &&
               storageNetwork == layer.storageNetwork );
   }

private:
    StorageNetworkType storageNetwork;
 
   // TODO: this is only for the mesh initializer - fix it
   public:
      using BaseType::getStorageNetwork;
      StorageNetworkType& getStorageNetwork( DimensionTag )
      {
         return this->storageNetwork;
      }
};

template< typename MeshConfig,
          typename EntityTopology,
          typename DimensionTag >
class MeshSuperentityStorageLayer< MeshConfig, EntityTopology, DimensionTag, false >
   : public MeshSuperentityStorageLayer< MeshConfig, EntityTopology, typename DimensionTag::Decrement >
{
};

template< typename MeshConfig,
          typename EntityTopology >
class MeshSuperentityStorageLayer< MeshConfig, EntityTopology, MeshDimensionTag< EntityTopology::dimensions >, false >
{
   using DimensionsTag = MeshDimensionsTag< EntityTopology::dimensions >;
   static constexpr int Dimensions = DimensionsTag::value;
   using MeshTraitsType = MeshTraits< MeshConfig >;
   using SuperentityTraitsType = typename MeshTraitsType::template SuperentityTraits< EntityTopology, Dimensions >;
   using ThisType = MeshSuperentityStorageLayer< MeshConfig, EntityTopology, DimensionsTag, false >;

protected:
   using GlobalIndexType    = typename SuperentityTraitsType::GlobalIndexType;
   using LocalIndexType     = typename SuperentityTraitsType::LocalIndexType;
   using StorageNetworkType = typename SuperentityTraitsType::StorageNetworkType;
 
   /****
    * These methods are due to 'using BaseType::...;' in the derived classes.
    */
   void setNumberOfSuperentities( DimensionsTag,
                                  const LocalIndexType size ) {}
   void getNumberOfSuperentities( DimensionsTag ) const {}
   void getSuperentityIndex( DimensionsTag,
                             const LocalIndexType localIndex ) {}
   void setSuperentityIndex( DimensionsTag,
                             const LocalIndexType localIndex,
                             const GlobalIndexType globalIndex ) {}

   void print( std::ostream& str ) const {}

   bool operator==( const ThisType& layer ) const
   {
      return true;
   }

   void getSuperentityIndices() {}

   void getSuperentityIndices() const {}

   bool save( File& file ) const
   {
      return true;
   }

   bool load( File& file )
   {
      return true;
   }
 
   void getStorageNetwork( DimensionsTag ) {}
};

template< typename MeshConfig,
          typename EntityTopology >
class MeshSuperentityStorageLayer< MeshConfig,
                                   EntityTopology,
                                   MeshDimensionsTag< EntityTopology::dimensions >,
                                   true >
{
   using DimensionsTag = MeshDimensionsTag< EntityTopology::dimensions >;
   static constexpr int Dimensions = DimensionsTag::value;
   using MeshTraitsType = MeshTraits< MeshConfig >;
   using SuperentityTraitsType = typename MeshTraitsType::template SuperentityTraits< EntityTopology, Dimensions >;
   using ThisType = MeshSuperentityStorageLayer< MeshConfig, EntityTopology, DimensionsTag, true >;

protected:
   using GlobalIndexType    = typename SuperentityTraitsType::GlobalIndexType;
   using LocalIndexType     = typename SuperentityTraitsType::LocalIndexType;
   using StorageNetworkType = typename SuperentityTraitsType::StorageNetworkType;
 
   /****
    * These methods are due to 'using BaseType::...;' in the derived classes.
    */
   void setNumberOfSuperentities( DimensionsTag,
                                  const LocalIndexType size ) {}
   void getNumberOfSuperentities( DimensionsTag ) const {}
   void getSuperentityIndex( DimensionsTag,
                             const LocalIndexType localIndex ) {}
   void setSuperentityIndex( DimensionsTag,
                             const LocalIndexType localIndex,
                             const GlobalIndexType globalIndex ) {}

   void print( std::ostream& str ) const {}

   bool operator==( const ThisType& layer ) const
   {
      return true;
   }

   void getSuperentityIndices() {}

   void getSuperentityIndices() const {}

   bool save( File& file ) const
   {
      return true;
   }

   bool load( File& file )
   {
      return true;
   }
 
   void getStorageNetwork( DimensionsTag ) {}
};

} // namespace Meshes
} // namespace TNL
