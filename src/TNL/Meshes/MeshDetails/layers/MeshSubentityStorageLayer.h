/***************************************************************************
                          MeshSubentityStorageLayer.h  -  description
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
#include <TNL/Meshes/MeshDimensionTag.h>
#include <TNL/Meshes/MeshDetails/traits/MeshSubentityTraits.h>
#include <TNL/Meshes/MeshDetails/MeshEntityOrientation.h>

namespace TNL {
namespace Meshes {

template< typename MeshConfig,
          typename EntityTopology,
          typename DimensionTag,
          bool SubentityStorage =
               MeshTraits< MeshConfig >::template SubentityTraits< EntityTopology, DimensionsTag::value >::storageEnabled,
          bool SubentityOrientationStorage =
               MeshTraits< MeshConfig >::template SubentityTraits< EntityTopology, DimensionsTag::value >::orientationEnabled >
class MeshSubentityStorageLayer;


template< typename MeshConfig,
          typename EntityTopology >
class MeshSubentityStorageLayers
   : public MeshSubentityStorageLayer< MeshConfig,
                                       EntityTopology,
                                       MeshDimensionsTag< EntityTopology::dimensions - 1 > >
{
   using BaseType = MeshSubentityStorageLayer< MeshConfig,
                                               EntityTopology,
                                               MeshDimensionsTag< EntityTopology::dimensions - 1 > >;

   static constexpr int Dimensions = EntityTopology::dimensions;
   using MeshTraitsType = MeshTraits< MeshConfig >;

   template< int Subdimensions >
   using SubentityTraits = typename MeshTraitsType::template SubentityTraits< EntityTopology, Subdimensions >;

public:
   template< int Subdimensions >
   constexpr typename SubentityTraits< Subdimensions >::LocalIndexType getNumberOfSubentities() const
   {
      return SubentityTraits< Subdimensions >::count;
   }

   template< int Subdimensions >
   void setSubentityIndex( const typename SubentityTraits< Subdimensions >::LocalIndexType& localIndex,
                           const typename SubentityTraits< Subdimensions >::GlobalIndexType& globalIndex )
   {
      static_assert( SubentityTraits< Subdimensions >::storageEnabled, "You try to set subentity which is not configured for storage." );
      Assert( 0 <= localIndex && localIndex < SubentityTraits< Subdimensions >::count,
                 std::cerr << "localIndex = " << localIndex
                           << " subentitiesCount = "
                           << SubentityTraits< Subdimensions >::count );
      BaseType::setSubentityIndex( MeshDimensionsTag< Subdimensions >(),
                                   localIndex,
                                   globalIndex );
   }

   template< int Subdimensions >
   typename SubentityTraits< Subdimensions >::GlobalIndexType
   getSubentityIndex( const typename SubentityTraits< Subdimensions >::LocalIndexType localIndex ) const
   {
      static_assert( SubentityTraits< Subdimensions >::storageEnabled, "You try to get subentity which is not configured for storage." );
      Assert( 0 <= localIndex && localIndex < SubentityTraits< Subdimensions >::count,
                 std::cerr << "localIndex = " << localIndex
                           << " subentitiesCount = "
                           << SubentityTraits< Subdimensions >::count );
      return BaseType::getSubentityIndex( MeshDimensionsTag< Subdimensions >(),
                                          localIndex );
   }

   template< int Subdimensions >
   typename SubentityTraits< Subdimensions >::OrientationArrayType& subentityOrientationsArray()
   {
      return BaseType::subentityOrientationsArray( MeshDimensionsTag< Subdimensions >() );
   }

   template< int Subdimensions >
   typename SubentityTraits< Subdimensions >::IdPermutationArrayType getSubentityOrientation( typename SubentityTraits< Subdimensions >::LocalIndexType index ) const
   {
      Assert( 0 <= index && index < SubentityTraits< Subdimensions >::count, );
      return BaseType::getSubentityOrientation( MeshDimensionsTag< Subdimensions >(), index );
   }
};


template< typename MeshConfig,
          typename EntityTopology,
          typename DimensionTag >
class MeshSubentityStorageLayer< MeshConfig,
                                 EntityTopology,
                                 DimensionsTag,
                                 true,
                                 true >
   : public MeshSubentityStorageLayer< MeshConfig,
                                       EntityTopology,
                                       typename DimensionsTag::Decrement >
{
   typedef MeshSubentityStorageLayer< MeshConfig,
                                      EntityTopology,
                                      typename DimensionsTag::Decrement > BaseType;

protected:
   static constexpr int Dimensions = DimensionsTag::value;
   using MeshTraitsType         = MeshTraits< MeshConfig >;
   using SubentityTraitsType    = typename MeshTraitsType::template SubentityTraits< EntityTopology, Dimensions >;
   using GlobalIndexType        = typename MeshTraitsType::GlobalIndexType;
   using LocalIndexType         = typename MeshTraitsType::LocalIndexType;
   using IdArrayType            = typename SubentityTraitsType::IdArrayType;
   using OrientationArrayType   = typename SubentityTraitsType::OrientationArrayType;
   using IdPermutationArrayType = typename SubentityTraitsType::IdPermutationArrayType;

   MeshSubentityStorageLayer()
   {
      this->subentitiesIndices.setValue( -1 );
   }

   MeshSubentityStorageLayer& operator=( const MeshSubentityStorageLayer& layer )
   {
      BaseType::operator=( layer );
      this->subentitiesIndices = layer.subentitiesIndices;
      return *this;
   }

   bool save( File& file ) const
   {
      if( ! BaseType::save( file ) ||
          ! this->subentitiesIndices.save( file ) )
      {
         std::cerr << "Saving of the entity subentities layer with " << DimensionTag::value << " failed." << std::endl;
         return false;
      }
      return true;
   }

   bool load( File& file )
   {
      if( ! BaseType::load( file ) ||
          ! this->subentitiesIndices.load( file ) )
      {
         std::cerr << "Loading of the entity subentities layer with " << DimensionTag::value << " failed." << std::endl;
         return false;
      }
      return true;
   }

   void print( std::ostream& str ) const
   {
      BaseType::print( str );
      str << "\t Subentities with " << DimensionsTag::value << " dimensions are: " << subentitiesIndices << "." << std::endl;
   }

   bool operator==( const MeshSubentityStorageLayer& layer ) const
   {
      return ( BaseType::operator==( layer ) &&
               subentitiesIndices == layer.subentitiesIndices );
   }

   /****
    * Make visible setters and getters of the lower subentities
    */
   using BaseType::getSubentityIndex;
   using BaseType::setSubentityIndex;

   /****
    * Define setter/getter for the current level of the subentities
    */
   void setSubentityIndex( DimensionTag,
                           const LocalIndexType localIndex,
                           const GlobalIndexType globalIndex )
   {
      this->subentitiesIndices[ localIndex ] = globalIndex;
   }

   GlobalIndexType getSubentityIndex( DimensionTag,
                                      const LocalIndexType localIndex ) const
   {
      return this->subentitiesIndices[ localIndex ];
   }

   using BaseType::getSubentityOrientation;
   const IdPermutationArrayType& getSubentityOrientation( DimensionsTag, LocalIndexType index) const
   {
      Assert( 0 <= index && index < SubentityTraitsType::count, );

      return this->subentityOrientations[ index ].getSubvertexPermutation();
   }

   using BaseType::subentityOrientationsArray;
	OrientationArrayType& subentityOrientationsArray( DimensionsTag ) { return this->subentityOrientations; }

private:
   IdArrayType subentitiesIndices;

   OrientationArrayType subentityOrientations;
};


template< typename MeshConfig,
          typename EntityTopology,
          typename DimensionTag >
class MeshSubentityStorageLayer< MeshConfig,
                                 EntityTopology,
                                 DimensionsTag,
                                 true,
                                 false >
   : public MeshSubentityStorageLayer< MeshConfig,
                                       EntityTopology,
                                       typename DimensionsTag::Decrement >
{
   typedef MeshSubentityStorageLayer< MeshConfig,
                                      EntityTopology,
                                      typename DimensionsTag::Decrement > BaseType;

protected:
   static constexpr int Dimensions = DimensionsTag::value;
   using MeshTraitsType      = MeshTraits< MeshConfig >;
   using SubentityTraitsType = typename MeshTraitsType::template SubentityTraits< EntityTopology, Dimensions >;
   using GlobalIndexType     = typename MeshTraitsType::GlobalIndexType;
   using LocalIndexType      = typename MeshTraitsType::LocalIndexType;
   using IdArrayType         = typename SubentityTraitsType::IdArrayType;

   MeshSubentityStorageLayer()
   {
      this->subentitiesIndices.setValue( -1 );
   }

   MeshSubentityStorageLayer& operator=( const MeshSubentityStorageLayer& layer )
   {
      BaseType::operator=( layer );
      this->subentitiesIndices = layer.subentitiesIndices;
      return *this;
   }

   bool save( File& file ) const
   {
      if( ! BaseType::save( file ) ||
          ! this->subentitiesIndices.save( file ) )
      {
         std::cerr << "Saving of the entity subentities layer with " << DimensionTag::value << " failed." << std::endl;
         return false;
      }
      return true;
   }

   bool load( File& file )
   {
      if( ! BaseType::load( file ) ||
          ! this->subentitiesIndices.load( file ) )
      {
         std::cerr << "Loading of the entity subentities layer with " << DimensionTag::value << " failed." << std::endl;
         return false;
      }
      return true;
   }

   void print( std::ostream& str ) const
   {
      BaseType::print( str );
      str << "\t Subentities with " << DimensionsTag::value << " dimensions are: " << subentitiesIndices << "." << std::endl;
   }

   bool operator==( const MeshSubentityStorageLayer& layer ) const
   {
      return ( BaseType::operator==( layer ) &&
               subentitiesIndices == layer.subentitiesIndices );
   }

   /****
    * Make visible setters and getters of the lower subentities
    */
   using BaseType::getSubentityIndex;
   using BaseType::setSubentityIndex;

   /****
    * Define setter/getter for the current level of the subentities
    */
   void setSubentityIndex( DimensionTag,
                           const LocalIndexType localIndex,
                           const GlobalIndexType globalIndex )
   {
      this->subentitiesIndices[ localIndex ] = globalIndex;
   }

   GlobalIndexType getSubentityIndex( DimensionTag,
                                      const LocalIndexType localIndex ) const
   {
      return this->subentitiesIndices[ localIndex ];
   }

   using BaseType::subentityOrientationsArray;
   void subentityOrientationsArray() {}

private:
   IdArrayType subentitiesIndices;
};

template< typename MeshConfig,
          typename EntityTopology,
          typename DimensionTag >
class MeshSubentityStorageLayer< MeshConfig,
                                 EntityTopology,
                                 DimensionsTag,
                                 false,
                                 false >
   : public MeshSubentityStorageLayer< MeshConfig,
                                       EntityTopology,
                                       typename DimensionsTag::Decrement >
{
};


template< typename MeshConfig,
          typename EntityTopology >
class MeshSubentityStorageLayer< MeshConfig,
                                 EntityTopology,
                                 MeshDimensionsTag< 0 >,
                                 true,
                                 false >
{
   typedef MeshDimensionTag< 0 >                           DimensionTag;

protected:
   static constexpr int Dimensions = 0;
   using MeshTraitsType      = MeshTraits< MeshConfig >;
   using SubentityTraitsType = typename MeshTraitsType::template SubentityTraits< EntityTopology, Dimensions >;
   using GlobalIndexType     = typename MeshTraitsType::GlobalIndexType;
   using LocalIndexType      = typename MeshTraitsType::LocalIndexType;
   using IdArrayType         = typename SubentityTraitsType::IdArrayType;

   MeshSubentityStorageLayer()
   {
      this->verticesIndices.setValue( -1 );
   }

   MeshSubentityStorageLayer& operator=( const MeshSubentityStorageLayer& layer )
   {
      this->verticesIndices = layer.verticesIndices;
      return *this;
   }

   bool save( File& file ) const
   {
      if( ! this->verticesIndices.save( file ) )
      {
         std::cerr << "Saving of the entity subentities layer with " << DimensionTag::value << " failed." << std::endl;
         return false;
      }
      return true;
   }

   bool load( File& file )
   {
      if( ! this->verticesIndices.load( file ) )
      {
         std::cerr << "Loading of the entity subentities layer with " << DimensionTag::value << " failed." << std::endl;
         return false;
      }
      return true;
   }

   void print( std::ostream& str ) const
   {
      str << "\t Subentities with " << DimensionsTag::value << " dimensions are: " << this->verticesIndices << "." << std::endl;
   }

   bool operator==( const MeshSubentityStorageLayer& layer ) const
   {
      return ( verticesIndices == layer.verticesIndices );
   }

   GlobalIndexType getSubentityIndex( DimensionTag,
                                      const LocalIndexType localIndex ) const
   {
      return this->verticesIndices[ localIndex ];
   }
   void setSubentityIndex( DimensionTag,
                           const LocalIndexType localIndex,
                           const GlobalIndexType globalIndex )
   {
      this->verticesIndices[ localIndex ] = globalIndex;
   }

protected:
   /***
    *  Necessary because of 'using BaseType::...;' in the derived classes
    */
   void getSubentityOrientation() {}
   void subentityOrientationsArray() {}

   IdArrayType verticesIndices;
};

template< typename MeshConfig,
          typename EntityTopology >
class MeshSubentityStorageLayer< MeshConfig,
                                 EntityTopology,
                                 MeshDimensionsTag< 0 >,
                                 false,
                                 false >
{
public:
   bool save( File& file ) const
   {
      return true;
   }

   bool load( File& file )
   {
      return true;
   }
};

} // namespace Meshes
} // namespace TNL
