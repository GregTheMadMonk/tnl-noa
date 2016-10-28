/***************************************************************************
                          MeshSubentityAccess.h  -  description
                             -------------------
    begin                : Oct 26, 2016
    copyright            : (C) 2014 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/File.h>
#include <TNL/Meshes/MeshDimensionsTag.h>
#include <TNL/Meshes/MeshDetails/traits/MeshTraits.h>
#include <TNL/Meshes/MeshDetails/traits/MeshSubentityTraits.h>
#include <TNL/Meshes/MeshDetails/MeshEntityOrientation.h>

namespace TNL {
namespace Meshes {

template< typename MeshConfig,
          typename EntityTopology,
          typename DimensionsTag,
          bool SubentityStorage =
               MeshConfig::subentityStorage( EntityTopology(), DimensionsTag::value ),
          bool SubentityOrientationStorage =
               MeshConfig::subentityOrientationStorage( EntityTopology(), DimensionsTag::value ) >
class MeshSubentityAccessLayer;


template< typename MeshConfig,
          typename EntityTopology >
class MeshSubentityAccess
   : public MeshSubentityAccessLayer< MeshConfig,
                                      EntityTopology,
                                      MeshDimensionsTag< 0 > >
{
   using BaseType = MeshSubentityAccessLayer< MeshConfig,
                                              EntityTopology,
                                              MeshDimensionsTag< 0 > >;

   using MeshTraitsType = MeshTraits< MeshConfig >;

   template< int Subdimensions >
   using SubentityTraits = typename MeshTraitsType::template SubentityTraits< EntityTopology, Subdimensions >;

public:
   template< int Subdimensions >
   void bindSubentitiesStorageNetwork( const typename SubentityTraits< Subdimensions >::SubentityAccessorType& storage )
   {
      static_assert( SubentityTraits< Subdimensions >::storageEnabled, "You try to bind subentities which are not configured for storage." );
      BaseType::bindSubentitiesStorageNetwork( MeshDimensionsTag< Subdimensions >(),
                                               storage );
   }

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
      BaseType::setSubentityIndex( MeshDimensionsTag< Subdimensions >(),
                                   localIndex,
                                   globalIndex );
   }

   template< int Subdimensions >
   typename SubentityTraits< Subdimensions >::GlobalIndexType
   getSubentityIndex( const typename SubentityTraits< Subdimensions >::LocalIndexType localIndex ) const
   {
      static_assert( SubentityTraits< Subdimensions >::storageEnabled, "You try to get subentity which is not configured for storage." );
      return BaseType::getSubentityIndex( MeshDimensionsTag< Subdimensions >(),
                                          localIndex );
   }

   template< int Subdimensions >
   typename SubentityTraits< Subdimensions >::OrientationArrayType& subentityOrientationsArray()
   {
      static_assert( SubentityTraits< Subdimensions >::orientationEnabled, "You try to get subentity orientation which is not configured for storage." );
      return BaseType::subentityOrientationsArray( MeshDimensionsTag< Subdimensions >() );
   }

   template< int Subdimensions >
   typename SubentityTraits< Subdimensions >::IdPermutationArrayType getSubentityOrientation( typename SubentityTraits< Subdimensions >::LocalIndexType index ) const
   {
      static_assert( SubentityTraits< Subdimensions >::orientationEnabled, "You try to get subentity orientation which is not configured for storage." );
      return BaseType::getSubentityOrientation( MeshDimensionsTag< Subdimensions >(), index );
   }

   bool operator==( const MeshSubentityAccess& other ) const
   {
      return BaseType::operator==( other );
   }

   void print( std::ostream& str ) const
   {
      BaseType::print( str );
   }
};


template< typename MeshConfig,
          typename EntityTopology,
          typename DimensionsTag >
class MeshSubentityAccessLayer< MeshConfig,
                                EntityTopology,
                                DimensionsTag,
                                true,
                                true >
   : public MeshSubentityAccessLayer< MeshConfig,
                                      EntityTopology,
                                      typename DimensionsTag::Increment >
{
   using BaseType = MeshSubentityAccessLayer< MeshConfig,
                                              EntityTopology,
                                              typename DimensionsTag::Increment >;

   using MeshTraitsType         = MeshTraits< MeshConfig >;
   using SubentityTraitsType    = typename MeshTraitsType::template SubentityTraits< EntityTopology, DimensionsTag::value >;

protected:
   using GlobalIndexType        = typename SubentityTraitsType::GlobalIndexType;
   using LocalIndexType         = typename SubentityTraitsType::LocalIndexType;
   using StorageNetworkType     = typename SubentityTraitsType::StorageNetworkType;
   using SubentityAccessorType  = typename SubentityTraitsType::SubentityAccessorType;
   using OrientationArrayType   = typename SubentityTraitsType::OrientationArrayType;
   using IdPermutationArrayType = typename SubentityTraitsType::IdPermutationArrayType;

   MeshSubentityAccessLayer() = default;

   explicit MeshSubentityAccessLayer( const MeshSubentityAccessLayer& layer )
      : BaseType( layer )
   {
      this->subentityIndices.bind( layer.subentityIndices );
   }

   MeshSubentityAccessLayer& operator=( const MeshSubentityAccessLayer& layer )
   {
      BaseType::operator=( layer );
      this->subentityIndices.bind( layer.subentityIndices );
      return *this;
   }

   bool save( File& file ) const
   {
      if( ! BaseType::save( file ) )
      {
         std::cerr << "Saving of the entity subentities layer with " << DimensionsTag::value << " failed." << std::endl;
         return false;
      }
      return true;
   }

   bool load( File& file )
   {
      if( ! BaseType::load( file ) )
      {
         std::cerr << "Loading of the entity subentities layer with " << DimensionsTag::value << " failed." << std::endl;
         return false;
      }
      return true;
   }

   void print( std::ostream& str ) const
   {
      BaseType::print( str );
      str << "\t Subentities with " << DimensionsTag::value << " dimensions are: " << subentityIndices << "." << std::endl;
   }

   bool operator==( const MeshSubentityAccessLayer& layer ) const
   {
      return ( BaseType::operator==( layer ) &&
               subentityIndices == layer.subentityIndices );
   }

   /****
    * Make visible setters and getters of the lower subentities
    */
   using BaseType::bindSubentitiesStorageNetwork;
   using BaseType::getSubentityIndex;
   using BaseType::setSubentityIndex;
   using BaseType::getSubentityIndices;

   /****
    * Define setter/getter for the current level of the subentities
    */
   void bindSubentitiesStorageNetwork( DimensionsTag,
                                       const SubentityAccessorType& storage )
   {
      this->subentityIndices.bind( storage );
   }

   void setSubentityIndex( DimensionsTag,
                           const LocalIndexType localIndex,
                           const GlobalIndexType globalIndex )
   {
      this->subentityIndices[ localIndex ] = globalIndex;
   }

   GlobalIndexType getSubentityIndex( DimensionsTag,
                                      const LocalIndexType localIndex ) const
   {
      return this->subentityIndices[ localIndex ];
   }

   const SubentityAccessorType& getSubentityIndices( DimensionsTag ) const
   {
      return this->subentityIndices;
   }

   SubentityAccessorType& getSubentityIndices( DimensionsTag )
   {
      return this->subentityIndices;
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
   SubentityAccessorType subentityIndices;

   OrientationArrayType subentityOrientations;
};

template< typename MeshConfig,
          typename EntityTopology,
          typename DimensionsTag >
class MeshSubentityAccessLayer< MeshConfig,
                                EntityTopology,
                                DimensionsTag,
                                true,
                                false >
   : public MeshSubentityAccessLayer< MeshConfig,
                                      EntityTopology,
                                      typename DimensionsTag::Increment >
{
   static_assert( DimensionsTag::value < EntityTopology::dimensions, "" );
   using BaseType = MeshSubentityAccessLayer< MeshConfig,
                                              EntityTopology,
                                              typename DimensionsTag::Increment >;

   using MeshTraitsType        = MeshTraits< MeshConfig >;
   using SubentityTraitsType   = typename MeshTraitsType::template SubentityTraits< EntityTopology, DimensionsTag::value >;

protected:
   using GlobalIndexType       = typename SubentityTraitsType::GlobalIndexType;
   using LocalIndexType        = typename SubentityTraitsType::LocalIndexType;
   using StorageNetworkType    = typename SubentityTraitsType::StorageNetworkType;
   using SubentityAccessorType = typename SubentityTraitsType::SubentityAccessorType;

   MeshSubentityAccessLayer() = default;

   explicit MeshSubentityAccessLayer( const MeshSubentityAccessLayer& layer )
      : BaseType( layer )
   {
      this->subentityIndices.bind( layer.subentityIndices );
   }

   MeshSubentityAccessLayer& operator=( const MeshSubentityAccessLayer& layer )
   {
      BaseType::operator=( layer );
      this->subentityIndices.bind( layer.subentityIndices );
      return *this;
   }

   bool save( File& file ) const
   {
      if( ! BaseType::save( file ) )
      {
         std::cerr << "Saving of the entity subentities layer with " << DimensionsTag::value << " failed." << std::endl;
         return false;
      }
      return true;
   }

   bool load( File& file )
   {
      if( ! BaseType::load( file ) )
      {
         std::cerr << "Loading of the entity subentities layer with " << DimensionsTag::value << " failed." << std::endl;
         return false;
      }
      return true;
   }

   void print( std::ostream& str ) const
   {
      BaseType::print( str );
      str << "\t Subentities with " << DimensionsTag::value << " dimensions are: " << subentityIndices << "." << std::endl;
   }

   bool operator==( const MeshSubentityAccessLayer& layer ) const
   {
      return ( BaseType::operator==( layer ) &&
               subentityIndices == layer.subentityIndices );
   }

   /****
    * Make visible setters and getters of the lower subentities
    */
   using BaseType::bindSubentitiesStorageNetwork;
   using BaseType::getSubentityIndex;
   using BaseType::setSubentityIndex;
   using BaseType::getSubentityIndices;

   /****
    * Define setter/getter for the current level of the subentities
    */
   void bindSubentitiesStorageNetwork( DimensionsTag,
                                       const SubentityAccessorType& storage )
   {
      this->subentityIndices.bind( storage );
   }

   void setSubentityIndex( DimensionsTag,
                           const LocalIndexType localIndex,
                           const GlobalIndexType globalIndex )
   {
      this->subentityIndices[ localIndex ] = globalIndex;
   }

   GlobalIndexType getSubentityIndex( DimensionsTag,
                                      const LocalIndexType localIndex ) const
   {
      return this->subentityIndices[ localIndex ];
   }

   const SubentityAccessorType& getSubentityIndices( DimensionsTag ) const
   {
      return this->subentityIndices;
   }

   SubentityAccessorType& getSubentityIndices( DimensionsTag )
   {
      return this->subentityIndices;
   }

private:
   SubentityAccessorType subentityIndices;
};


template< typename MeshConfig,
          typename EntityTopology >
class MeshSubentityAccessLayer< MeshConfig,
                                EntityTopology,
                                MeshDimensionsTag< EntityTopology::dimensions >,
                                true,
                                true >
{
   using DimensionsTag = MeshDimensionsTag< EntityTopology::dimensions >;

protected:
   /***
    * Necessary because of 'using BaseType::...;' in the derived classes
    */
   template< typename SubentityAccessorType >
   void bindSubentitiesStorageNetwork( DimensionsTag,
                                       const SubentityAccessorType& storage ) {}
   void getNumberOfSubentities( DimensionsTag ) const {}
   template< typename LocalIndexType >
   void getSubentityIndex( DimensionsTag,
                           const LocalIndexType localIndex ) const {}
   template< typename LocalIndexType, typename GlobalIndexType >
   void setSubentityIndex( DimensionsTag,
                           const LocalIndexType& localIndex,
                           const GlobalIndexType& globalIndex ) {}
   void getSubentityIndices() {}

   template< typename LocalIndexType >
   void getSubentityOrientation( DimensionsTag, LocalIndexType index) const {}
	void subentityOrientationsArray( DimensionsTag ) {}

   bool save( File& file ) const
   {
      return true;
   }

   bool load( File& file )
   {
      return true;
   }

   bool operator==( const MeshSubentityAccessLayer& other ) const
   {
      return true;
   }

   void print( std::ostream& str ) const {}
};

template< typename MeshConfig,
          typename EntityTopology >
class MeshSubentityAccessLayer< MeshConfig,
                                EntityTopology,
                                MeshDimensionsTag< EntityTopology::dimensions >,
                                true,
                                false >
{
   using DimensionsTag = MeshDimensionsTag< EntityTopology::dimensions >;

protected:
   /***
    * Necessary because of 'using BaseType::...;' in the derived classes
    */
   template< typename SubentityAccessorType >
   void bindSubentitiesStorageNetwork( DimensionsTag,
                                       const SubentityAccessorType& storage ) {}
   void getNumberOfSubentities( DimensionsTag ) const {}
   template< typename LocalIndexType >
   void getSubentityIndex( DimensionsTag,
                           const LocalIndexType localIndex ) const {}
   template< typename LocalIndexType, typename GlobalIndexType >
   void setSubentityIndex( DimensionsTag,
                           const LocalIndexType& localIndex,
                           const GlobalIndexType& globalIndex ) {}
   void getSubentityIndices() {}

   template< typename LocalIndexType >
   void getSubentityOrientation( DimensionsTag, LocalIndexType index) const {}
	void subentityOrientationsArray( DimensionsTag ) {}

   bool save( File& file ) const
   {
      return true;
   }

   bool load( File& file )
   {
      return true;
   }

   bool operator==( const MeshSubentityAccessLayer& other ) const
   {
      return true;
   }

   void print( std::ostream& str ) const {}
};

template< typename MeshConfig,
          typename EntityTopology >
class MeshSubentityAccessLayer< MeshConfig,
                                EntityTopology,
                                MeshDimensionsTag< EntityTopology::dimensions >,
                                false,
                                true >
{
   using DimensionsTag = MeshDimensionsTag< EntityTopology::dimensions >;

protected:
   /***
    * Necessary because of 'using BaseType::...;' in the derived classes
    */
   template< typename SubentityAccessorType >
   void bindSubentitiesStorageNetwork( DimensionsTag,
                                       const SubentityAccessorType& storage ) {}
   void getNumberOfSubentities( DimensionsTag ) const {}
   template< typename LocalIndexType >
   void getSubentityIndex( DimensionsTag,
                           const LocalIndexType localIndex ) const {}
   template< typename LocalIndexType, typename GlobalIndexType >
   void setSubentityIndex( DimensionsTag,
                           const LocalIndexType& localIndex,
                           const GlobalIndexType& globalIndex ) {}
   void getSubentityIndices() {}

   template< typename LocalIndexType >
   void getSubentityOrientation( DimensionsTag, LocalIndexType index) const {}
	void subentityOrientationsArray( DimensionsTag ) {}

   bool save( File& file ) const
   {
      return true;
   }

   bool load( File& file )
   {
      return true;
   }

   bool operator==( const MeshSubentityAccessLayer& other ) const
   {
      return true;
   }

   void print( std::ostream& str ) const {}
};

template< typename MeshConfig,
          typename EntityTopology,
          typename DimensionsTag >
class MeshSubentityAccessLayer< MeshConfig,
                                EntityTopology,
                                DimensionsTag,
                                false,
                                false >
{
};

} // namespace Meshes
} // namespace TNL
