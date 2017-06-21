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
#include <TNL/Meshes/DimensionTag.h>
#include <TNL/Meshes/MeshDetails/traits/MeshTraits.h>
#include <TNL/Meshes/MeshDetails/traits/MeshSubentityTraits.h>
#include <TNL/Meshes/MeshDetails/MeshEntityOrientation.h>

namespace TNL {
namespace Meshes {

template< typename MeshConfig,
          typename Device,
          typename EntityTopology,
          typename DimensionTag,
          bool SubentityStorage =
               MeshConfig::subentityStorage( EntityTopology(), DimensionTag::value ),
          bool SubentityOrientationStorage =
               MeshConfig::subentityOrientationStorage( EntityTopology(), DimensionTag::value ) >
class MeshSubentityAccessLayer;


template< typename MeshConfig,
          typename Device,
          typename EntityTopology >
class MeshSubentityAccess
   : public MeshSubentityAccessLayer< MeshConfig,
                                      Device,
                                      EntityTopology,
                                      Meshes::DimensionTag< 0 > >
{
   using BaseType = MeshSubentityAccessLayer< MeshConfig,
                                              Device,
                                              EntityTopology,
                                              Meshes::DimensionTag< 0 > >;

   using MeshTraitsType = MeshTraits< MeshConfig, Device >;

   template< int Subdimension >
   using SubentityTraits = typename MeshTraitsType::template SubentityTraits< EntityTopology, Subdimension >;

public:
   template< int Subdimension >
   __cuda_callable__
   void bindSubentitiesStorageNetwork( const typename SubentityTraits< Subdimension >::SubentityAccessorType& storage )
   {
      static_assert( SubentityTraits< Subdimension >::storageEnabled, "You try to bind subentities which are not configured for storage." );
      BaseType::bindSubentitiesStorageNetwork( Meshes::DimensionTag< Subdimension >(),
                                               storage );
   }

   template< int Subdimension >
   static constexpr typename SubentityTraits< Subdimension >::LocalIndexType getSubentitiesCount()
   {
      return SubentityTraits< Subdimension >::count;
   }

   template< int Subdimension >
   __cuda_callable__
   void setSubentityIndex( const typename SubentityTraits< Subdimension >::LocalIndexType& localIndex,
                           const typename SubentityTraits< Subdimension >::GlobalIndexType& globalIndex )
   {
      static_assert( SubentityTraits< Subdimension >::storageEnabled, "You try to set subentity which is not configured for storage." );
      BaseType::setSubentityIndex( Meshes::DimensionTag< Subdimension >(),
                                   localIndex,
                                   globalIndex );
   }

   template< int Subdimension >
   __cuda_callable__
   typename SubentityTraits< Subdimension >::GlobalIndexType
   getSubentityIndex( const typename SubentityTraits< Subdimension >::LocalIndexType localIndex ) const
   {
      static_assert( SubentityTraits< Subdimension >::storageEnabled, "You try to get subentity which is not configured for storage." );
      return BaseType::getSubentityIndex( Meshes::DimensionTag< Subdimension >(),
                                          localIndex );
   }

   template< int Subdimension >
   __cuda_callable__
   typename SubentityTraits< Subdimension >::OrientationArrayType& subentityOrientationsArray()
   {
      static_assert( SubentityTraits< Subdimension >::orientationEnabled, "You try to get subentity orientation which is not configured for storage." );
      return BaseType::subentityOrientationsArray( Meshes::DimensionTag< Subdimension >() );
   }

   template< int Subdimension >
   __cuda_callable__
   typename SubentityTraits< Subdimension >::IdPermutationArrayType getSubentityOrientation( typename SubentityTraits< Subdimension >::LocalIndexType index ) const
   {
      static_assert( SubentityTraits< Subdimension >::orientationEnabled, "You try to get subentity orientation which is not configured for storage." );
      return BaseType::getSubentityOrientation( Meshes::DimensionTag< Subdimension >(), index );
   }

   __cuda_callable__
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
          typename Device,
          typename EntityTopology,
          typename DimensionTag >
class MeshSubentityAccessLayer< MeshConfig,
                                Device,
                                EntityTopology,
                                DimensionTag,
                                true,
                                true >
   : public MeshSubentityAccessLayer< MeshConfig,
                                      Device,
                                      EntityTopology,
                                      typename DimensionTag::Increment >
{
   using BaseType = MeshSubentityAccessLayer< MeshConfig,
                                              Device,
                                              EntityTopology,
                                              typename DimensionTag::Increment >;

   using MeshTraitsType         = MeshTraits< MeshConfig, Device >;
   using SubentityTraitsType    = typename MeshTraitsType::template SubentityTraits< EntityTopology, DimensionTag::value >;

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

   __cuda_callable__
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
         std::cerr << "Saving of the entity subentities layer with " << DimensionTag::value << " failed." << std::endl;
         return false;
      }
      return true;
   }

   bool load( File& file )
   {
      if( ! BaseType::load( file ) )
      {
         std::cerr << "Loading of the entity subentities layer with " << DimensionTag::value << " failed." << std::endl;
         return false;
      }
      return true;
   }

   void print( std::ostream& str ) const
   {
      BaseType::print( str );
      str << "\t Subentities with dimension " << DimensionTag::value << " are: " << subentityIndices << "." << std::endl;
   }

   __cuda_callable__
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
   __cuda_callable__
   void bindSubentitiesStorageNetwork( DimensionTag,
                                       const SubentityAccessorType& storage )
   {
      this->subentityIndices.bind( storage );
   }

   __cuda_callable__
   void setSubentityIndex( DimensionTag,
                           const LocalIndexType localIndex,
                           const GlobalIndexType globalIndex )
   {
      this->subentityIndices[ localIndex ] = globalIndex;
   }

   __cuda_callable__
   GlobalIndexType getSubentityIndex( DimensionTag,
                                      const LocalIndexType localIndex ) const
   {
      return this->subentityIndices[ localIndex ];
   }

   __cuda_callable__
   const SubentityAccessorType& getSubentityIndices( DimensionTag ) const
   {
      return this->subentityIndices;
   }

   __cuda_callable__
   SubentityAccessorType& getSubentityIndices( DimensionTag )
   {
      return this->subentityIndices;
   }

   using BaseType::getSubentityOrientation;
   __cuda_callable__
   const IdPermutationArrayType& getSubentityOrientation( DimensionTag, LocalIndexType index) const
   {
      TNL_ASSERT_GE( index, 0, "index must be non-negative" );
      TNL_ASSERT_LT( index, SubentityTraitsType::count, "index is out of bounds" );
      return this->subentityOrientations[ index ].getSubvertexPermutation();
   }

   using BaseType::subentityOrientationsArray;
   __cuda_callable__
	OrientationArrayType& subentityOrientationsArray( DimensionTag ) { return this->subentityOrientations; }

private:
   SubentityAccessorType subentityIndices;

   OrientationArrayType subentityOrientations;
};

template< typename MeshConfig,
          typename Device,
          typename EntityTopology,
          typename DimensionTag >
class MeshSubentityAccessLayer< MeshConfig,
                                Device,
                                EntityTopology,
                                DimensionTag,
                                true,
                                false >
   : public MeshSubentityAccessLayer< MeshConfig,
                                      Device,
                                      EntityTopology,
                                      typename DimensionTag::Increment >
{
   static_assert( DimensionTag::value < EntityTopology::dimension, "" );
   using BaseType = MeshSubentityAccessLayer< MeshConfig,
                                              Device,
                                              EntityTopology,
                                              typename DimensionTag::Increment >;

   using MeshTraitsType        = MeshTraits< MeshConfig, Device >;
   using SubentityTraitsType   = typename MeshTraitsType::template SubentityTraits< EntityTopology, DimensionTag::value >;

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

   __cuda_callable__
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
         std::cerr << "Saving of the entity subentities layer with " << DimensionTag::value << " failed." << std::endl;
         return false;
      }
      return true;
   }

   bool load( File& file )
   {
      if( ! BaseType::load( file ) )
      {
         std::cerr << "Loading of the entity subentities layer with " << DimensionTag::value << " failed." << std::endl;
         return false;
      }
      return true;
   }

   void print( std::ostream& str ) const
   {
      BaseType::print( str );
      str << "\t Subentities with dimension " << DimensionTag::value << " are: " << subentityIndices << "." << std::endl;
   }

   __cuda_callable__
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
   __cuda_callable__
   void bindSubentitiesStorageNetwork( DimensionTag,
                                       const SubentityAccessorType& storage )
   {
      this->subentityIndices.bind( storage );
   }

   __cuda_callable__
   void setSubentityIndex( DimensionTag,
                           const LocalIndexType localIndex,
                           const GlobalIndexType globalIndex )
   {
      this->subentityIndices[ localIndex ] = globalIndex;
   }

   __cuda_callable__
   GlobalIndexType getSubentityIndex( DimensionTag,
                                      const LocalIndexType localIndex ) const
   {
      return this->subentityIndices[ localIndex ];
   }

   __cuda_callable__
   const SubentityAccessorType& getSubentityIndices( DimensionTag ) const
   {
      return this->subentityIndices;
   }

   __cuda_callable__
   SubentityAccessorType& getSubentityIndices( DimensionTag )
   {
      return this->subentityIndices;
   }

private:
   SubentityAccessorType subentityIndices;
};


template< typename MeshConfig,
          typename Device,
          typename EntityTopology >
class MeshSubentityAccessLayer< MeshConfig,
                                Device,
                                EntityTopology,
                                Meshes::DimensionTag< EntityTopology::dimension >,
                                true,
                                true >
{
   using DimensionTag = Meshes::DimensionTag< EntityTopology::dimension >;

protected:
   /***
    * Necessary because of 'using BaseType::...;' in the derived classes
    */
   template< typename SubentityAccessorType >
   __cuda_callable__
   void bindSubentitiesStorageNetwork( DimensionTag,
                                       const SubentityAccessorType& storage ) {}
   __cuda_callable__
   void getSubentitiesCount( DimensionTag ) const {}
   template< typename LocalIndexType >
   __cuda_callable__
   void getSubentityIndex( DimensionTag,
                           const LocalIndexType localIndex ) const {}
   template< typename LocalIndexType, typename GlobalIndexType >
   __cuda_callable__
   void setSubentityIndex( DimensionTag,
                           const LocalIndexType& localIndex,
                           const GlobalIndexType& globalIndex ) {}
   __cuda_callable__
   void getSubentityIndices() {}

   template< typename LocalIndexType >
   __cuda_callable__
   void getSubentityOrientation( DimensionTag, LocalIndexType index) const {}
   __cuda_callable__
	void subentityOrientationsArray( DimensionTag ) {}

   bool save( File& file ) const
   {
      return true;
   }

   bool load( File& file )
   {
      return true;
   }

   __cuda_callable__
   bool operator==( const MeshSubentityAccessLayer& other ) const
   {
      return true;
   }

   void print( std::ostream& str ) const {}
};

template< typename MeshConfig,
          typename Device,
          typename EntityTopology >
class MeshSubentityAccessLayer< MeshConfig,
                                Device,
                                EntityTopology,
                                Meshes::DimensionTag< EntityTopology::dimension >,
                                true,
                                false >
{
   using DimensionTag = Meshes::DimensionTag< EntityTopology::dimension >;

protected:
   /***
    * Necessary because of 'using BaseType::...;' in the derived classes
    */
   template< typename SubentityAccessorType >
   __cuda_callable__
   void bindSubentitiesStorageNetwork( DimensionTag,
                                       const SubentityAccessorType& storage ) {}
   __cuda_callable__
   void getSubentitiesCount( DimensionTag ) const {}
   template< typename LocalIndexType >
   __cuda_callable__
   void getSubentityIndex( DimensionTag,
                           const LocalIndexType localIndex ) const {}
   template< typename LocalIndexType, typename GlobalIndexType >
   __cuda_callable__
   void setSubentityIndex( DimensionTag,
                           const LocalIndexType& localIndex,
                           const GlobalIndexType& globalIndex ) {}
   __cuda_callable__
   void getSubentityIndices() {}

   template< typename LocalIndexType >
   __cuda_callable__
   void getSubentityOrientation( DimensionTag, LocalIndexType index) const {}
   __cuda_callable__
	void subentityOrientationsArray( DimensionTag ) {}

   bool save( File& file ) const
   {
      return true;
   }

   bool load( File& file )
   {
      return true;
   }

   __cuda_callable__
   bool operator==( const MeshSubentityAccessLayer& other ) const
   {
      return true;
   }

   void print( std::ostream& str ) const {}
};

template< typename MeshConfig,
          typename Device,
          typename EntityTopology >
class MeshSubentityAccessLayer< MeshConfig,
                                Device,
                                EntityTopology,
                                Meshes::DimensionTag< EntityTopology::dimension >,
                                false,
                                true >
{
   using DimensionTag = Meshes::DimensionTag< EntityTopology::dimension >;

protected:
   /***
    * Necessary because of 'using BaseType::...;' in the derived classes
    */
   template< typename SubentityAccessorType >
   __cuda_callable__
   void bindSubentitiesStorageNetwork( DimensionTag,
                                       const SubentityAccessorType& storage ) {}
   void getSubentitiesCount( DimensionTag ) const {}
   template< typename LocalIndexType >
   __cuda_callable__
   void getSubentityIndex( DimensionTag,
                           const LocalIndexType localIndex ) const {}
   template< typename LocalIndexType, typename GlobalIndexType >
   __cuda_callable__
   void setSubentityIndex( DimensionTag,
                           const LocalIndexType& localIndex,
                           const GlobalIndexType& globalIndex ) {}
   __cuda_callable__
   void getSubentityIndices() {}

   template< typename LocalIndexType >
   __cuda_callable__
   void getSubentityOrientation( DimensionTag, LocalIndexType index) const {}
   __cuda_callable__
	void subentityOrientationsArray( DimensionTag ) {}

   bool save( File& file ) const
   {
      return true;
   }

   bool load( File& file )
   {
      return true;
   }

   __cuda_callable__
   bool operator==( const MeshSubentityAccessLayer& other ) const
   {
      return true;
   }

   void print( std::ostream& str ) const {}
};

template< typename MeshConfig,
          typename Device,
          typename EntityTopology,
          typename DimensionTag >
class MeshSubentityAccessLayer< MeshConfig,
                                Device,
                                EntityTopology,
                                DimensionTag,
                                false,
                                false >
{
protected:
   /***
    * Necessary because of 'using BaseType::...;' in the derived classes
    */
   template< typename SubentityAccessorType >
   __cuda_callable__
   void bindSubentitiesStorageNetwork( DimensionTag,
                                       const SubentityAccessorType& storage ) {}
   __cuda_callable__
   void getSubentitiesCount( DimensionTag ) const {}
   template< typename LocalIndexType >
   __cuda_callable__
   void getSubentityIndex( DimensionTag,
                           const LocalIndexType localIndex ) const {}
   template< typename LocalIndexType, typename GlobalIndexType >
   __cuda_callable__
   void setSubentityIndex( DimensionTag,
                           const LocalIndexType& localIndex,
                           const GlobalIndexType& globalIndex ) {}
   __cuda_callable__
   void getSubentityIndices() {}

   template< typename LocalIndexType >
   __cuda_callable__
   void getSubentityOrientation( DimensionTag, LocalIndexType index) const {}
   __cuda_callable__
	void subentityOrientationsArray( DimensionTag ) {}

   bool save( File& file ) const
   {
      return true;
   }

   bool load( File& file )
   {
      return true;
   }

   __cuda_callable__
   bool operator==( const MeshSubentityAccessLayer& other ) const
   {
      return true;
   }

   void print( std::ostream& str ) const {}
};

} // namespace Meshes
} // namespace TNL
