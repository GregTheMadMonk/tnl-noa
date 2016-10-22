/***************************************************************************
                          MeshSuperentityAccess.h  -  description
                             -------------------
    begin                : Aug 15, 2015
    copyright            : (C) 2015 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

/***
 * Authors:
 * Oberhuber Tomas, tomas.oberhuber@fjfi.cvut.cz
 * Zabka Vitezslav, zabkav@gmail.com
 */

#pragma once

#include <TNL/Meshes/MeshDetails/traits/MeshTraits.h>
#include <TNL/Meshes/MeshDetails/traits/MeshSuperentityTraits.h>

namespace TNL {
namespace Meshes {

template< typename MeshConfig,
          typename EntityTopology,
          typename DimensionsTag,
          bool SuperentityStorage =
             MeshTraits< MeshConfig >::template SuperentityTraits< EntityTopology, DimensionsTag::value >::storageEnabled >
class MeshSuperentityAccessLayer;


template< typename MeshConfig,
          typename EntityTopology >
class MeshSuperentityAccess
   : public MeshSuperentityAccessLayer< MeshConfig,
                                        EntityTopology,
                                        MeshDimensionsTag< MeshTraits< MeshConfig >::meshDimensions > >
{
   using BaseType = MeshSuperentityAccessLayer< MeshConfig,
                                                EntityTopology,
                                                MeshDimensionsTag< MeshTraits< MeshConfig >::meshDimensions > >;

   static constexpr int Dimensions = MeshTraits< MeshConfig >::meshDimensions;
   using MeshTraitsType = MeshTraits< MeshConfig >;

   template< int Superdimensions >
   using SuperentityTraits = typename MeshTraitsType::template SuperentityTraits< EntityTopology, Superdimensions >;

public:
   template< int Superdimensions >
   void bindSuperentitiesStorageNetwork( const typename SuperentityTraits< Superdimensions >::SuperentityAccessorType& storage )
   {
      static_assert( SuperentityTraits< Superdimensions >::storageEnabled, "You try to bind superentities which are not configured for storage." );
      BaseType::bindSuperentitiesStorageNetwork( MeshDimensionsTag< Superdimensions >(),
                                                 storage );
   }

   template< int Superdimensions >
   bool setNumberOfSuperentities( const typename SuperentityTraits< Superdimensions >::LocalIndexType size )
   {
      static_assert( SuperentityTraits< Superdimensions >::storageEnabled, "You try to set number of superentities which are not configured for storage." );
      return BaseType::setNumberOfSuperentities( MeshDimensionsTag< Superdimensions >(),
                                                 size );
   }

   template< int Superdimensions >
   typename SuperentityTraits< Superdimensions >::LocalIndexType
   getNumberOfSuperentities() const
   {
      static_assert( SuperentityTraits< Superdimensions >::storageEnabled, "You try to get number of superentities which are not configured for storage." );
      return BaseType::getNumberOfSuperentities( MeshDimensionsTag< Superdimensions >() );
   }

   template< int Superdimensions >
   void
   setSuperentityIndex( const typename SuperentityTraits< Superdimensions >::LocalIndexType& localIndex,
                        const typename SuperentityTraits< Superdimensions >::GlobalIndexType& globalIndex )
   {
      static_assert( SuperentityTraits< Superdimensions >::storageEnabled, "You try to set superentities which are not configured for storage." );
      BaseType::setSuperentityIndex( MeshDimensionsTag< Superdimensions >(),
                                     localIndex,
                                     globalIndex );
   }

   template< int Superdimensions >
   typename SuperentityTraits< Superdimensions >::GlobalIndexType
   getSuperentityIndex( const typename SuperentityTraits< Superdimensions >::LocalIndexType localIndex ) const
   {
      static_assert( SuperentityTraits< Superdimensions >::storageEnabled, "You try to get superentities which are not configured for storage." );
      return BaseType::getSuperentityIndex( MeshDimensionsTag< Superdimensions >(),
                                            localIndex );
   }

   bool operator==( const MeshSuperentityAccess& other ) const
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
class MeshSuperentityAccessLayer< MeshConfig,
                                  EntityTopology,
                                  DimensionsTag,
                                  true >
   : public MeshSuperentityAccessLayer< MeshConfig, EntityTopology, typename DimensionsTag::Decrement >
{
	using BaseType = MeshSuperentityAccessLayer< MeshConfig, EntityTopology, typename DimensionsTag::Decrement >;

   using MeshTraitsType = MeshTraits< MeshConfig >;
   using SuperentityTraitsType = typename MeshTraitsType::template SuperentityTraits< EntityTopology, DimensionsTag::value >;

public:
   using GlobalIndexType         = typename SuperentityTraitsType::GlobalIndexType;
   using LocalIndexType          = typename SuperentityTraitsType::LocalIndexType;
   using StorageNetworkType      = typename SuperentityTraitsType::StorageNetworkType;
   using SuperentityAccessorType = typename SuperentityTraitsType::SuperentityAccessorType;

   /****
     * Make visible setters and getters of the lower superentities
     */
   using BaseType::bindSuperentitiesStorageNetwork;
   using BaseType::setNumberOfSuperentities;
   using BaseType::getNumberOfSuperentities;
   using BaseType::setSuperentityIndex;
   using BaseType::getSuperentityIndex;
   using BaseType::getSuperentityIndices;

   MeshSuperentityAccessLayer() = default;

   explicit MeshSuperentityAccessLayer( const MeshSuperentityAccessLayer& layer )
      : BaseType( layer )
   {
      this->superentityIndices.bind( layer.superentityIndices );
   }

   MeshSuperentityAccessLayer& operator=( const MeshSuperentityAccessLayer& layer )
   {
      BaseType::operator=( layer );
      this->superentityIndices.bind( layer.superentityIndices );
   }

   /****
    * Define setter/getter for the current level of the superentities
    */
   void bindSuperentitiesStorageNetwork( DimensionsTag,
                                         const SuperentityAccessorType& storage )
   {
      this->superentityIndices.bind( storage );
   }

   bool setNumberOfSuperentities( DimensionsTag,
                                  const LocalIndexType size )
   {
      return this->superentityIndices.setSize( size );
   }

   LocalIndexType getNumberOfSuperentities( DimensionsTag ) const
   {
      return this->superentityIndices.getSize();
   }

   void setSuperentityIndex( DimensionsTag,
                             const LocalIndexType& localIndex,
                             const GlobalIndexType& globalIndex )
   {
      this->superentityIndices[ localIndex ] = globalIndex;
   }

   GlobalIndexType getSuperentityIndex( DimensionsTag,
                                        const LocalIndexType localIndex ) const
   {
      return this->superentityIndices[ localIndex ];
   }

   const SuperentityAccessorType& getSuperentityIndices( DimensionsTag ) const
   {
      return this->superentityIndices;
   }

   SuperentityAccessorType& getSuperentityIndices( DimensionsTag )
   {
      return this->superentityIndices;
   }

   bool operator==( const MeshSuperentityAccessLayer& other ) const
   {
      return ( BaseType::operator==( other ) && superentityIndices == other.superentityIndices );
   }

   void print( std::ostream& str ) const
   {
      BaseType::print( str );
      str << "\t Superentities with " << DimensionsTag::value << " dimensions are: " << this->superentityIndices << "." << std::endl;
   }

private:
   SuperentityAccessorType superentityIndices;
};

template< typename MeshConfig,
          typename EntityTopology,
          typename DimensionsTag >
class MeshSuperentityAccessLayer< MeshConfig,
                                  EntityTopology,
                                  DimensionsTag,
                                  false >
   : public MeshSuperentityAccessLayer< MeshConfig, EntityTopology, typename DimensionsTag::Decrement >
{
};

template< typename MeshConfig,
          typename EntityTopology >
class MeshSuperentityAccessLayer< MeshConfig,
                                  EntityTopology,
                                  MeshDimensionsTag< EntityTopology::dimensions >,
                                  false >
{
   static constexpr int Dimensions = EntityTopology::dimensions;
   using DimensionsTag = MeshDimensionsTag< EntityTopology::dimensions >;
   using MeshTraitsType = MeshTraits< MeshConfig >;
   using SuperentityTraitsType = typename MeshTraitsType::template SuperentityTraits< EntityTopology, DimensionsTag::value >;

protected:
   using GlobalIndexType         = typename SuperentityTraitsType::GlobalIndexType;
   using LocalIndexType          = typename SuperentityTraitsType::LocalIndexType;
   using SuperentityAccessorType = typename SuperentityTraitsType::SuperentityAccessorType;

   /***
    * Necessary because of 'using BaseType::...;' in the derived classes
    */
   void bindSuperentitiesStorageNetwork( DimensionsTag,
                                         const SuperentityAccessorType& storage ) {}
   void setNumberOfSuperentities( DimensionsTag,
                                  const LocalIndexType size ) {}
   void getNumberOfSuperentities( DimensionsTag ) const {}
   void getSuperentityIndex( DimensionsTag,
                             const LocalIndexType localIndex ) const {}
   void setSuperentityIndex( DimensionsTag,
                             const LocalIndexType& localIndex,
                             const GlobalIndexType& globalIndex ) {}

   void getSuperentityIndices() {}

   bool operator==( const MeshSuperentityAccess< MeshConfig, EntityTopology >& other ) const
   {
      return true;
   }

   void print( std::ostream& str ) const {}
};

template< typename MeshConfig,
          typename EntityTopology >
class MeshSuperentityAccessLayer< MeshConfig,
                                  EntityTopology,
                                  MeshDimensionsTag< EntityTopology::dimensions >,
                                  true >
{
   static constexpr int Dimensions = EntityTopology::dimensions;
   using DimensionsTag = MeshDimensionsTag< EntityTopology::dimensions >;
   using MeshTraitsType = MeshTraits< MeshConfig >;
   using SuperentityTraitsType = typename MeshTraitsType::template SuperentityTraits< EntityTopology, DimensionsTag::value >;

protected:
   using GlobalIndexType         = typename SuperentityTraitsType::GlobalIndexType;
   using LocalIndexType          = typename SuperentityTraitsType::LocalIndexType;
   using SuperentityAccessorType = typename SuperentityTraitsType::SuperentityAccessorType;

   /***
    * Necessary because of 'using BaseType::...;' in the derived classes
    */
   void bindSuperentitiesStorageNetwork( DimensionsTag,
                                         const SuperentityAccessorType& storage ) {}
   void setNumberOfSuperentities( DimensionsTag,
                                  const LocalIndexType size ) {}
   void getNumberOfSuperentities( DimensionsTag ) const {}
   void getSuperentityIndex( DimensionsTag,
                             const LocalIndexType localIndex ) const {}
   void setSuperentityIndex( DimensionsTag,
                             const LocalIndexType& localIndex,
                             const GlobalIndexType& globalIndex ) {}

   void getSuperentityIndices() {}

   bool operator==( const MeshSuperentityAccessLayer& other ) const
   {
      return true;
   }

   void print( std::ostream& str ) const {}
};

} // namespace Meshes
} // namespace TNL

