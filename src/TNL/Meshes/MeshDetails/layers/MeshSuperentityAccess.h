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
   bool operator == ( const MeshSuperentityAccess< MeshConfig, EntityTopology >& a ) const { return true; } // TODO: fix

   void print( std::ostream& str ) const
   {
      BaseType::print( str );
   }

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
   using IdArrayAccessorType     = typename MeshTraitsType::IdArrayAccessorType;
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

   using BaseType::superentityIds;
   IdArrayAccessorType superentityIds( DimensionsTag ) const { return m_superentityIndices; }

   using BaseType::superentityIdsArray;
   IdArrayAccessorType &superentityIdsArray( DimensionsTag ) { return m_superentityIndices; }

   using BaseType::getSuperentityIndices;
   const SuperentityAccessorType& getSuperentityIndices( DimensionsTag ) const
   {
      return this->superentityIndices;
   }

   SuperentityAccessorType& getSuperentityIndices( DimensionsTag )
   {
      return this->superentityIndices;
   }

   void print( std::ostream& str ) const
   {
      BaseType::print( str );
      str << "\t Superentities with " << DimensionsTag::value << " dimensions are: " << this->superentityIndices << "." << std::endl;
   }

   //bool operator == ( const MeshSuperentityAccessLayer< MeshConfig, EntityTopology, Dimensions, tnlStorageTraits< true > >& l ) { return true; } // TODO: fix

private:
   // TODO: used only in mesh initializer, should be removed
   IdArrayAccessorType m_superentityIndices;

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

   void superentityIds()      {}
   void superentityIdsArray() {}

   void getSuperentityIndices() {}

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

   void superentityIds()      {}
   void superentityIdsArray() {}

   void getSuperentityIndices() {}

   void print( std::ostream& str ) const {}
};

} // namespace Meshes
} // namespace TNL

