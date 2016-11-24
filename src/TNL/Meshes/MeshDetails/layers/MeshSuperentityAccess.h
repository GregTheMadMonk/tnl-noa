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
          typename DimensionTag,
          bool SuperentityStorage =
             MeshTraits< MeshConfig >::template SuperentityTraits< EntityTopology, DimensionTag::value >::storageEnabled >
class MeshSuperentityAccessLayer;


template< typename MeshConfig,
          typename EntityTopology >
class MeshSuperentityAccess
   : public MeshSuperentityAccessLayer< MeshConfig,
                                        EntityTopology,
                                        Meshes::DimensionTag< MeshTraits< MeshConfig >::meshDimension > >
{
   using BaseType = MeshSuperentityAccessLayer< MeshConfig,
                                                EntityTopology,
                                                Meshes::DimensionTag< MeshTraits< MeshConfig >::meshDimension > >;

   using MeshTraitsType = MeshTraits< MeshConfig >;

   template< int Superdimension >
   using SuperentityTraits = typename MeshTraitsType::template SuperentityTraits< EntityTopology, Superdimension >;

public:
   template< int Superdimension >
   void bindSuperentitiesStorageNetwork( const typename SuperentityTraits< Superdimension >::SuperentityAccessorType& storage )
   {
      static_assert( SuperentityTraits< Superdimension >::storageEnabled, "You try to bind superentities which are not configured for storage." );
      BaseType::bindSuperentitiesStorageNetwork( Meshes::DimensionTag< Superdimension >(),
                                                 storage );
   }

   template< int Superdimension >
   bool setNumberOfSuperentities( const typename SuperentityTraits< Superdimension >::LocalIndexType size )
   {
      static_assert( SuperentityTraits< Superdimension >::storageEnabled, "You try to set number of superentities which are not configured for storage." );
      return BaseType::setNumberOfSuperentities( Meshes::DimensionTag< Superdimension >(),
                                                 size );
   }

   template< int Superdimension >
   typename SuperentityTraits< Superdimension >::LocalIndexType
   getNumberOfSuperentities() const
   {
      static_assert( SuperentityTraits< Superdimension >::storageEnabled, "You try to get number of superentities which are not configured for storage." );
      return BaseType::getNumberOfSuperentities( Meshes::DimensionTag< Superdimension >() );
   }

   template< int Superdimension >
   void
   setSuperentityIndex( const typename SuperentityTraits< Superdimension >::LocalIndexType& localIndex,
                        const typename SuperentityTraits< Superdimension >::GlobalIndexType& globalIndex )
   {
      static_assert( SuperentityTraits< Superdimension >::storageEnabled, "You try to set superentities which are not configured for storage." );
      BaseType::setSuperentityIndex( Meshes::DimensionTag< Superdimension >(),
                                     localIndex,
                                     globalIndex );
   }

   template< int Superdimension >
   typename SuperentityTraits< Superdimension >::GlobalIndexType
   getSuperentityIndex( const typename SuperentityTraits< Superdimension >::LocalIndexType localIndex ) const
   {
      static_assert( SuperentityTraits< Superdimension >::storageEnabled, "You try to get superentities which are not configured for storage." );
      return BaseType::getSuperentityIndex( Meshes::DimensionTag< Superdimension >(),
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
          typename DimensionTag >
class MeshSuperentityAccessLayer< MeshConfig,
                                  EntityTopology,
                                  DimensionTag,
                                  true >
   : public MeshSuperentityAccessLayer< MeshConfig, EntityTopology, typename DimensionTag::Decrement >
{
	using BaseType = MeshSuperentityAccessLayer< MeshConfig, EntityTopology, typename DimensionTag::Decrement >;

   using MeshTraitsType = MeshTraits< MeshConfig >;
   using SuperentityTraitsType = typename MeshTraitsType::template SuperentityTraits< EntityTopology, DimensionTag::value >;

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
      return *this;
   }

   /****
    * Define setter/getter for the current level of the superentities
    */
   void bindSuperentitiesStorageNetwork( DimensionTag,
                                         const SuperentityAccessorType& storage )
   {
      this->superentityIndices.bind( storage );
   }

   bool setNumberOfSuperentities( DimensionTag,
                                  const LocalIndexType size )
   {
      return this->superentityIndices.setSize( size );
   }

   LocalIndexType getNumberOfSuperentities( DimensionTag ) const
   {
      return this->superentityIndices.getSize();
   }

   void setSuperentityIndex( DimensionTag,
                             const LocalIndexType& localIndex,
                             const GlobalIndexType& globalIndex )
   {
      this->superentityIndices[ localIndex ] = globalIndex;
   }

   GlobalIndexType getSuperentityIndex( DimensionTag,
                                        const LocalIndexType localIndex ) const
   {
      return this->superentityIndices[ localIndex ];
   }

   const SuperentityAccessorType& getSuperentityIndices( DimensionTag ) const
   {
      return this->superentityIndices;
   }

   SuperentityAccessorType& getSuperentityIndices( DimensionTag )
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
      str << "\t Superentities with " << DimensionTag::value << " dimension are: " << this->superentityIndices << "." << std::endl;
   }

private:
   SuperentityAccessorType superentityIndices;
};

template< typename MeshConfig,
          typename EntityTopology,
          typename DimensionTag >
class MeshSuperentityAccessLayer< MeshConfig,
                                  EntityTopology,
                                  DimensionTag,
                                  false >
   : public MeshSuperentityAccessLayer< MeshConfig, EntityTopology, typename DimensionTag::Decrement >
{
};

template< typename MeshConfig,
          typename EntityTopology >
class MeshSuperentityAccessLayer< MeshConfig,
                                  EntityTopology,
                                  Meshes::DimensionTag< EntityTopology::dimension >,
                                  false >
{
   using DimensionTag = Meshes::DimensionTag< EntityTopology::dimension >;
   using MeshTraitsType = MeshTraits< MeshConfig >;
   using SuperentityTraitsType = typename MeshTraitsType::template SuperentityTraits< EntityTopology, DimensionTag::value >;

protected:
   using GlobalIndexType         = typename SuperentityTraitsType::GlobalIndexType;
   using LocalIndexType          = typename SuperentityTraitsType::LocalIndexType;
   using SuperentityAccessorType = typename SuperentityTraitsType::SuperentityAccessorType;

   /***
    * Necessary because of 'using BaseType::...;' in the derived classes
    */
   void bindSuperentitiesStorageNetwork( DimensionTag,
                                         const SuperentityAccessorType& storage ) {}
   void setNumberOfSuperentities( DimensionTag,
                                  const LocalIndexType size ) {}
   void getNumberOfSuperentities( DimensionTag ) const {}
   void getSuperentityIndex( DimensionTag,
                             const LocalIndexType localIndex ) const {}
   void setSuperentityIndex( DimensionTag,
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
                                  Meshes::DimensionTag< EntityTopology::dimension >,
                                  true >
{
   using DimensionTag = Meshes::DimensionTag< EntityTopology::dimension >;
   using MeshTraitsType = MeshTraits< MeshConfig >;
   using SuperentityTraitsType = typename MeshTraitsType::template SuperentityTraits< EntityTopology, DimensionTag::value >;

protected:
   using GlobalIndexType         = typename SuperentityTraitsType::GlobalIndexType;
   using LocalIndexType          = typename SuperentityTraitsType::LocalIndexType;
   using SuperentityAccessorType = typename SuperentityTraitsType::SuperentityAccessorType;

   /***
    * Necessary because of 'using BaseType::...;' in the derived classes
    */
   void bindSuperentitiesStorageNetwork( DimensionTag,
                                         const SuperentityAccessorType& storage ) {}
   void setNumberOfSuperentities( DimensionTag,
                                  const LocalIndexType size ) {}
   void getNumberOfSuperentities( DimensionTag ) const {}
   void getSuperentityIndex( DimensionTag,
                             const LocalIndexType localIndex ) const {}
   void setSuperentityIndex( DimensionTag,
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

