/***************************************************************************
                          SuperentityAccess.h  -  description
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
          typename Device,
          typename EntityTopology,
          typename DimensionTag,
          bool SuperentityStorage = WeakSuperentityStorageTrait< MeshConfig, Device, EntityTopology, DimensionTag >::storageEnabled >
class SuperentityAccessLayer;


template< typename MeshConfig,
          typename Device,
          typename EntityTopology >
class SuperentityAccessLayerFamily
   : public SuperentityAccessLayer< MeshConfig,
                                    Device,
                                    EntityTopology,
                                    Meshes::DimensionTag< MeshTraits< MeshConfig, Device >::meshDimension > >
{
   using BaseType = SuperentityAccessLayer< MeshConfig,
                                            Device,
                                            EntityTopology,
                                            Meshes::DimensionTag< MeshTraits< MeshConfig, Device >::meshDimension > >;

   using MeshTraitsType = MeshTraits< MeshConfig, Device >;

   template< int Superdimension >
   using SuperentityTraits = typename MeshTraitsType::template SuperentityTraits< EntityTopology, Superdimension >;

public:
   template< int Superdimension >
   __cuda_callable__
   void bindSuperentitiesStorageNetwork( const typename SuperentityTraits< Superdimension >::SuperentityAccessorType& storage )
   {
      static_assert( SuperentityTraits< Superdimension >::storageEnabled, "You try to bind superentities which are not configured for storage." );
      BaseType::bindSuperentitiesStorageNetwork( Meshes::DimensionTag< Superdimension >(),
                                                 storage );
   }

   template< int Superdimension >
   __cuda_callable__
   void setNumberOfSuperentities( const typename MeshTraitsType::LocalIndexType size )
   {
      static_assert( SuperentityTraits< Superdimension >::storageEnabled, "You try to set number of superentities which are not configured for storage." );
      BaseType::setNumberOfSuperentities( Meshes::DimensionTag< Superdimension >(),
                                          size );
   }

   template< int Superdimension >
   __cuda_callable__
   typename MeshTraitsType::LocalIndexType
   getSuperentitiesCount() const
   {
      static_assert( SuperentityTraits< Superdimension >::storageEnabled, "You try to get number of superentities which are not configured for storage." );
      return BaseType::getSuperentitiesCount( Meshes::DimensionTag< Superdimension >() );
   }

   template< int Superdimension >
   __cuda_callable__
   void
   setSuperentityIndex( const typename MeshTraitsType::LocalIndexType& localIndex,
                        const typename MeshTraitsType::GlobalIndexType& globalIndex )
   {
      static_assert( SuperentityTraits< Superdimension >::storageEnabled, "You try to set superentities which are not configured for storage." );
      BaseType::setSuperentityIndex( Meshes::DimensionTag< Superdimension >(),
                                     localIndex,
                                     globalIndex );
   }

   template< int Superdimension >
   __cuda_callable__
   typename MeshTraitsType::GlobalIndexType
   getSuperentityIndex( const typename MeshTraitsType::LocalIndexType localIndex ) const
   {
      static_assert( SuperentityTraits< Superdimension >::storageEnabled, "You try to get superentities which are not configured for storage." );
      return BaseType::getSuperentityIndex( Meshes::DimensionTag< Superdimension >(),
                                            localIndex );
   }

   __cuda_callable__
   bool operator==( const SuperentityAccessLayerFamily& other ) const
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
class SuperentityAccessLayer< MeshConfig,
                              Device,
                              EntityTopology,
                              DimensionTag,
                              true >
   : public SuperentityAccessLayer< MeshConfig, Device, EntityTopology, typename DimensionTag::Decrement >
{
	using BaseType = SuperentityAccessLayer< MeshConfig, Device, EntityTopology, typename DimensionTag::Decrement >;
   using MeshTraitsType = MeshTraits< MeshConfig, Device >;
   using SuperentityTraitsType = typename MeshTraitsType::template SuperentityTraits< EntityTopology, DimensionTag::value >;

public:
   using GlobalIndexType         = typename MeshTraitsType::GlobalIndexType;
   using LocalIndexType          = typename MeshTraitsType::LocalIndexType;
   using SuperentityAccessorType = typename SuperentityTraitsType::SuperentityAccessorType;

   /****
     * Make visible setters and getters of the lower superentities
     */
   using BaseType::bindSuperentitiesStorageNetwork;
   using BaseType::setNumberOfSuperentities;
   using BaseType::getSuperentitiesCount;
   using BaseType::setSuperentityIndex;
   using BaseType::getSuperentityIndex;
   using BaseType::getSuperentityIndices;

   SuperentityAccessLayer() = default;

   __cuda_callable__
   explicit SuperentityAccessLayer( const SuperentityAccessLayer& layer )
      : BaseType( layer )
   {
      this->superentityIndices.bind( layer.superentityIndices );
   }

   __cuda_callable__
   SuperentityAccessLayer& operator=( const SuperentityAccessLayer& layer )
   {
      BaseType::operator=( layer );
      this->superentityIndices.bind( layer.superentityIndices );
      return *this;
   }

   /****
    * Define setter/getter for the current level of the superentities
    */
   __cuda_callable__
   void bindSuperentitiesStorageNetwork( DimensionTag,
                                         const SuperentityAccessorType& storage )
   {
      this->superentityIndices.bind( storage );
   }

   __cuda_callable__
   void setNumberOfSuperentities( DimensionTag,
                                  const LocalIndexType size )
   {
      this->superentityIndices.setSize( size );
   }

   __cuda_callable__
   LocalIndexType getSuperentitiesCount( DimensionTag ) const
   {
      return this->superentityIndices.getSize();
   }

   __cuda_callable__
   void setSuperentityIndex( DimensionTag,
                             const LocalIndexType& localIndex,
                             const GlobalIndexType& globalIndex )
   {
      this->superentityIndices[ localIndex ] = globalIndex;
   }

   __cuda_callable__
   GlobalIndexType getSuperentityIndex( DimensionTag,
                                        const LocalIndexType localIndex ) const
   {
      return this->superentityIndices[ localIndex ];
   }

   __cuda_callable__
   const SuperentityAccessorType& getSuperentityIndices( DimensionTag ) const
   {
      return this->superentityIndices;
   }

   __cuda_callable__
   SuperentityAccessorType& getSuperentityIndices( DimensionTag )
   {
      return this->superentityIndices;
   }

   __cuda_callable__
   bool operator==( const SuperentityAccessLayer& other ) const
   {
      return ( BaseType::operator==( other ) && superentityIndices == other.superentityIndices );
   }

   void print( std::ostream& str ) const
   {
      BaseType::print( str );
      str << "\t Superentities with dimension " << DimensionTag::value << " are: " << this->superentityIndices << "." << std::endl;
   }

private:
   SuperentityAccessorType superentityIndices;
};

template< typename MeshConfig,
          typename Device,
          typename EntityTopology,
          typename DimensionTag >
class SuperentityAccessLayer< MeshConfig,
                              Device,
                              EntityTopology,
                              DimensionTag,
                              false >
   : public SuperentityAccessLayer< MeshConfig, Device, EntityTopology, typename DimensionTag::Decrement >
{
};

// termination of recursive inheritance (everything is reduced to EntityStorage == false thanks to the WeakSuperentityStorageTrait)
template< typename MeshConfig,
          typename Device,
          typename EntityTopology >
class SuperentityAccessLayer< MeshConfig,
                              Device,
                              EntityTopology,
                              Meshes::DimensionTag< EntityTopology::dimension >,
                              false >
{
   using DimensionTag = Meshes::DimensionTag< EntityTopology::dimension >;

protected:
   using GlobalIndexType = typename MeshConfig::GlobalIndexType;
   using LocalIndexType  = typename MeshConfig::LocalIndexType;

   /***
    * Necessary because of 'using BaseType::...;' in the derived classes
    */
   template< typename SuperentityAccessorType >
   __cuda_callable__
   void bindSuperentitiesStorageNetwork( DimensionTag,
                                         const SuperentityAccessorType& storage ) {}
   __cuda_callable__
   void setNumberOfSuperentities( DimensionTag,
                                  const LocalIndexType size ) {}
   __cuda_callable__
   void getSuperentitiesCount( DimensionTag ) const {}
   __cuda_callable__
   void getSuperentityIndex( DimensionTag,
                             const LocalIndexType localIndex ) const {}
   __cuda_callable__
   void setSuperentityIndex( DimensionTag,
                             const LocalIndexType& localIndex,
                             const GlobalIndexType& globalIndex ) {}
   __cuda_callable__
   void getSuperentityIndices() {}

   __cuda_callable__
   bool operator==( const SuperentityAccessLayer& other ) const
   {
      return true;
   }

   void print( std::ostream& str ) const {}
};

} // namespace Meshes
} // namespace TNL
