/***************************************************************************
                          SuperentityStorageLayer.h  -  description
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
#include <TNL/Meshes/MeshDetails/traits/WeakStorageTraits.h>

namespace TNL {
namespace Meshes {

template< typename MeshConfig,
          typename Device,
          typename EntityTopologyDimensionTag,
          typename SuperdimensionTag,
          bool SuperentityStorage = WeakSuperentityStorageTrait< MeshConfig, Device, typename MeshTraits< MeshConfig, Device >::template EntityTraits< EntityTopologyDimensionTag::value >::EntityTopology, SuperdimensionTag >::storageEnabled >
class SuperentityStorageLayer;

template< typename MeshConfig,
          typename Device,
          typename EntityTopologyDimensionTag >
class SuperentityStorageLayerFamily
   : public SuperentityStorageLayer< MeshConfig,
                                     Device,
                                     EntityTopologyDimensionTag,
                                     DimensionTag< MeshTraits< MeshConfig, Device >::meshDimension > >
{
   using BaseType = SuperentityStorageLayer< MeshConfig,
                                             Device,
                                             EntityTopologyDimensionTag,
                                             DimensionTag< MeshTraits< MeshConfig, Device >::meshDimension > >;
   using MeshTraitsType = MeshTraits< MeshConfig, Device >;

public:
   // inherit constructors and assignment operators (including templated versions)
   using BaseType::BaseType;
   using BaseType::operator=;

protected:
   template< int Superdimension >
   __cuda_callable__
   typename MeshTraitsType::NeighborCountsArray&
   getSuperentitiesCountsArray()
   {
      static_assert( EntityTopologyDimensionTag::value < Superdimension, "Invalid combination of Dimension and Superdimension." );
      return BaseType::getSuperentitiesCountsArray( DimensionTag< Superdimension >() );
   }

   template< int Superdimension >
   __cuda_callable__
   const typename MeshTraitsType::NeighborCountsArray&
   getSuperentitiesCountsArray() const
   {
      static_assert( EntityTopologyDimensionTag::value < Superdimension, "Invalid combination of Dimension and Superdimension." );
      return BaseType::getSuperentitiesCountsArray( DimensionTag< Superdimension >() );
   }

   template< int Superdimension >
   __cuda_callable__
   typename MeshTraitsType::SuperentityMatrixType&
   getSuperentitiesMatrix()
   {
      static_assert( EntityTopologyDimensionTag::value < Superdimension, "Invalid combination of Dimension and Superdimension." );
      return BaseType::getSuperentitiesMatrix( DimensionTag< Superdimension >() );
   }

   template< int Superdimension >
   __cuda_callable__
   const typename MeshTraitsType::SuperentityMatrixType&
   getSuperentitiesMatrix() const
   {
      static_assert( EntityTopologyDimensionTag::value < Superdimension, "Invalid combination of Dimension and Superdimension." );
      return BaseType::getSuperentitiesMatrix( DimensionTag< Superdimension >() );
   }
};

template< typename MeshConfig,
          typename Device,
          typename EntityTopologyDimensionTag,
          typename SuperdimensionTag >
class SuperentityStorageLayer< MeshConfig, Device, EntityTopologyDimensionTag, SuperdimensionTag, true >
   : public SuperentityStorageLayer< MeshConfig, Device, EntityTopologyDimensionTag, typename SuperdimensionTag::Decrement >
{
   using BaseType = SuperentityStorageLayer< MeshConfig, Device, EntityTopologyDimensionTag, typename SuperdimensionTag::Decrement >;
   using MeshTraitsType = MeshTraits< MeshConfig, Device >;

protected:
   using NeighborCountsArray = typename MeshTraitsType::NeighborCountsArray;
   using SuperentityMatrixType = typename MeshTraitsType::SuperentityMatrixType;

   SuperentityStorageLayer() = default;

   explicit SuperentityStorageLayer( const SuperentityStorageLayer& other )
   {
      operator=( other );
   }

   template< typename Device_ >
   SuperentityStorageLayer( const SuperentityStorageLayer< MeshConfig, Device_, EntityTopologyDimensionTag, SuperdimensionTag >& other )
   {
      operator=( other );
   }

   SuperentityStorageLayer& operator=( const SuperentityStorageLayer& other )
   {
      BaseType::operator=( other );
      superentitiesCounts = other.superentitiesCounts;
      matrix = other.matrix;
      return *this;
   }

   template< typename Device_ >
   SuperentityStorageLayer& operator=( const SuperentityStorageLayer< MeshConfig, Device_, EntityTopologyDimensionTag, SuperdimensionTag >& other )
   {
      BaseType::operator=( other );
      superentitiesCounts = other.superentitiesCounts;
      matrix = other.matrix;
      return *this;
   }


   void print( std::ostream& str ) const
   {
      BaseType::print( str );
      str << "Adjacency matrix for superentities with dimension " << SuperdimensionTag::value << " of entities with dimension " << EntityTopologyDimensionTag::value << " is: " << std::endl;
      str << matrix << std::endl;
   }

   bool operator==( const SuperentityStorageLayer& layer ) const
   {
      return ( BaseType::operator==( layer ) &&
               superentitiesCounts == layer.superentitiesCounts &&
               matrix == layer.matrix );
   }

protected:
   using BaseType::getSuperentitiesCountsArray;
   __cuda_callable__
   NeighborCountsArray& getSuperentitiesCountsArray( SuperdimensionTag )
   {
      return superentitiesCounts;
   }

   __cuda_callable__
   const NeighborCountsArray& getSuperentitiesCountsArray( SuperdimensionTag ) const
   {
      return superentitiesCounts;
   }

   using BaseType::getSuperentitiesMatrix;
   __cuda_callable__
   SuperentityMatrixType& getSuperentitiesMatrix( SuperdimensionTag )
   {
      return matrix;
   }

   __cuda_callable__
   const SuperentityMatrixType& getSuperentitiesMatrix( SuperdimensionTag ) const
   {
      return matrix;
   }

private:
   NeighborCountsArray superentitiesCounts;
   SuperentityMatrixType matrix;

   // friend class is needed for templated assignment operators
   template< typename MeshConfig_, typename Device_, typename EntityTopology_, typename SuperdimensionTag_, bool Storage_ >
   friend class SuperentityStorageLayer;
};

template< typename MeshConfig,
          typename Device,
          typename EntityTopologyDimensionTag,
          typename SuperdimensionTag >
class SuperentityStorageLayer< MeshConfig, Device, EntityTopologyDimensionTag, SuperdimensionTag, false >
   : public SuperentityStorageLayer< MeshConfig, Device, EntityTopologyDimensionTag, typename SuperdimensionTag::Decrement >
{
   using BaseType = SuperentityStorageLayer< MeshConfig, Device, EntityTopologyDimensionTag, typename SuperdimensionTag::Decrement >;
public:
   // inherit constructors and assignment operators (including templated versions)
   using BaseType::BaseType;
   using BaseType::operator=;
};

// termination of recursive inheritance (everything is reduced to EntityStorage == false thanks to the WeakSuperentityStorageTrait)
template< typename MeshConfig,
          typename Device,
          typename EntityTopologyDimensionTag >
class SuperentityStorageLayer< MeshConfig, Device, EntityTopologyDimensionTag, EntityTopologyDimensionTag, false >
{
   using SuperdimensionTag = EntityTopologyDimensionTag;

protected:
   SuperentityStorageLayer() = default;
   explicit SuperentityStorageLayer( const SuperentityStorageLayer& other ) {}
   template< typename Device_ >
   SuperentityStorageLayer( const SuperentityStorageLayer< MeshConfig, Device_, EntityTopologyDimensionTag, SuperdimensionTag >& other ) {}
   template< typename Device_ >
   SuperentityStorageLayer& operator=( const SuperentityStorageLayer< MeshConfig, Device_, EntityTopologyDimensionTag, SuperdimensionTag >& other ) { return *this; }

   void getSuperentitiesCountsArray() {}

   void print( std::ostream& str ) const {}

   bool operator==( const SuperentityStorageLayer& layer ) const
   {
      return true;
   }

   void getSuperentitiesMatrix( SuperdimensionTag ) {}
};

} // namespace Meshes
} // namespace TNL
