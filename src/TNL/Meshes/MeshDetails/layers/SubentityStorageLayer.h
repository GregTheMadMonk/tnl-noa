/***************************************************************************
                          SubentityStorageLayer.h  -  description
                             -------------------
    begin                : Feb 11, 2014
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
          typename EntityTopology,
          typename SubdimensionTag,
          bool SubentityStorage = WeakSubentityStorageTrait< MeshConfig, Device, EntityTopology, SubdimensionTag >::storageEnabled >
class SubentityStorageLayer;

template< typename MeshConfig,
          typename Device,
          typename EntityTopology >
class SubentityStorageLayerFamily
   : public SubentityStorageLayer< MeshConfig, Device, EntityTopology, DimensionTag< 0 > >
{
   using BaseType = SubentityStorageLayer< MeshConfig, Device, EntityTopology, DimensionTag< 0 > >;
   using MeshTraitsType = MeshTraits< MeshConfig, Device >;

public:
   // inherit constructors and assignment operators (including templated versions)
   using BaseType::BaseType;
   using BaseType::operator=;

protected:
   template< int Subdimension >
   __cuda_callable__
   typename MeshTraitsType::LocalIndexType
   getSubentitiesCount() const
   {
      static_assert( EntityTopology::dimension > Subdimension, "Invalid combination of Dimension and Subdimension." );
      return BaseType::getSubentitiesCount( DimensionTag< Subdimension >() );
   }

   template< int Subdimension >
   __cuda_callable__
   typename MeshTraitsType::SubentityMatrixType&
   getSubentitiesMatrix()
   {
      static_assert( EntityTopology::dimension > Subdimension, "Invalid combination of Dimension and Subdimension." );
      return BaseType::getSubentitiesMatrix( DimensionTag< Subdimension >() );
   }

   template< int Subdimension >
   __cuda_callable__
   const typename MeshTraitsType::SubentityMatrixType&
   getSubentitiesMatrix() const
   {
      static_assert( EntityTopology::dimension > Subdimension, "Invalid combination of Dimension and Subdimension." );
      return BaseType::getSubentitiesMatrix( DimensionTag< Subdimension >() );
   }
};

template< typename MeshConfig,
          typename Device,
          typename EntityTopology,
          typename SubdimensionTag >
class SubentityStorageLayer< MeshConfig,
                             Device,
                             EntityTopology,
                             SubdimensionTag,
                             true >
   : public SubentityStorageLayer< MeshConfig, Device, EntityTopology, typename SubdimensionTag::Increment >
{
   using BaseType = SubentityStorageLayer< MeshConfig, Device, EntityTopology, typename SubdimensionTag::Increment >;
   using MeshTraitsType      = MeshTraits< MeshConfig, Device >;

protected:
   using GlobalIndexType    = typename MeshTraitsType::GlobalIndexType;
   using LocalIndexType     = typename MeshTraitsType::LocalIndexType;
   using SubentityMatrixType = typename MeshTraitsType::SubentityMatrixType;

   SubentityStorageLayer() = default;

   explicit SubentityStorageLayer( const SubentityStorageLayer& other )
   {
      operator=( other );
   }

   template< typename Device_ >
   SubentityStorageLayer( const SubentityStorageLayer< MeshConfig, Device_, EntityTopology, SubdimensionTag >& other )
   {
      operator=( other );
   }

   SubentityStorageLayer& operator=( const SubentityStorageLayer& other )
   {
      BaseType::operator=( other );
      matrix = other.matrix;
      return *this;
   }

   template< typename Device_ >
   SubentityStorageLayer& operator=( const SubentityStorageLayer< MeshConfig, Device_, EntityTopology, SubdimensionTag >& other )
   {
      BaseType::operator=( other );
      matrix = other.matrix;
      return *this;
   }


   void print( std::ostream& str ) const
   {
      BaseType::print( str );
      str << "Adjacency matrix for subentities with dimension " << SubdimensionTag::value << " of entities with dimension " << EntityTopology::dimension << " is: " << std::endl;
      str << matrix << std::endl;
   }

   bool operator==( const SubentityStorageLayer& layer ) const
   {
      return ( BaseType::operator==( layer ) &&
               matrix == layer.matrix );
   }

protected:
   using BaseType::getSubentitiesCount;
   __cuda_callable__
   LocalIndexType getSubentitiesCount( SubdimensionTag ) const
   {
      using SubentityTraitsType = typename MeshTraitsType::template SubentityTraits< EntityTopology, SubdimensionTag::value >;
      return SubentityTraitsType::count;
   }

   using BaseType::getSubentitiesMatrix;
   __cuda_callable__
   SubentityMatrixType& getSubentitiesMatrix( SubdimensionTag )
   {
      return matrix;
   }

   __cuda_callable__
   const SubentityMatrixType& getSubentitiesMatrix( SubdimensionTag ) const
   {
      return matrix;
   }

private:
   SubentityMatrixType matrix;

   // friend class is needed for templated assignment operators
   template< typename MeshConfig_, typename Device_, typename EntityTopology_, typename SubdimensionTag_, bool Storage_ >
   friend class SubentityStorageLayer;
};

template< typename MeshConfig,
          typename Device,
          typename EntityTopology,
          typename SubdimensionTag >
class SubentityStorageLayer< MeshConfig,
                             Device,
                             EntityTopology,
                             SubdimensionTag,
                             false >
   : public SubentityStorageLayer< MeshConfig, Device, EntityTopology, typename SubdimensionTag::Increment >
{
   using BaseType = SubentityStorageLayer< MeshConfig, Device, EntityTopology, typename SubdimensionTag::Increment >;
public:
   // inherit constructors and assignment operators (including templated versions)
   using BaseType::BaseType;
   using BaseType::operator=;
};

// termination of recursive inheritance (everything is reduced to EntityStorage == false thanks to the WeakSubentityStorageTrait)
template< typename MeshConfig,
          typename Device,
          typename EntityTopology >
class SubentityStorageLayer< MeshConfig,
                             Device,
                             EntityTopology,
                             DimensionTag< EntityTopology::dimension >,
                             false >
{
   using SubdimensionTag = DimensionTag< EntityTopology::dimension >;

protected:
   using GlobalIndexType = typename MeshConfig::GlobalIndexType;

   SubentityStorageLayer() = default;
   explicit SubentityStorageLayer( const SubentityStorageLayer& other ) {}
   template< typename Device_ >
   SubentityStorageLayer( const SubentityStorageLayer< MeshConfig, Device_, EntityTopology, SubdimensionTag >& other ) {}
   template< typename Device_ >
   SubentityStorageLayer& operator=( const SubentityStorageLayer< MeshConfig, Device_, EntityTopology, SubdimensionTag >& other ) { return *this; }

   void print( std::ostream& str ) const {}

   bool operator==( const SubentityStorageLayer& layer ) const
   {
      return true;
   }

   void getSubentitiesCount( SubdimensionTag ) {}
   void getSubentitiesMatrix( SubdimensionTag ) {}
};

} // namespace Meshes
} // namespace TNL
