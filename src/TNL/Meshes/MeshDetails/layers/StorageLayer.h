/***************************************************************************
                          StorageLayer.h  -  description
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
#include <TNL/Meshes/MeshDetails/traits/WeakStorageTraits.h>
#include <TNL/Meshes/MeshDetails/layers/SubentityStorageLayer.h>
#include <TNL/Meshes/MeshDetails/layers/SubentityOrientationsLayer.h>
#include <TNL/Meshes/MeshDetails/layers/SuperentityStorageLayer.h>
#include <TNL/Meshes/MeshDetails/layers/DualGraphLayer.h>

namespace TNL {
namespace Meshes {

template< typename MeshConfig,
          typename Device,
          typename DimensionTag,
          bool EntityStorage = (DimensionTag::value <= MeshConfig::meshDimension) >
class StorageLayer;


template< typename MeshConfig, typename Device >
class StorageLayerFamily
   : public StorageLayer< MeshConfig, Device, DimensionTag< 0 > >,
     public DualGraphLayer< MeshConfig, Device >
{
   using MeshTraitsType = MeshTraits< MeshConfig, Device >;
   using BaseType       = StorageLayer< MeshConfig, Device, DimensionTag< 0 > >;
   template< int Dimension >
   using EntityTraits = typename MeshTraitsType::template EntityTraits< Dimension >;

   template< int Dimension, int Subdimension >
   using SubentityTraits = typename MeshTraitsType::template SubentityTraits< typename EntityTraits< Dimension >::EntityTopology, Subdimension >;

   template< int Dimension, int Superdimension >
   using SuperentityTraits = typename MeshTraitsType::template SuperentityTraits< typename EntityTraits< Dimension >::EntityTopology, Superdimension >;

protected:
   typename MeshTraitsType::PointArrayType points;

public:
   StorageLayerFamily() = default;

   explicit StorageLayerFamily( const StorageLayerFamily& other )
   {
      operator=( other );
   }

   template< typename Device_ >
   StorageLayerFamily( const StorageLayerFamily< MeshConfig, Device_ >& other )
   {
      operator=( other );
   }

   StorageLayerFamily& operator=( const StorageLayerFamily& layer )
   {
      points = layer.getPoints();
      BaseType::operator=( layer );
      DualGraphLayer< MeshConfig, Device >::operator=( layer );
      return *this;
   }

   template< typename Device_ >
   StorageLayerFamily& operator=( const StorageLayerFamily< MeshConfig, Device_ >& layer )
   {
      points = layer.getPoints();
      BaseType::operator=( layer );
      DualGraphLayer< MeshConfig, Device >::operator=( layer );
      return *this;
   }

   bool operator==( const StorageLayerFamily& layer ) const
   {
      return ( points == layer.points &&
               BaseType::operator==( layer ) &&
               DualGraphLayer< MeshConfig, Device >::operator==( layer ) );
   }

   void save( File& file ) const
   {
      file << points;
      BaseType::save( file );
   }

   void load( File& file )
   {
      file >> points;
      BaseType::load( file );
   }

   void print( std::ostream& str ) const
   {
      str << "Vertex coordinates are: " << points << std::endl;
      BaseType::print( str );
   }

   const typename MeshTraitsType::PointArrayType& getPoints() const
   {
      return points;
   }

   typename MeshTraitsType::PointArrayType& getPoints()
   {
      return points;
   }

   template< int Dimension >
   void setEntitiesCount( const typename MeshTraitsType::GlobalIndexType& entitiesCount )
   {
      BaseType::setEntitiesCount( DimensionTag< Dimension >(), entitiesCount );
      if( Dimension == 0 )
         points.setSize( entitiesCount );
   }

   template< int Dimension, int Subdimension >
   __cuda_callable__
   typename MeshTraitsType::LocalIndexType
   getSubentitiesCount() const
   {
      static_assert( Dimension > Subdimension, "Invalid combination of Dimension and Subdimension." );
      static_assert( SubentityTraits< Dimension, Subdimension >::storageEnabled,
                     "You try to get subentities count for subentities which are disabled in the mesh configuration." );
      using BaseType = SubentityStorageLayerFamily< MeshConfig,
                                                   Device,
                                                   typename EntityTraits< Dimension >::EntityTopology >;
      return BaseType::template getSubentitiesCount< Subdimension >();
   }

   template< int Dimension, int Subdimension >
   __cuda_callable__
   typename MeshTraitsType::SubentityMatrixType&
   getSubentitiesMatrix()
   {
      static_assert( Dimension > Subdimension, "Invalid combination of Dimension and Subdimension." );
      static_assert( SubentityTraits< Dimension, Subdimension >::storageEnabled,
                     "You try to get subentities matrix which is disabled in the mesh configuration." );
      using BaseType = SubentityStorageLayerFamily< MeshConfig,
                                                   Device,
                                                   typename EntityTraits< Dimension >::EntityTopology >;
      return BaseType::template getSubentitiesMatrix< Subdimension >();
   }

   template< int Dimension, int Subdimension >
   __cuda_callable__
   const typename MeshTraitsType::SubentityMatrixType&
   getSubentitiesMatrix() const
   {
      static_assert( Dimension > Subdimension, "Invalid combination of Dimension and Subdimension." );
      static_assert( SubentityTraits< Dimension, Subdimension >::storageEnabled,
                     "You try to get subentities matrix which is disabled in the mesh configuration." );
      using BaseType = SubentityStorageLayerFamily< MeshConfig,
                                                   Device,
                                                   typename EntityTraits< Dimension >::EntityTopology >;
      return BaseType::template getSubentitiesMatrix< Subdimension >();
   }

   template< int Dimension, int Superdimension >
   __cuda_callable__
   typename MeshTraitsType::NeighborCountsArray&
   getSuperentitiesCountsArray()
   {
      static_assert( Dimension < Superdimension, "Invalid combination of Dimension and Superdimension." );
      static_assert( SuperentityTraits< Dimension, Superdimension >::storageEnabled,
                     "You try to get superentities counts array which is disabled in the mesh configuration." );
      using BaseType = SuperentityStorageLayerFamily< MeshConfig,
                                                     Device,
                                                     typename EntityTraits< Dimension >::EntityTopology >;
      return BaseType::template getSuperentitiesCountsArray< Superdimension >();
   }

   template< int Dimension, int Superdimension >
   __cuda_callable__
   const typename MeshTraitsType::NeighborCountsArray&
   getSuperentitiesCountsArray() const
   {
      static_assert( Dimension < Superdimension, "Invalid combination of Dimension and Superdimension." );
      static_assert( SuperentityTraits< Dimension, Superdimension >::storageEnabled,
                     "You try to get superentities counts array which is disabled in the mesh configuration." );
      using BaseType = SuperentityStorageLayerFamily< MeshConfig,
                                                     Device,
                                                     typename EntityTraits< Dimension >::EntityTopology >;
      return BaseType::template getSuperentitiesCountsArray< Superdimension >();
   }

   template< int Dimension, int Superdimension >
   __cuda_callable__
   typename MeshTraitsType::SuperentityMatrixType&
   getSuperentitiesMatrix()
   {
      static_assert( Dimension < Superdimension, "Invalid combination of Dimension and Superdimension." );
      static_assert( SuperentityTraits< Dimension, Superdimension >::storageEnabled,
                     "You try to get superentities matrix which is disabled in the mesh configuration." );
      using BaseType = SuperentityStorageLayerFamily< MeshConfig,
                                                     Device,
                                                     typename EntityTraits< Dimension >::EntityTopology >;
      return BaseType::template getSuperentitiesMatrix< Superdimension >();
   }

   template< int Dimension, int Superdimension >
   __cuda_callable__
   const typename MeshTraitsType::SuperentityMatrixType&
   getSuperentitiesMatrix() const
   {
      static_assert( Dimension < Superdimension, "Invalid combination of Dimension and Superdimension." );
      static_assert( SuperentityTraits< Dimension, Superdimension >::storageEnabled,
                     "You try to get superentities matrix which is disabled in the mesh configuration." );
      using BaseType = SuperentityStorageLayerFamily< MeshConfig,
                                                     Device,
                                                     typename EntityTraits< Dimension >::EntityTopology >;
      return BaseType::template getSuperentitiesMatrix< Superdimension >();
   }

   template< int Dimension, int Subdimension >
   __cuda_callable__
   typename SubentityTraits< Dimension, Subdimension >::IdPermutationArrayType
   getSubentityOrientation( typename MeshTraitsType::GlobalIndexType entityIndex, typename MeshTraitsType::LocalIndexType localIndex ) const
   {
      static_assert( SubentityTraits< Dimension, Subdimension >::orientationEnabled,
                     "You try to get subentity orientation which is not configured for storage." );
      return BaseType::getSubentityOrientation( DimensionTag< Dimension >(), DimensionTag< Subdimension >(), entityIndex, localIndex );
   }

   template< int Dimension, int Subdimension >
   __cuda_callable__
   typename SubentityTraits< Dimension, Subdimension >::OrientationArrayType&
   subentityOrientationsArray( typename MeshTraitsType::GlobalIndexType entityIndex )
   {
      static_assert( SubentityTraits< Dimension, Subdimension >::orientationEnabled,
                     "You try to get subentity orientation which is not configured for storage." );
      return BaseType::subentityOrientationsArray( DimensionTag< Dimension >(), DimensionTag< Subdimension >(), entityIndex );
   }
};


template< typename MeshConfig,
          typename Device,
          typename DimensionTag >
class StorageLayer< MeshConfig,
                    Device,
                    DimensionTag,
                    true >
   : public SubentityStorageLayerFamily< MeshConfig,
                                         Device,
                                         typename MeshTraits< MeshConfig, Device >::template EntityTraits< DimensionTag::value >::EntityTopology >,
     public SubentityOrientationsLayerFamily< MeshConfig,
                                              Device,
                                              typename MeshTraits< MeshConfig, Device >::template EntityTraits< DimensionTag::value >::EntityTopology >,
     public SuperentityStorageLayerFamily< MeshConfig,
                                           Device,
                                           typename MeshTraits< MeshConfig, Device >::template EntityTraits< DimensionTag::value >::EntityTopology >,
     public StorageLayer< MeshConfig, Device, typename DimensionTag::Increment >
{
public:
   using BaseType = StorageLayer< MeshConfig, Device, typename DimensionTag::Increment >;
   using MeshTraitsType   = MeshTraits< MeshConfig, Device >;
   using GlobalIndexType  = typename MeshTraitsType::GlobalIndexType;
   using EntityTraitsType = typename MeshTraitsType::template EntityTraits< DimensionTag::value >;
   using EntityType       = typename EntityTraitsType::EntityType;
   using EntityTopology   = typename EntityTraitsType::EntityTopology;
   using SubentityStorageBaseType = SubentityStorageLayerFamily< MeshConfig, Device, EntityTopology >;
   using SubentityOrientationsBaseType = SubentityOrientationsLayerFamily< MeshConfig, Device, EntityTopology >;
   using SuperentityStorageBaseType = SuperentityStorageLayerFamily< MeshConfig, Device, EntityTopology >;

   using BaseType::subentityOrientationsArray;
   using SubentityOrientationsBaseType::subentityOrientationsArray;

   StorageLayer() = default;

   explicit StorageLayer( const StorageLayer& other )
   {
      operator=( other );
   }

   template< typename Device_ >
   StorageLayer( const StorageLayer< MeshConfig, Device_, DimensionTag >& other )
   {
      operator=( other );
   }

   StorageLayer& operator=( const StorageLayer& other )
   {
      entitiesCount = other.entitiesCount;
      SubentityStorageBaseType::operator=( other );
      SuperentityStorageBaseType::operator=( other );
      BaseType::operator=( other );
      return *this;
   }

   template< typename Device_ >
   StorageLayer& operator=( const StorageLayer< MeshConfig, Device_, DimensionTag >& other )
   {
      entitiesCount = other.getEntitiesCount( DimensionTag() );
      SubentityStorageBaseType::operator=( other );
      SuperentityStorageBaseType::operator=( other );
      BaseType::operator=( other );
      return *this;
   }

   void save( File& file ) const
   {
      SubentityStorageBaseType::save( file );
      SuperentityStorageBaseType::save( file );
      file.save( &entitiesCount, 1 );
      BaseType::save( file );
   }

   void load( File& file )
   {
      SubentityStorageBaseType::load( file );
      SuperentityStorageBaseType::load( file );
      file.load( &entitiesCount, 1 );
      BaseType::load( file );
   }

   void print( std::ostream& str ) const
   {
      str << "Number of entities with dimension " << DimensionTag::value << ": " << entitiesCount << std::endl;
      SubentityStorageBaseType::print( str );
      SuperentityStorageBaseType::print( str );
      str << std::endl;
      BaseType::print( str );
   }

   bool operator==( const StorageLayer& meshLayer ) const
   {
      return ( entitiesCount == meshLayer.entitiesCount &&
               SubentityStorageBaseType::operator==( meshLayer ) &&
               SuperentityStorageBaseType::operator==( meshLayer ) &&
               BaseType::operator==( meshLayer ) );
   }


   using BaseType::getEntitiesCount;
   __cuda_callable__
   GlobalIndexType getEntitiesCount( DimensionTag ) const
   {
      return this->entitiesCount;
   }

protected:
   using BaseType::setEntitiesCount;
   void setEntitiesCount( DimensionTag, const GlobalIndexType& entitiesCount )
   {
      this->entitiesCount = entitiesCount;
      SubentityOrientationsBaseType::setEntitiesCount( entitiesCount );
   }

   GlobalIndexType entitiesCount = 0;

   // friend class is needed for templated assignment operators
   template< typename MeshConfig_, typename Device_, typename DimensionTag_, bool Storage_ >
   friend class StorageLayer;
};

template< typename MeshConfig,
          typename Device >
class StorageLayer< MeshConfig, Device, DimensionTag< MeshConfig::meshDimension + 1 >, false >
{
protected:
   using DimensionTag     = Meshes::DimensionTag< MeshConfig::meshDimension >;
   using GlobalIndexType  = typename MeshConfig::GlobalIndexType;

   StorageLayer() = default;

   explicit StorageLayer( const StorageLayer& other ) {}

   template< typename Device_ >
   StorageLayer( const StorageLayer< MeshConfig, Device_, DimensionTag >& other ) {}

   StorageLayer& operator=( const StorageLayer& other )
   {
      return *this;
   }

   template< typename Device_ >
   StorageLayer& operator=( const StorageLayer< MeshConfig, Device_, DimensionTag >& other )
   {
      return *this;
   }


   void subentityOrientationsArray() {}
   void setEntitiesCount() {}
   void getEntitiesCount() const {}

   void save( File& file ) const {}
   void load( File& file ) {}

   void print( std::ostream& str ) const {}

   bool operator==( const StorageLayer& meshLayer ) const
   {
      return true;
   }
};

} // namespace Meshes
} // namespace TNL
