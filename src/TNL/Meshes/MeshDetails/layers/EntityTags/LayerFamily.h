/***************************************************************************
                          LayerFamily.h  -  description
                             -------------------
    begin                : Nov 9, 2017
    copyright            : (C) 2017 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include "ConfigValidator.h"
#include "Initializer.h"
#include "Layer.h"
#include "Traits.h"

namespace TNL {
namespace Meshes {
namespace EntityTags {

template< typename MeshConfig, typename Device, typename Dimension = DimensionTag< 0 > >
class LayerInheritor
   : public Layer< MeshConfig, Device, Dimension >,
     public LayerInheritor< MeshConfig, Device, typename Dimension::Increment >
{
   using LayerType = Layer< MeshConfig, Device, Dimension >;
   using BaseType = LayerInheritor< MeshConfig, Device, typename Dimension::Increment >;
protected:
   using LayerType::setEntitiesCount;
   using LayerType::resetEntityTags;
   using LayerType::getEntityTag;
   using LayerType::addEntityTag;
   using LayerType::removeEntityTag;
   using LayerType::isBoundaryEntity;
   using LayerType::isGhostEntity;
   using LayerType::updateEntityTagsLayer;
   using LayerType::getBoundaryEntitiesCount;
   using LayerType::getBoundaryEntityIndex;
   using LayerType::getInteriorEntitiesCount;
   using LayerType::getInteriorEntityIndex;
   using LayerType::getGhostEntitiesCount;
   using LayerType::getGhostEntitiesOffset;

   using BaseType::setEntitiesCount;
   using BaseType::resetEntityTags;
   using BaseType::getEntityTag;
   using BaseType::addEntityTag;
   using BaseType::removeEntityTag;
   using BaseType::isBoundaryEntity;
   using BaseType::isGhostEntity;
   using BaseType::updateEntityTagsLayer;
   using BaseType::getBoundaryEntitiesCount;
   using BaseType::getBoundaryEntityIndex;
   using BaseType::getInteriorEntitiesCount;
   using BaseType::getInteriorEntityIndex;
   using BaseType::getGhostEntitiesCount;
   using BaseType::getGhostEntitiesOffset;


   LayerInheritor() = default;

   explicit LayerInheritor( const LayerInheritor& other )
   {
      operator=( other );
   }

   template< typename Device_ >
   LayerInheritor( const LayerInheritor< MeshConfig, Device_, Dimension >& other )
   {
      operator=( other );
   }

   LayerInheritor& operator=( const LayerInheritor& other )
   {
      LayerType::operator=( other );
      BaseType::operator=( other );
      return *this;
   }

   template< typename Device_ >
   LayerInheritor& operator=( const LayerInheritor< MeshConfig, Device_, Dimension >& other )
   {
      LayerType::operator=( other );
      BaseType::operator=( other );
      return *this;
   }


   void save( File& file ) const
   {
      LayerType::save( file );
      BaseType::save( file );
   }

   void load( File& file )
   {
      LayerType::load( file );
      BaseType::load( file );
   }

   void print( std::ostream& str ) const
   {
      LayerType::print( str );
      BaseType::print( str );
   }

   bool operator==( const LayerInheritor& layer ) const
   {
      return LayerType::operator==( layer ) &&
             BaseType::operator==( layer );
   }
};

template< typename MeshConfig, typename Device >
class LayerInheritor< MeshConfig, Device, DimensionTag< MeshConfig::meshDimension + 1 > >
{
protected:
   void setEntitiesCount();
   void resetEntityTags();
   void getEntityTag() const;
   void addEntityTag();
   void removeEntityTag();
   void isBoundaryEntity() const;
   void isGhostEntity() const;
   void updateEntityTagsLayer();
   void getBoundaryEntitiesCount() const;
   void getBoundaryEntityIndex() const;
   void getInteriorEntitiesCount() const;
   void getInteriorEntityIndex() const;
   void getGhostEntitiesCount() const;
   void getGhostEntitiesOffset() const;

   LayerInheritor() = default;
   explicit LayerInheritor( const LayerInheritor& other ) {}
   template< typename Device_ >
   LayerInheritor( const LayerInheritor< MeshConfig, Device_, DimensionTag< MeshConfig::meshDimension + 1 > >& other ) {}
   LayerInheritor& operator=( const LayerInheritor& other ) { return *this; }
   template< typename Device_ >
   LayerInheritor& operator=( const LayerInheritor< MeshConfig, Device_, DimensionTag< MeshConfig::meshDimension + 1 > >& other ) { return *this; }

   void save( File& file ) const {}
   void load( File& file ) {}

   void print( std::ostream& str ) const {}

   bool operator==( const LayerInheritor& layer ) const
   {
      return true;
   }
};

// Note that MeshType is an incomplete type and therefore cannot be used to access
// MeshType::Config etc. at the time of declaration of this class template.
template< typename MeshConfig, typename Device, typename MeshType >
class LayerFamily
   : public ConfigValidator< MeshConfig >,
     public Initializer< MeshConfig, Device, MeshType >,
     public LayerInheritor< MeshConfig, Device >
{
   using MeshTraitsType = MeshTraits< MeshConfig, Device >;
   using GlobalIndexType = typename MeshTraitsType::GlobalIndexType;
   using TagType = typename MeshTraitsType::EntityTagType;
   using BaseType = LayerInheritor< MeshConfig, Device, DimensionTag< 0 > >;
   template< int Dimension >
   using EntityTraits = typename MeshTraitsType::template EntityTraits< Dimension >;
   template< int Dimension >
   using WeakTrait = WeakStorageTrait< MeshConfig, Device, DimensionTag< Dimension > >;

   friend Initializer< MeshConfig, Device, MeshType >;

public:
   // inherit constructors and assignment operators (including templated versions)
   using BaseType::BaseType;
   using BaseType::operator=;

   template< int Dimension >
   __cuda_callable__
   TagType getEntityTag( const GlobalIndexType& entityIndex ) const
   {
      static_assert( WeakTrait< Dimension >::entityTagsEnabled, "You try to access entity tags which are not configured for storage." );
      return BaseType::getEntityTag( DimensionTag< Dimension >(), entityIndex );
   }

   template< int Dimension >
   __cuda_callable__
   void addEntityTag( const GlobalIndexType& entityIndex, TagType tag )
   {
      static_assert( WeakTrait< Dimension >::entityTagsEnabled, "You try to access entity tags which are not configured for storage." );
      BaseType::addEntityTag( DimensionTag< Dimension >(), entityIndex, tag );
   }

   template< int Dimension >
   __cuda_callable__
   void removeEntityTag( const GlobalIndexType& entityIndex, TagType tag )
   {
      static_assert( WeakTrait< Dimension >::entityTagsEnabled, "You try to access entity tags which are not configured for storage." );
      BaseType::removeEntityTag( DimensionTag< Dimension >(), entityIndex, tag );
   }

   template< int Dimension >
   __cuda_callable__
   bool isBoundaryEntity( const GlobalIndexType& entityIndex ) const
   {
      static_assert( WeakTrait< Dimension >::entityTagsEnabled, "You try to access entity tags which are not configured for storage." );
      return BaseType::isBoundaryEntity( DimensionTag< Dimension >(), entityIndex );
   }

   template< int Dimension >
   __cuda_callable__
   bool isGhostEntity( const GlobalIndexType& entityIndex ) const
   {
      static_assert( WeakTrait< Dimension >::entityTagsEnabled, "You try to access entity tags which are not configured for storage." );
      return BaseType::isGhostEntity( DimensionTag< Dimension >(), entityIndex );
   }

   template< int Dimension >
   __cuda_callable__
   GlobalIndexType getBoundaryEntitiesCount() const
   {
      static_assert( WeakTrait< Dimension >::entityTagsEnabled, "You try to access entity tags which are not configured for storage." );
      return BaseType::getBoundaryEntitiesCount( DimensionTag< Dimension >() );
   }

   template< int Dimension >
   __cuda_callable__
   GlobalIndexType getBoundaryEntityIndex( const GlobalIndexType& i ) const
   {
      static_assert( WeakTrait< Dimension >::entityTagsEnabled, "You try to access entity tags which are not configured for storage." );
      return BaseType::getBoundaryEntityIndex( DimensionTag< Dimension >(), i );
   }

   template< int Dimension >
   __cuda_callable__
   GlobalIndexType getInteriorEntitiesCount() const
   {
      static_assert( WeakTrait< Dimension >::entityTagsEnabled, "You try to access entity tags which are not configured for storage." );
      return BaseType::getInteriorEntitiesCount( DimensionTag< Dimension >() );
   }

   template< int Dimension >
   __cuda_callable__
   GlobalIndexType getInteriorEntityIndex( const GlobalIndexType& i ) const
   {
      static_assert( WeakTrait< Dimension >::entityTagsEnabled, "You try to access entity tags which are not configured for storage." );
      return BaseType::getInteriorEntityIndex( DimensionTag< Dimension >(), i );
   }

   template< int Dimension >
   __cuda_callable__
   GlobalIndexType getGhostEntitiesCount() const
   {
      static_assert( WeakTrait< Dimension >::entityTagsEnabled, "You try to access entity tags which are not configured for storage." );
      return BaseType::getGhostEntitiesCount( DimensionTag< Dimension >() );
   }

   template< int Dimension >
   __cuda_callable__
   GlobalIndexType getGhostEntitiesOffset() const
   {
      static_assert( WeakTrait< Dimension >::entityTagsEnabled, "You try to access entity tags which are not configured for storage." );
      return BaseType::getGhostEntitiesOffset( DimensionTag< Dimension >() );
   }

   template< int Dimension >
   void updateEntityTagsLayer()
   {
      BaseType::updateEntityTagsLayer( DimensionTag< Dimension >() );
   }

protected:
   template< int Dimension >
   void entityTagsSetEntitiesCount( const GlobalIndexType& entitiesCount )
   {
      BaseType::setEntitiesCount( DimensionTag< Dimension >(), entitiesCount );
   }

   template< int Dimension >
   void resetEntityTags()
   {
      BaseType::resetEntityTags( DimensionTag< Dimension >() );
   }
};

} // namespace EntityTags
} // namespace Meshes
} // namespace TNL
