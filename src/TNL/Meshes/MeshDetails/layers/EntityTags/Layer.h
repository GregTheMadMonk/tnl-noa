/***************************************************************************
                          Layer.h  -  description
                             -------------------
    begin                : Dec 25, 2016
    copyright            : (C) 2016 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/File.h>
#include <TNL/Meshes/MeshDetails/traits/MeshTraits.h>
#include <TNL/Containers/VectorView.h>

#include "Traits.h"

namespace TNL {
namespace Meshes {
namespace EntityTags {

// This is the implementation of the boundary tags layer for one specific dimension.
// It is inherited by the EntityTags::LayerFamily.
template< typename MeshConfig,
          typename Device,
          typename DimensionTag,
          bool TagStorage = WeakStorageTrait< MeshConfig, Device, DimensionTag >::entityTagsEnabled >
class Layer
{
   using MeshTraitsType    = MeshTraits< MeshConfig, Device >;

public:
   using GlobalIndexType     = typename MeshTraitsType::GlobalIndexType;
   using EntityTagsArrayType = typename MeshTraitsType::EntityTagsArrayType;
   using TagType             = typename MeshTraitsType::EntityTagType;
   using OrderingArray       = Containers::Array< GlobalIndexType, Device, GlobalIndexType >;

   Layer() = default;

   explicit Layer( const Layer& other )
   {
      operator=( other );
   }

   template< typename Device_ >
   Layer( const Layer< MeshConfig, Device_, DimensionTag >& other )
   {
      operator=( other );
   }

   Layer& operator=( const Layer& other )
   {
      tags.setLike( other.tags );
      boundaryIndices.setLike( other.boundaryIndices );
      interiorIndices.setLike( other.interiorIndices );
      tags = other.tags;
      boundaryIndices = other.boundaryIndices;
      interiorIndices = other.interiorIndices;
      ghostsOffset = other.ghostsOffset;
      return *this;
   }

   template< typename Device_ >
   Layer& operator=( const Layer< MeshConfig, Device_, DimensionTag >& other )
   {
      tags.setLike( other.tags );
      boundaryIndices.setLike( other.boundaryIndices );
      interiorIndices.setLike( other.interiorIndices );
      tags = other.tags;
      boundaryIndices = other.boundaryIndices;
      interiorIndices = other.interiorIndices;
      ghostsOffset = other.ghostsOffset;
      return *this;
   }


   void setEntitiesCount( DimensionTag, const GlobalIndexType& entitiesCount )
   {
      tags.setSize( entitiesCount );
      ghostsOffset = entitiesCount;
   }

   typename EntityTagsArrayType::ViewType
   getEntityTagsView( DimensionTag )
   {
      return tags.getView();
   }

   typename EntityTagsArrayType::ConstViewType
   getEntityTagsView( DimensionTag ) const
   {
      return tags.getConstView();
   }

   __cuda_callable__
   TagType getEntityTag( DimensionTag, const GlobalIndexType& entityIndex ) const
   {
      return tags[ entityIndex ];
   }

   __cuda_callable__
   void addEntityTag( DimensionTag, const GlobalIndexType& entityIndex, TagType tag )
   {
      tags[ entityIndex ] |= tag;
   }

   __cuda_callable__
   void removeEntityTag( DimensionTag, const GlobalIndexType& entityIndex, TagType tag )
   {
      tags[ entityIndex ] ^= tag;
   }

   __cuda_callable__
   bool isBoundaryEntity( DimensionTag, const GlobalIndexType& entityIndex ) const
   {
      return tags[ entityIndex ] & EntityTags::BoundaryEntity;
   }

   __cuda_callable__
   bool isGhostEntity( DimensionTag, const GlobalIndexType& entityIndex ) const
   {
      return tags[ entityIndex ] & EntityTags::GhostEntity;
   }

   void updateEntityTagsLayer( DimensionTag )
   {
      // count boundary entities - custom reduction because expression templates don't support filtering bits this way:
      //    const GlobalIndexType boundaryEntities = sum(cast< GlobalIndexType >( _tagsVector & EntityTags::BoundaryEntity ));
      // NOTE: boundary/interior entities may overlap with ghost entities, so we count all categories separately
      const auto tags_view = tags.getConstView();
      auto is_boundary = [=] __cuda_callable__ ( GlobalIndexType entityIndex ) -> GlobalIndexType
      {
         return bool(tags_view[ entityIndex ] & EntityTags::BoundaryEntity);
      };
      auto is_ghost = [=] __cuda_callable__ ( GlobalIndexType entityIndex ) -> GlobalIndexType
      {
         return bool(tags_view[ entityIndex ] & EntityTags::GhostEntity);
      };
      const GlobalIndexType boundaryEntities = Algorithms::Reduction< Device >::reduce( (GlobalIndexType) 0, tags.getSize(), is_boundary, std::plus<>{}, (GlobalIndexType) 0 );
      const GlobalIndexType ghostEntities = Algorithms::Reduction< Device >::reduce( (GlobalIndexType) 0, tags.getSize(), is_ghost, std::plus<>{}, (GlobalIndexType) 0 );

      interiorIndices.setSize( tags.getSize() - boundaryEntities );
      boundaryIndices.setSize( boundaryEntities );
      ghostsOffset = tags.getSize() - ghostEntities;

      if( ! std::is_same< Device, Devices::Cuda >::value ) {
         GlobalIndexType i = 0, b = 0;
         for( GlobalIndexType e = 0; e < tags.getSize(); e++ ) {
            if( tags[ e ] & EntityTags::BoundaryEntity )
               boundaryIndices[ b++ ] = e;
            else
               interiorIndices[ i++ ] = e;
            if( tags[ e ] & EntityTags::GhostEntity && ghostEntities > 0 && e < ghostsOffset )
               throw std::runtime_error( "The mesh is inconsistent - ghost entities of dimension " + std::to_string(DimensionTag::value) + " are not ordered after local entities." );
         }
      }
      // TODO: parallelize directly on the device
      else {
         using EntityTagsHostArray = typename EntityTagsArrayType::template Self< typename EntityTagsArrayType::ValueType, Devices::Host >;
         using OrderingHostArray   = typename OrderingArray::template Self< typename OrderingArray::ValueType, Devices::Host >;

         EntityTagsHostArray hostTags;
         OrderingHostArray hostBoundaryIndices, hostInteriorIndices;

         hostTags.setLike( tags );
         hostInteriorIndices.setLike( interiorIndices );
         hostBoundaryIndices.setLike( boundaryIndices );

         hostTags = tags;

         GlobalIndexType i = 0, b = 0;
         for( GlobalIndexType e = 0; e < tags.getSize(); e++ ) {
            if( hostTags[ e ] & EntityTags::BoundaryEntity )
               hostBoundaryIndices[ b++ ] = e;
            else
               hostInteriorIndices[ i++ ] = e;
            if( hostTags[ e ] & EntityTags::GhostEntity && ghostEntities > 0 && e < ghostsOffset )
               throw std::runtime_error( "The mesh is inconsistent - ghost entities of dimension " + std::to_string(DimensionTag::value) + " are not ordered after local entities." );
         }

         interiorIndices = hostInteriorIndices;
         boundaryIndices = hostBoundaryIndices;
      }
   }

   auto getBoundaryIndices( DimensionTag ) const
   {
      return boundaryIndices.getConstView();
   }

   auto getInteriorIndices( DimensionTag ) const
   {
      return interiorIndices.getConstView();
   }

   __cuda_callable__
   GlobalIndexType getGhostEntitiesCount( DimensionTag ) const
   {
      return tags.getSize() - ghostsOffset;
   }

   __cuda_callable__
   GlobalIndexType getGhostEntitiesOffset( DimensionTag ) const
   {
      return ghostsOffset;
   }

   void save( File& file ) const
   {
      file << tags;
   }

   void load( File& file )
   {
      file >> tags;
      updateEntityTagsLayer( DimensionTag() );
   }

   void print( std::ostream& str ) const
   {
      str << "Boundary tags for entities of dimension " << DimensionTag::value << " are: ";
      str << tags << std::endl;
      str << "Indices of the interior entities of dimension " << DimensionTag::value << " are: ";
      str << interiorIndices << std::endl;
      str << "Indices of the boundary entities of dimension " << DimensionTag::value << " are: ";
      str << boundaryIndices << std::endl;
      str << "Index of the first ghost entity of dimension " << DimensionTag::value << " is: ";
      str << ghostsOffset << std::endl;
   }

   bool operator==( const Layer& layer ) const
   {
      TNL_ASSERT( ( tags == layer.tags && interiorIndices == layer.interiorIndices && boundaryIndices == layer.boundaryIndices && ghostsOffset == layer.ghostsOffset ) ||
                  ( tags != layer.tags && interiorIndices != layer.interiorIndices && boundaryIndices != layer.boundaryIndices && ghostsOffset != layer.ghostsOffset ),
                  std::cerr << "The EntityTags layer is in inconsistent state - this is probably a bug in the boundary tags initializer." << std::endl
                            << "tags                  = " << tags << std::endl
                            << "layer.tags            = " << layer.tags << std::endl
                            << "interiorIndices       = " << interiorIndices << std::endl
                            << "layer.interiorIndices = " << layer.interiorIndices << std::endl
                            << "boundaryIndices       = " << boundaryIndices << std::endl
                            << "layer.boundaryIndices = " << layer.boundaryIndices << std::endl
                            << "ghostsOffset          = " << ghostsOffset << std::endl
                            << "layer.ghostsOffset    = " << layer.ghostsOffset << std::endl; );
      return tags == layer.tags;
   }

private:
   EntityTagsArrayType tags;
   OrderingArray interiorIndices, boundaryIndices;
   GlobalIndexType ghostsOffset = 0;

   // friend class is needed for templated assignment operators
   template< typename MeshConfig_, typename Device_, typename DimensionTag_, bool TagStorage_ >
   friend class Layer;
};

template< typename MeshConfig,
          typename Device,
          typename DimensionTag >
class Layer< MeshConfig, Device, DimensionTag, false >
{
protected:
   using GlobalIndexType = typename MeshConfig::GlobalIndexType;
   using TagType         = typename MeshTraits< MeshConfig, Device >::EntityTagsArrayType::ValueType;

   Layer() = default;
   explicit Layer( const Layer& other ) {}
   template< typename Device_ >
   Layer( const Layer< MeshConfig, Device_, DimensionTag >& other ) {}
   Layer& operator=( const Layer& other ) { return *this; }
   template< typename Device_ >
   Layer& operator=( const Layer< MeshConfig, Device_, DimensionTag >& other ) { return *this; }

   void setEntitiesCount( DimensionTag, const GlobalIndexType& entitiesCount ) {}
   void getEntityTagsView( DimensionTag ) {}
   void getEntityTag( DimensionTag, const GlobalIndexType& ) const {}
   void addEntityTag( DimensionTag, const GlobalIndexType&, TagType ) const {}
   void removeEntityTag( DimensionTag, const GlobalIndexType&, TagType ) const {}
   void isBoundaryEntity( DimensionTag, const GlobalIndexType& ) const {}
   void isGhostEntity( DimensionTag, const GlobalIndexType& ) const {}
   void updateEntityTagsLayer( DimensionTag ) {}
   void getBoundaryIndices( DimensionTag ) const {}
   void getInteriorIndices( DimensionTag ) const {}
   void getGhostEntitiesCount() const;
   void getGhostEntitiesOffset() const;

   void save( File& file ) const {}
   void load( File& file ) {}

   void print( std::ostream& str ) const {}

   bool operator==( const Layer& layer ) const
   {
      return true;
   }
};

} // namespace EntityTags
} // namespace Meshes
} // namespace TNL
