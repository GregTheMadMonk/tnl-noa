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
namespace BoundaryTags {

// This is the implementation of the boundary tags layer for one specific dimension.
// It is inherited by the BoundaryTags::LayerFamily.
template< typename MeshConfig,
          typename Device,
          typename DimensionTag,
          bool TagStorage = WeakStorageTrait< MeshConfig, Device, DimensionTag >::boundaryTagsEnabled >
class Layer
{
   using MeshTraitsType    = MeshTraits< MeshConfig, Device >;

public:
   using GlobalIndexType   = typename MeshTraitsType::GlobalIndexType;
   using BoundaryTagsArray = Containers::Array< bool, Device, GlobalIndexType >;
   using OrderingArray     = Containers::Array< GlobalIndexType, Device, GlobalIndexType >;

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
      boundaryTags.setLike( other.boundaryTags );
      boundaryIndices.setLike( other.boundaryIndices );
      interiorIndices.setLike( other.interiorIndices );
      boundaryTags = other.boundaryTags;
      boundaryIndices = other.boundaryIndices;
      interiorIndices = other.interiorIndices;
      return *this;
   }

   template< typename Device_ >
   Layer& operator=( const Layer< MeshConfig, Device_, DimensionTag >& other )
   {
      boundaryTags.setLike( other.boundaryTags );
      boundaryIndices.setLike( other.boundaryIndices );
      interiorIndices.setLike( other.interiorIndices );
      boundaryTags = other.boundaryTags;
      boundaryIndices = other.boundaryIndices;
      interiorIndices = other.interiorIndices;
      return *this;
   }


   void setEntitiesCount( DimensionTag, const GlobalIndexType& entitiesCount )
   {
      boundaryTags.setSize( entitiesCount );
   }

   void resetBoundaryTags( DimensionTag )
   {
      boundaryTags.setValue( false );
   }

   __cuda_callable__
   void setIsBoundaryEntity( DimensionTag, const GlobalIndexType& entityIndex, bool isBoundary )
   {
      boundaryTags[ entityIndex ] = isBoundary;
   }

   __cuda_callable__
   bool isBoundaryEntity( DimensionTag, const GlobalIndexType& entityIndex ) const
   {
      return boundaryTags[ entityIndex ];
   }

   void updateBoundaryIndices( DimensionTag )
   {
      // Array does not have sum(), Vector of bools does not fit due to arithmetics
      Containers::VectorView< typename BoundaryTagsArray::ValueType, typename BoundaryTagsArray::DeviceType, typename BoundaryTagsArray::IndexType > _boundaryTagsVector;
      _boundaryTagsVector.bind( boundaryTags.getData(), boundaryTags.getSize() );
      const GlobalIndexType boundaryEntities = _boundaryTagsVector.template sum< GlobalIndexType >();
      boundaryIndices.setSize( boundaryEntities );
      interiorIndices.setSize( boundaryTags.getSize() - boundaryEntities );

      if( std::is_same< Device, Devices::Host >::value ) {
         GlobalIndexType b = 0;
         GlobalIndexType i = 0;
         while( b + i < boundaryTags.getSize() ) {
            const GlobalIndexType e = b + i;
            if( boundaryTags[ e ] )
               boundaryIndices[ b++ ] = e;
            else
               interiorIndices[ i++ ] = e;
         }
      }
      // TODO: parallelize directly on the device
      else {
         using BoundaryTagsHostArray = typename BoundaryTagsArray::HostType;
         using OrderingHostArray     = typename OrderingArray::HostType;

         BoundaryTagsHostArray hostBoundaryTags;
         OrderingHostArray hostBoundaryIndices;
         OrderingHostArray hostInteriorIndices;

         hostBoundaryTags.setLike( boundaryTags );
         hostBoundaryIndices.setLike( boundaryIndices );
         hostInteriorIndices.setLike( interiorIndices );

         hostBoundaryTags = boundaryTags;

         GlobalIndexType b = 0;
         GlobalIndexType i = 0;
         while( b + i < boundaryTags.getSize() ) {
            const GlobalIndexType e = b + i;
            if( hostBoundaryTags[ e ] )
               hostBoundaryIndices[ b++ ] = e;
            else
               hostInteriorIndices[ i++ ] = e;
         }

         boundaryIndices = hostBoundaryIndices;
         interiorIndices = hostInteriorIndices;
      }
   }

   __cuda_callable__
   GlobalIndexType getBoundaryEntitiesCount( DimensionTag ) const
   {
      return boundaryIndices.getSize();
   }

   __cuda_callable__
   GlobalIndexType getBoundaryEntityIndex( DimensionTag, const GlobalIndexType& i ) const
   {
      return boundaryIndices[ i ];
   }

   __cuda_callable__
   GlobalIndexType getInteriorEntitiesCount( DimensionTag ) const
   {
      return interiorIndices.getSize();
   }

   __cuda_callable__
   GlobalIndexType getInteriorEntityIndex( DimensionTag, const GlobalIndexType& i ) const
   {
      return interiorIndices[ i ];
   }

   void save( File& file ) const
   {
      boundaryTags.save( file );
   }

   void load( File& file )
   {
      boundaryTags.load( file );
      updateBoundaryIndices( DimensionTag() );
   }

   void print( std::ostream& str ) const
   {
      str << "Boundary tags for entities of dimension " << DimensionTag::value << " are: ";
      str << boundaryTags << std::endl;
      str << "Indices of the boundary entities of dimension " << DimensionTag::value << " are: ";
      str << boundaryIndices << std::endl;
      str << "Indices of the interior entities of dimension " << DimensionTag::value << " are: ";
      str << interiorIndices << std::endl;
   }

   bool operator==( const Layer& layer ) const
   {
      TNL_ASSERT( ( boundaryTags == layer.boundaryTags && boundaryIndices == layer.boundaryIndices && interiorIndices == layer.interiorIndices ) ||
                  ( boundaryTags != layer.boundaryTags && boundaryIndices != layer.boundaryIndices && interiorIndices != layer.interiorIndices ),
                  std::cerr << "The BoundaryTags layer is in inconsistent state - this is probably a bug in the boundary tags initializer." << std::endl
                            << "boundaryTags          = " << boundaryTags << std::endl
                            << "layer.boundaryTags    = " << layer.boundaryTags << std::endl
                            << "boundaryIndices       = " << boundaryIndices << std::endl
                            << "layer.boundaryIndices = " << layer.boundaryIndices << std::endl
                            << "interiorIndices       = " << interiorIndices << std::endl
                            << "layer.interiorIndices = " << layer.interiorIndices << std::endl; );
      return boundaryTags == layer.boundaryTags;
   }

private:
   BoundaryTagsArray boundaryTags;
   OrderingArray boundaryIndices;
   OrderingArray interiorIndices;

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

   Layer() = default;
   explicit Layer( const Layer& other ) {}
   template< typename Device_ >
   Layer( const Layer< MeshConfig, Device_, DimensionTag >& other ) {}
   Layer& operator=( const Layer& other ) { return *this; }
   template< typename Device_ >
   Layer& operator=( const Layer< MeshConfig, Device_, DimensionTag >& other ) { return *this; }

   void setEntitiesCount( DimensionTag, const GlobalIndexType& entitiesCount ) {}
   void resetBoundaryTags( DimensionTag ) {}
   void setIsBoundaryEntity( DimensionTag, const GlobalIndexType& entityIndex, bool isBoundary ) {}
   void isBoundaryEntity( DimensionTag, const GlobalIndexType& entityIndex ) const {}
   void updateBoundaryIndices( DimensionTag ) {}
   void getBoundaryEntitiesCount( DimensionTag ) const {}
   void getBoundaryEntityIndex( DimensionTag, const GlobalIndexType& i ) const {}
   void getInteriorEntitiesCount( DimensionTag ) const {}
   void getInteriorEntityIndex( DimensionTag, const GlobalIndexType& i ) const {}

   void save( File& file ) const {}
   void load( File& file ) {}

   void print( std::ostream& str ) const {}

   bool operator==( const Layer& layer ) const
   {
      return true;
   }
};

} // namespace BoundaryTags
} // namespace Meshes
} // namespace TNL
