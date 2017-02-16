/***************************************************************************
                          MeshBoundaryTagsLayer.h  -  description
                             -------------------
    begin                : Dec 25, 2016
    copyright            : (C) 2016 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/File.h>
#include <TNL/Meshes/MeshDetails/traits/MeshTraits.h>

namespace TNL {
namespace Meshes {

// This is the implementation of the BoundaryTags layer for one specific dimension.
// It is inherited by MeshStorageLayer.
template< typename MeshConfig,
          typename Device,
          typename DimensionTag,
          bool TagStorage = MeshConfig::boundaryTagsStorage( typename MeshTraits< MeshConfig, Device >::template EntityTraits< DimensionTag::value >::EntityTopology() ) >
class MeshBoundaryTagsLayer
{
   using MeshTraitsType    = MeshTraits< MeshConfig, Device >;
   using EntityTraitsType  = typename MeshTraitsType::template EntityTraits< DimensionTag::value >;

public:
   using GlobalIndexType   = typename EntityTraitsType::GlobalIndexType;
   using BoundaryTagsArray = typename MeshTraitsType::BoundaryTagsArrayType;
   using OrderingArray     = typename MeshTraitsType::GlobalIndexOrderingArrayType;

   MeshBoundaryTagsLayer() = default;

   explicit MeshBoundaryTagsLayer( const MeshBoundaryTagsLayer& other )
   {
      operator=( other );
   }

   template< typename Device_ >
   MeshBoundaryTagsLayer( const MeshBoundaryTagsLayer< MeshConfig, Device_, DimensionTag >& other )
   {
      operator=( other );
   }

   MeshBoundaryTagsLayer& operator=( const MeshBoundaryTagsLayer& other )
   {
      // TODO: throw exception if allocation fails
      boundaryTags.setLike( other.boundaryTags );
      boundaryIndices.setLike( other.boundaryIndices );
      interiorIndices.setLike( other.interiorIndices );
      boundaryTags = other.boundaryTags;
      boundaryIndices = other.boundaryIndices;
      interiorIndices = other.interiorIndices;
      return *this;
   }

   template< typename Device_ >
   MeshBoundaryTagsLayer& operator=( const MeshBoundaryTagsLayer< MeshConfig, Device_, DimensionTag >& other )
   {
      // TODO: throw exception if allocation fails
      boundaryTags.setLike( other.boundaryTags );
      boundaryIndices.setLike( other.boundaryIndices );
      interiorIndices.setLike( other.interiorIndices );
      boundaryTags = other.boundaryTags;
      boundaryIndices = other.boundaryIndices;
      interiorIndices = other.interiorIndices;
      return *this;
   }


   bool setNumberOfEntities( const GlobalIndexType& entitiesCount )
   {
      return boundaryTags.setSize( entitiesCount );
   }

   void resetBoundaryTags( DimensionTag )
   {
      boundaryTags.setValue( false );
   }

   void setIsBoundaryEntity( DimensionTag, const GlobalIndexType& entityIndex, bool isBoundary )
   {
      boundaryTags[ entityIndex ] = isBoundary;
   }

   bool isBoundaryEntity( DimensionTag, const GlobalIndexType& entityIndex ) const
   {
      return boundaryTags[ entityIndex ];
   }

   bool updateBoundaryIndices( DimensionTag )
   {
      if( std::is_same< Device, Devices::Host >::value ) {
         // we can't just sum an array/vector of bools, because the result would also be bool
         // TODO: perhaps Containers::Vector::sum should provide a template parameter to force the result type
         // (by default it would be the RealType of the vector)
         GlobalIndexType boundaryEntities = 0;
         for( GlobalIndexType i = 0; i < boundaryTags.getSize(); i++ )
            if( boundaryTags[ i ] )
               boundaryEntities++;

         if( ! boundaryIndices.setSize( boundaryEntities ) ||
             ! interiorIndices.setSize( boundaryTags.getSize() - boundaryEntities ) )
            return false;

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

         if( ! hostBoundaryTags.setLike( boundaryTags ) )
            return false;
         hostBoundaryTags = boundaryTags;

         // we can't just sum an array/vector of bools, because the result would also be bool
         // TODO: perhaps Containers::Vector::sum should provide a template parameter to force the result type
         // (by default it would be the RealType of the vector)
         GlobalIndexType boundaryEntities = 0;
         for( GlobalIndexType i = 0; i < boundaryTags.getSize(); i++ )
            if( hostBoundaryTags[ i ] )
               boundaryEntities++;

         if( ! hostBoundaryIndices.setSize( boundaryEntities ) ||
             ! hostInteriorIndices.setSize( boundaryTags.getSize() - boundaryEntities ) )
            return false;

         GlobalIndexType b = 0;
         GlobalIndexType i = 0;
         while( b + i < boundaryTags.getSize() ) {
            const GlobalIndexType e = b + i;
            if( hostBoundaryTags[ e ] )
               hostBoundaryIndices[ b++ ] = e;
            else
               hostInteriorIndices[ i++ ] = e;
         }

         if( ! boundaryIndices.setLike( hostBoundaryIndices ) ||
             ! interiorIndices.setLike( hostInteriorIndices ) )
            return false;
         boundaryIndices = hostBoundaryIndices;
         interiorIndices = hostInteriorIndices;
      }

      return true;
   }

   GlobalIndexType getBoundaryEntitiesCount( DimensionTag ) const
   {
      return boundaryIndices.getSize();
   }

   GlobalIndexType getBoundaryEntityIndex( DimensionTag, const GlobalIndexType& i ) const
   {
      return boundaryIndices[ i ];
   }

   GlobalIndexType getInteriorEntitiesCount( DimensionTag ) const
   {
      return interiorIndices.getSize();
   }

   GlobalIndexType getInteriorEntityIndex( DimensionTag, const GlobalIndexType& i ) const
   {
      return interiorIndices[ i ];
   }

   bool save( File& file ) const
   {
      if( ! boundaryTags.save( file ) )
      {
         std::cerr << "Failed to save the boundary tags of the entities with dimension " << DimensionTag::value << "." << std::endl;
         return false;
      }
      return true;
   }

   bool load( File& file )
   {
      if( ! boundaryTags.load( file ) )
      {
         std::cerr << "Failed to load the boundary tags of the entities with dimension " << DimensionTag::value << "." << std::endl;
         return false;
      }
      if( ! updateBoundaryIndices( DimensionTag() ) )
      {
         std::cerr << "Failed to update the boundary indices of the entities with dimension " << DimensionTag::value << "." << std::endl;
         return false;
      }
      return true;
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

   bool operator==( const MeshBoundaryTagsLayer& layer ) const
   {
      TNL_ASSERT( ( boundaryTags == layer.boundaryTags && boundaryIndices == layer.boundaryIndices && interiorIndices == layer.interiorIndices ) ||
                  ( boundaryTags != layer.boundaryTags && boundaryIndices != layer.boundaryIndices && interiorIndices != layer.interiorIndices ),
                  std::cerr << "The BoundaryTags layer is in inconsistent state - this is probably a bug in the mesh initializer." << std::endl
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
   friend class MeshBoundaryTagsLayer;
};

template< typename MeshConfig,
          typename Device,
          typename DimensionTag >
class MeshBoundaryTagsLayer< MeshConfig, Device, DimensionTag, false >
{
   using MeshTraitsType    = MeshTraits< MeshConfig, Device >;
   using EntityTraitsType  = typename MeshTraitsType::template EntityTraits< DimensionTag::value >;

public:
   using GlobalIndexType   = typename EntityTraitsType::GlobalIndexType;

   MeshBoundaryTagsLayer() = default;
   explicit MeshBoundaryTagsLayer( const MeshBoundaryTagsLayer& other ) {}
   template< typename Device_ >
   MeshBoundaryTagsLayer( const MeshBoundaryTagsLayer< MeshConfig, Device_, DimensionTag >& other ) {}
   MeshBoundaryTagsLayer& operator=( const MeshBoundaryTagsLayer& other ) { return *this; }
   template< typename Device_ >
   MeshBoundaryTagsLayer& operator=( const MeshBoundaryTagsLayer< MeshConfig, Device_, DimensionTag >& other ) { return *this; }

   bool setNumberOfEntities( const GlobalIndexType& entitiesCount )
   {
      return true;
   }

   void resetBoundaryTags( DimensionTag ) {}
   void setIsBoundaryEntity( DimensionTag, const GlobalIndexType& entityIndex, bool isBoundary ) {}
   void isBoundaryEntity( DimensionTag, const GlobalIndexType& entityIndex ) const {}
   bool updateBoundaryIndices( DimensionTag ) { return true; }
   void getBoundaryEntitiesCount( DimensionTag ) const {}
   void getBoundaryEntityIndex( DimensionTag, const GlobalIndexType& i ) const {}
   void getInteriorEntitiesCount( DimensionTag ) const {}
   void getInteriorEntityIndex( DimensionTag, const GlobalIndexType& i ) const {}

   bool save( File& file ) const
   {
      return true;
   }

   bool load( File& file )
   {
      return true;
   }
 
   void print( std::ostream& str ) const {}

   bool operator==( const MeshBoundaryTagsLayer& layer ) const
   {
      return true;
   }
};

} // namespace Meshes
} // namespace TNL
