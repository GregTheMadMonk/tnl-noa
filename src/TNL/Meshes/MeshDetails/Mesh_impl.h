/***************************************************************************
                          Mesh_impl.h  -  description
                             -------------------
    begin                : Sep 5, 2015
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

#include <TNL/Meshes/Mesh.h>
#include <TNL/Meshes/MeshDetails/layers/MeshEntityStorageRebinder.h>
#include <TNL/Meshes/MeshDetails/initializer/MeshInitializer.h>
#include <TNL/Meshes/MeshDetails/IndexPermutationApplier.h>

namespace TNL {
namespace Meshes {

template< typename MeshConfig, typename Device, typename MeshType >
bool
MeshInitializableBase< MeshConfig, Device, MeshType >::
init( typename MeshTraitsType::PointArrayType& points,
      typename MeshTraitsType::CellSeedArrayType& cellSeeds )
{
   MeshInitializer< typename MeshType::Config > meshInitializer;
   if( ! meshInitializer.createMesh( points, cellSeeds, *static_cast<MeshType*>(this) ) )
      return false;
   return true;
}


template< typename MeshConfig, typename Device >
Mesh< MeshConfig, Device >::
Mesh( const Mesh& mesh )
   : StorageBaseType( mesh )
{
   // update pointers from entities into the subentity and superentity storage networks
   MeshEntityStorageRebinder< Mesh< MeshConfig, Device > >::exec( *this );
}

template< typename MeshConfig, typename Device >
   template< typename Device_ >
Mesh< MeshConfig, Device >::
Mesh( const Mesh< MeshConfig, Device_ >& mesh )
   // clang complains that subclass cannot be cast to its private/protected base type,
   // but for some reason it works fine for the non-template copy-constructor
//   : StorageBaseType( *static_cast< const MeshStorageLayers< MeshConfig, Device_ >* >( &mesh ) )
   : StorageBaseType( *( (const MeshStorageLayers< MeshConfig, Device_ >*) &mesh ) )
{
   // update pointers from entities into the subentity and superentity storage networks
   MeshEntityStorageRebinder< Mesh< MeshConfig, Device > >::exec( *this );
}

template< typename MeshConfig, typename Device >
Mesh< MeshConfig, Device >&
Mesh< MeshConfig, Device >::
operator=( const Mesh& mesh )
{
   StorageBaseType::operator=( *( (const MeshStorageLayers< MeshConfig, Device >*) &mesh ) );
   // update pointers from entities into the subentity and superentity storage networks
   MeshEntityStorageRebinder< Mesh< MeshConfig, Device > >::exec( *this );
   return *this;
}

template< typename MeshConfig, typename Device >
   template< typename Device_ >
Mesh< MeshConfig, Device >&
Mesh< MeshConfig, Device >::
operator=( const Mesh< MeshConfig, Device_ >& mesh )
{
   StorageBaseType::operator=( *( (const MeshStorageLayers< MeshConfig, Device_ >*) &mesh ) );
   // update pointers from entities into the subentity and superentity storage networks
   MeshEntityStorageRebinder< Mesh< MeshConfig, Device > >::exec( *this );
   return *this;
}

template< typename MeshConfig, typename Device >
constexpr int
Mesh< MeshConfig, Device >::
getMeshDimension()
{
   return MeshTraitsType::meshDimension;
}

template< typename MeshConfig, typename Device >
String
Mesh< MeshConfig, Device >::
getType()
{
   return String( "Meshes::Mesh< ") + MeshConfig::getType() + " >";
}

template< typename MeshConfig, typename Device >
String
Mesh< MeshConfig, Device >::
getTypeVirtual() const
{
   return this->getType();
}

template< typename MeshConfig, typename Device >
String
Mesh< MeshConfig, Device >::
getSerializationType()
{
   return Mesh::getType();
}

template< typename MeshConfig, typename Device >
String
Mesh< MeshConfig, Device >::
getSerializationTypeVirtual() const
{
   return this->getSerializationType();
}

template< typename MeshConfig, typename Device >
   template< int Dimension >
constexpr bool
Mesh< MeshConfig, Device >::
entitiesAvailable()
{
   return MeshTraitsType::template EntityTraits< Dimension >::storageEnabled;
}

template< typename MeshConfig, typename Device >
   template< int Dimension >
__cuda_callable__
typename Mesh< MeshConfig, Device >::GlobalIndexType
Mesh< MeshConfig, Device >::
getEntitiesCount() const
{
   static_assert( EntityTraits< Dimension >::storageEnabled, "You try to get number of entities which are not configured for storage." );
   return StorageBaseType::getEntitiesCount( DimensionTag< Dimension >() );
}

template< typename MeshConfig, typename Device >
   template< int Dimension >
__cuda_callable__
typename Mesh< MeshConfig, Device >::template EntityType< Dimension >&
Mesh< MeshConfig, Device >::
getEntity( const GlobalIndexType& entityIndex )
{
   static_assert( EntityTraits< Dimension >::storageEnabled, "You try to get entity which is not configured for storage." );
   return StorageBaseType::getEntity( DimensionTag< Dimension >(), entityIndex );
}

template< typename MeshConfig, typename Device >
   template< int Dimension >
__cuda_callable__
const typename Mesh< MeshConfig, Device >::template EntityType< Dimension >&
Mesh< MeshConfig, Device >::
getEntity( const GlobalIndexType& entityIndex ) const
{
   static_assert( EntityTraits< Dimension >::storageEnabled, "You try to get entity which is not configured for storage." );
   return StorageBaseType::getEntity( DimensionTag< Dimension >(), entityIndex );
}


// duplicated for compatibility with grids
template< typename MeshConfig, typename Device >
   template< typename Entity >
__cuda_callable__
typename Mesh< MeshConfig, Device >::GlobalIndexType
Mesh< MeshConfig, Device >::
getEntitiesCount() const
{
   return getEntitiesCount< Entity::getEntityDimension() >();
}

template< typename MeshConfig, typename Device >
   template< typename Entity >
__cuda_callable__
Entity&
Mesh< MeshConfig, Device >::
getEntity( const GlobalIndexType& entityIndex )
{
   return getEntity< Entity::getEntityDimension() >( entityIndex );
}

template< typename MeshConfig, typename Device >
   template< typename Entity >
__cuda_callable__
const Entity&
Mesh< MeshConfig, Device >::
getEntity( const GlobalIndexType& entityIndex ) const
{
   return getEntity< Entity::getEntityDimension() >( entityIndex );
}


template< typename MeshConfig, typename Device >
   template< int Dimension >
bool
Mesh< MeshConfig, Device >::
reorderEntities( const IndexPermutationVector& perm,
                 const IndexPermutationVector& iperm )
{
   static_assert( entitiesAvailable< Dimension >(), "Entities which are not stored cannot be reordered." );

   const GlobalIndexType entitiesCount = getEntitiesCount< Dimension >();

   // basic sanity check
   if( perm.getSize() != entitiesCount || iperm.getSize() != entitiesCount ) {
      std::cerr << "Wrong size of permutation vectors: perm = " << perm << ", iperm = " << iperm << std::endl;
      return false;
   }
   TNL_ASSERT( perm.min() == 0 && perm.max() == entitiesCount - 1,
               std::cerr << "Given array is not a permutation: min = " << perm.min()
                         << ", max = " << perm.max()
                         << ", number of entities = " << entitiesCount
                         << ", array = " << perm << std::endl; );
   TNL_ASSERT( iperm.min() == 0 && iperm.max() == entitiesCount - 1,
               std::cerr << "Given array is not a permutation: min = " << iperm.min()
                         << ", max = " << iperm.max()
                         << ", number of entities = " << entitiesCount
                         << ", array = " << iperm << std::endl; );

   const bool status = IndexPermutationApplier< Mesh, Dimension >::exec( *this, perm, iperm );
   if( status ) {
      // update pointers from entities into the subentity and superentity storage networks
      // TODO: it would be enough to rebind just the permuted entities
      MeshEntityStorageRebinder< Mesh< MeshConfig, Device > >::exec( *this );

      // update boundary tags
      BoundaryTagsInitializer< Mesh >::exec( *this );
   }

   return status;
}


template< typename MeshConfig, typename Device >
bool
Mesh< MeshConfig, Device >::
save( File& file ) const
{
   if( ! Object::save( file ) ||
       ! StorageBaseType::save( file ) )
   {
      std::cerr << "Mesh saving failed." << std::endl;
      return false;
   }
   return true;
}

template< typename MeshConfig, typename Device >
bool
Mesh< MeshConfig, Device >::
load( File& file )
{
   if( ! Object::load( file ) ||
       ! StorageBaseType::load( file ) )
   {
      std::cerr << "Mesh loading failed." << std::endl;
      return false;
   }
   // TODO: this could be done from the storage layer
   // update pointers from entities into the subentity and superentity storage networks
   MeshEntityStorageRebinder< Mesh< MeshConfig, Device > >::exec( *this );
   return true;
}

template< typename MeshConfig, typename Device >
void
Mesh< MeshConfig, Device >::
print( std::ostream& str ) const
{
   // FIXME: there is a problem with multimaps and accessors holding pointers into the device memory
   if( std::is_same< Device, Devices::Cuda >::value ) {
      str << "Textual representation of meshes stored on the CUDA device is not supported." << std::endl;
   }
   else {
      StorageBaseType::print( str );
   }
}

template< typename MeshConfig, typename Device >
bool
Mesh< MeshConfig, Device >::
operator==( const Mesh& mesh ) const
{
   return StorageBaseType::operator==( mesh );
}

template< typename MeshConfig, typename Device >
bool
Mesh< MeshConfig, Device >::
operator!=( const Mesh& mesh ) const
{
   return ! operator==( mesh );
}

template< typename MeshConfig, typename Device >
void
Mesh< MeshConfig, Device >::
writeProlog( Logger& logger )
{
   logger.writeParameter( "Dimension:", getMeshDimension() );
   logger.writeParameter( "Number of cells:", getEntitiesCount< getMeshDimension() >() );
   if( getMeshDimension() > 1 )
      logger.writeParameter( "Number of faces:", getEntitiesCount< getMeshDimension() - 1 >() );
   logger.writeParameter( "Number of vertices:", getEntitiesCount< 0 >() );
   // TODO: more parameters?
}


template< typename MeshConfig, typename Device >
std::ostream& operator<<( std::ostream& str, const Mesh< MeshConfig, Device >& mesh )
{
   mesh.print( str );
   return str;
}

} // namespace Meshes
} // namespace TNL

