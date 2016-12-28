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

namespace TNL {
namespace Meshes {

template< typename MeshConfig >
String
Mesh< MeshConfig >::
getType()
{
   return String( "Meshes::Mesh< ") + MeshConfig::getType() + " >";
}

template< typename MeshConfig >
String
Mesh< MeshConfig >::
getTypeVirtual() const
{
   return this->getType();
}

template< typename MeshConfig >
String
Mesh< MeshConfig >::
getSerializationType()
{
   return Mesh::getType();
}

template< typename MeshConfig >
String
Mesh< MeshConfig >::
getSerializationTypeVirtual() const
{
   return this->getSerializationType();
}

template< typename MeshConfig >
constexpr int
Mesh< MeshConfig >::
getMeshDimension()
{
   return MeshTraitsType::meshDimension;
}

template< typename MeshConfig >
   template< int Dimension >
constexpr bool
Mesh< MeshConfig >::
entitiesAvailable()
{
   return MeshTraitsType::template EntityTraits< Dimension >::storageEnabled;
}

template< typename MeshConfig >
   template< int Dimension >
typename Mesh< MeshConfig >::GlobalIndexType
Mesh< MeshConfig >::
getEntitiesCount() const
{
   static_assert( EntityTraits< Dimension >::storageEnabled, "You try to get number of entities which are not configured for storage." );
   return StorageBaseType::getEntitiesCount( DimensionTag< Dimension >() );
}

template< typename MeshConfig >
   template< int Dimension >
typename Mesh< MeshConfig >::template EntityType< Dimension >&
Mesh< MeshConfig >::
getEntity( const GlobalIndexType& entityIndex )
{
   static_assert( EntityTraits< Dimension >::storageEnabled, "You try to get entity which is not configured for storage." );
   return StorageBaseType::getEntity( DimensionTag< Dimension >(), entityIndex );
}

template< typename MeshConfig >
   template< int Dimension >
const typename Mesh< MeshConfig >::template EntityType< Dimension >&
Mesh< MeshConfig >::
getEntity( const GlobalIndexType& entityIndex ) const
{
   static_assert( EntityTraits< Dimension >::storageEnabled, "You try to get entity which is not configured for storage." );
   return StorageBaseType::getEntity( DimensionTag< Dimension >(), entityIndex );
}

template< typename MeshConfig >
typename Mesh< MeshConfig >::GlobalIndexType
Mesh< MeshConfig >::
getCellsCount() const
{
   return this->template getEntitiesCount< getMeshDimension() >();
}

template< typename MeshConfig >
typename Mesh< MeshConfig >::CellType&
Mesh< MeshConfig >::
getCell( const GlobalIndexType& cellIndex )
{
   return this->template getEntity< getMeshDimension() >( cellIndex );
}

template< typename MeshConfig >
const typename Mesh< MeshConfig >::CellType&
Mesh< MeshConfig >::
getCell( const GlobalIndexType& cellIndex ) const
{
   return this->template getEntity< getMeshDimension() >( cellIndex );
}

template< typename MeshConfig >
bool
Mesh< MeshConfig >::
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

template< typename MeshConfig >
bool
Mesh< MeshConfig >::
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
   MeshEntityStorageRebinder< Mesh< MeshConfig > >::exec( *this );
   return true;
}

template< typename MeshConfig >
void
Mesh< MeshConfig >::
print( std::ostream& str ) const
{
   StorageBaseType::print( str );
}

template< typename MeshConfig >
bool
Mesh< MeshConfig >::
operator==( const Mesh& mesh ) const
{
   return StorageBaseType::operator==( mesh );
}

template< typename MeshConfig >
bool
Mesh< MeshConfig >::
init( typename MeshTraitsType::PointArrayType& points,
      typename MeshTraitsType::CellSeedArrayType& cellSeeds )
{
   MeshInitializer< MeshConfig> meshInitializer;
   if( ! meshInitializer.createMesh( points, cellSeeds, *this ) )
      return false;
   return true;
}


template< typename MeshConfig >
std::ostream& operator<<( std::ostream& str, const Mesh< MeshConfig >& mesh )
{
   mesh.print( str );
   return str;
}

} // namespace Meshes
} // namespace TNL

