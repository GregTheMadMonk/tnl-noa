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
constexpr int
Mesh< MeshConfig >::
getMeshDimension()
{
   return dimension;
}

template< typename MeshConfig >
   template< int Dimensions >
constexpr bool
Mesh< MeshConfig >::
entitiesAvailable()
{
   return MeshTraitsType::template EntityTraits< Dimensions >::storageEnabled;
}

template< typename MeshConfig >
   template< int Dimension >
typename Mesh< MeshConfig >::GlobalIndexType
Mesh< MeshConfig >::
getNumberOfEntities() const
{
   return StorageBaseType::getNumberOfEntities( MeshDimensionsTag< Dimensions >() );
}

template< typename MeshConfig >
   template< int Dimensions >
typename Mesh< MeshConfig >::template EntityType< Dimensions >&
Mesh< MeshConfig >::
getEntity( const GlobalIndexType& entityIndex )
{
   return StorageBaseType::getEntity( MeshDimensionsTag< Dimensions >(), entityIndex );
}

template< typename MeshConfig >
   template< int Dimensions >
const typename Mesh< MeshConfig >::template EntityType< Dimensions >&
Mesh< MeshConfig >::
getEntity( const GlobalIndexType& entityIndex ) const
{
   return StorageBaseType::getEntity( MeshDimensionsTag< Dimensions >(), entityIndex );
}

template< typename MeshConfig >
typename Mesh< MeshConfig >::GlobalIndexType
Mesh< MeshConfig >::
getNumberOfCells() const
{
   return this->template getNumberOfEntities< dimensions >();
}

template< typename MeshConfig >
typename Mesh< MeshConfig >::CellType&
Mesh< MeshConfig >::
getCell( const GlobalIndexType& cellIndex )
{
   return this->template getEntity< dimensions >( cellIndex );
}

template< typename MeshConfig >
const typename Mesh< MeshConfig >::CellType&
Mesh< MeshConfig >::
getCell( const GlobalIndexType& cellIndex ) const
{
   return this->template getEntity< dimensions >( cellIndex );
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
   // update pointers from entities into the superentity storage network
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
init( const typename MeshTraitsType::PointArrayType& points,
      const typename MeshTraitsType::CellSeedArrayType& cellSeeds )
{
   MeshInitializer< MeshConfig> meshInitializer;
   if( ! meshInitializer.createMesh( points, cellSeeds, *this ) )
      return false;
   // update pointers from entities into the superentity storage network
   MeshEntityStorageRebinder< Mesh< MeshConfig > >::exec( *this );
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

