/***************************************************************************
                          Mesh_impl.h  -  description
                             -------------------
    begin                : Sep 5, 2015
    copyright            : (C) 2015 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

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
getDimensions()
{
   return dimensions;
}

template< typename MeshConfig >
   template< int Dimensions >
bool
Mesh< MeshConfig >::
entitiesAvalable() const
{
   return MeshTraitsType::template EntityTraits< Dimensions >::available;
}

template< typename MeshConfig >
   template< int Dimensions >
typename Mesh< MeshConfig >::GlobalIndexType
Mesh< MeshConfig >::
getNumberOfEntities() const
{
   return entitiesStorage.getNumberOfEntities( MeshDimensionsTag< Dimensions >() );
}

template< typename MeshConfig >
typename Mesh< MeshConfig >::GlobalIndexType
Mesh< MeshConfig >::
template getNumberOfCells() const
{
   return entitiesStorage.getNumberOfEntities( MeshDimensionsTag< dimensions >() );
}

template< typename MeshConfig >
typename Mesh< MeshConfig >::CellType&
Mesh< MeshConfig >::
getCell( const GlobalIndexType cellIndex )
{
   return entitiesStorage.getEntity( MeshDimensionsTag< dimensions >(), cellIndex );
}

template< typename MeshConfig >
const typename Mesh< MeshConfig >::CellType&
Mesh< MeshConfig >::
getCell( const GlobalIndexType cellIndex ) const
{
   return entitiesStorage.getEntity( MeshDimensionsTag< dimensions >(), cellIndex );
}

template< typename MeshConfig >
   template< int Dimensions >
typename Mesh< MeshConfig >::template EntityType< Dimensions >&
Mesh< MeshConfig >::
getEntity( const GlobalIndexType entityIndex )
{
   return entitiesStorage.getEntity( MeshDimensionsTag< Dimensions >(), entityIndex );
}

template< typename MeshConfig >
   template< int Dimensions >
const typename Mesh< MeshConfig >::template EntityType< Dimensions >&
Mesh< MeshConfig >::
getEntity( const GlobalIndexType entityIndex ) const
{
   return entitiesStorage.getEntity( MeshDimensionsTag< Dimensions >(), entityIndex );
}
 
template< typename MeshConfig >
bool
Mesh< MeshConfig >::
save( File& file ) const
{
   if( ! Object::save( file ) ||
       ! entitiesStorage.save( file ) )
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
       ! entitiesStorage.load( file ) )
   {
      std::cerr << "Mesh loading failed." << std::endl;
      return false;
   }
   return true;
}

template< typename MeshConfig >
void
Mesh< MeshConfig >::
print( std::ostream& str ) const
{
   entitiesStorage.print( str );
}

template< typename MeshConfig >
bool
Mesh< MeshConfig >::
operator==( const Mesh& mesh ) const
{
   return entitiesStorage.operator==( mesh.entitiesStorage );
}

template< typename MeshConfig >
   template< typename DimensionsTag >
typename Mesh< MeshConfig >::template EntityTraits< DimensionsTag::value >::StorageArrayType&
Mesh< MeshConfig >::
entitiesArray()
{
   return entitiesStorage.entitiesArray( DimensionsTag() );
}

template< typename MeshConfig >
   template< typename DimensionsTag, typename SuperDimensionsTag >
typename Mesh< MeshConfig >::MeshTraitsType::GlobalIdArrayType&
Mesh< MeshConfig >::
superentityIdsArray()
{
   return entitiesStorage.template superentityIdsArray< SuperDimensionsTag >( DimensionsTag() );
}

template< typename MeshConfig >
bool
Mesh< MeshConfig >::
init( const typename Mesh< MeshConfig >::MeshTraitsType::PointArrayType& points,
      const typename Mesh< MeshConfig >::MeshTraitsType::CellSeedArrayType& cellSeeds )
{
   MeshInitializer< MeshConfig> meshInitializer;
   return meshInitializer.createMesh( points, cellSeeds, *this );
}


template< typename MeshConfig >
std::ostream& operator <<( std::ostream& str, const Mesh< MeshConfig >& mesh )
{
   mesh.print( str );
   return str;
}

} // namespace Meshes
} // namespace TNL

