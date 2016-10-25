/***************************************************************************
                          MeshEntity_impl.h  -  description
                             -------------------
    begin                : Sep 8, 2015
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

#include <TNL/Meshes/MeshEntity.h>

namespace TNL {
namespace Meshes {

template< typename MeshConfig,
          typename EntityTopology >
String
MeshEntity< MeshConfig, EntityTopology >::
getType()
{
   return String( "MeshEntity< ... >" );
}

template< typename MeshConfig,
          typename EntityTopology >
String
MeshEntity< MeshConfig, EntityTopology >::
getTypeVirtual() const
{
   return this->getType();
}

template< typename MeshConfig,
          typename EntityTopology >
bool
MeshEntity< MeshConfig, EntityTopology >::
save( File& file ) const
{
   if( ! MeshSubentityStorageLayers< MeshConfig, EntityTopology >::save( file ) /*||
       ! MeshSuperentityStorageLayers< MeshConfig, EntityTopology >::save( file )*/ )
      return false;
   return true;
}

template< typename MeshConfig,
          typename EntityTopology >
bool
MeshEntity< MeshConfig, EntityTopology >::
load( File& file )
{
   if( ! MeshSubentityStorageLayers< MeshConfig, EntityTopology >::load( file ) /*||
       ! MeshSuperentityStorageLayers< MeshConfig, EntityTopology >::load( file ) */ )
      return false;
   return true;
}

template< typename MeshConfig,
          typename EntityTopology >
void
MeshEntity< MeshConfig, EntityTopology >::
print( std::ostream& str ) const
{
   str << "\t Mesh entity dimension: " << EntityTopology::dimensions << std::endl;
   MeshSubentityStorageLayers< MeshConfig, EntityTopology >::print( str );
   MeshSuperentityAccess< MeshConfig, EntityTopology >::print( str );
}

template< typename MeshConfig,
          typename EntityTopology >
bool
MeshEntity< MeshConfig, EntityTopology >::
operator==( const MeshEntity& entity ) const
{
   return ( MeshSubentityStorageLayers< MeshConfig, EntityTopology >::operator==( entity ) &&
            MeshSuperentityAccess< MeshConfig, EntityTopology >::operator==( entity ) &&
            MeshEntityId< typename MeshConfig::IdType,
                          typename MeshConfig::GlobalIndexType >::operator==( entity ) );
}

template< typename MeshConfig,
          typename EntityTopology >
bool
MeshEntity< MeshConfig, EntityTopology >::
operator!=( const MeshEntity& entity ) const
{
   return ! ( *this == entity );
}

template< typename MeshConfig,
          typename EntityTopology >
constexpr int
MeshEntity< MeshConfig, EntityTopology >::
getEntityDimension() const
{
   return EntityTopology::dimensions;
}

/****
 * Subentities
 */
template< typename MeshConfig,
          typename EntityTopology >
   template< int Subdimensions >
constexpr typename MeshEntity< MeshConfig, EntityTopology >::LocalIndexType
MeshEntity< MeshConfig, EntityTopology >::
getNumberOfSubentities() const
{
   return SubentityTraits< Subdimensions >::count;
}

template< typename MeshConfig,
          typename EntityTopology >
   template< int Subdimensions >
typename MeshEntity< MeshConfig, EntityTopology >::GlobalIndexType
MeshEntity< MeshConfig, EntityTopology >::
getSubentityIndex( const LocalIndexType localIndex) const
{
   static_assert( SubentityTraits< Subdimensions >::storageEnabled, "You try to get subentity which is not configured for storage." );
   Assert( 0 <= localIndex && localIndex < SubentityTraits< Subdimensions >::count,
              std::cerr << "localIndex = " << localIndex
                        << " subentitiesCount = "
                        << SubentityTraits< Subdimensions >::count );
   typedef MeshSubentityStorageLayers< MeshConfig, EntityTopology >  SubentityBaseType;
   return SubentityBaseType::getSubentityIndex( MeshDimensionTag< Subdimensions >(),
                                                localIndex );
}

template< typename MeshConfig,
          typename EntityTopology >
   template< int Subdimensions >
typename MeshEntity< MeshConfig, EntityTopology >::template SubentityTraits< Subdimensions >::AccessArrayType&
MeshEntity< MeshConfig, EntityTopology >::
getSubentityIndices()
{
   static_assert( SubentityTraits< Subdimensions >::storageEnabled, "You try to get subentities which are not configured for storage." );
   typedef MeshSubentityStorageLayers< MeshConfig, EntityTopology >  SubentityBaseType;
   // FIXME: method does not exist
   return SubentityBaseType::getSubentityIndices( MeshDimensionsTag< Subdimensions >() );
}

template< typename MeshConfig,
          typename EntityTopology >
   template< int Subdimensions >
const typename MeshEntity< MeshConfig, EntityTopology >::template SubentityTraits< Subdimensions >::AccessArrayType&
MeshEntity< MeshConfig, EntityTopology >::
getSubentityIndices() const
{
   static_assert( SubentityTraits< Subdimensions >::storageEnabled, "You try to set subentities which are not configured for storage." );
   typedef MeshSubentityStorageLayers< MeshConfig, EntityTopology >  SubentityBaseType;
   // FIXME: method does not exist
   return SubentityBaseType::getSubentityIndices( MeshDimensionsTag< Subdimensions >() );
}

template< typename MeshConfig,
          typename EntityTopology >
constexpr typename MeshEntity< MeshConfig, EntityTopology >::LocalIndexType
MeshEntity< MeshConfig, EntityTopology >::
getNumberOfVertices() const
{
   return SubentityTraits< 0 >::count;
}

template< typename MeshConfig,
          typename EntityTopology >
typename MeshEntity< MeshConfig, EntityTopology >::GlobalIndexType
MeshEntity< MeshConfig, EntityTopology >::
getVertexIndex( const LocalIndexType localIndex ) const
{
   return this->getSubentityIndex< 0 >( localIndex  );
}

template< typename MeshConfig,
          typename EntityTopology >
typename MeshEntity< MeshConfig, EntityTopology >::template SubentityTraits< 0 >::AccessArrayType&
MeshEntity< MeshConfig, EntityTopology >::
getVerticesIndices()
{
   return this->getSubentityIndices< 0 >();
}

template< typename MeshConfig,
          typename EntityTopology >
const typename MeshEntity< MeshConfig, EntityTopology >::template SubentityTraits< 0 >::AccessArrayType&
MeshEntity< MeshConfig, EntityTopology >::
getVerticesIndices() const
{
   return this->getSubentityIndices< 0 >();
}

template< typename MeshConfig,
          typename EntityTopology >
   template< int Dimension >
typename MeshEntity< MeshConfig, EntityTopology >::IdPermutationArrayAccessorType
MeshEntity< MeshConfig, EntityTopology >::
subentityOrientation( LocalIndexType index ) const
{
   static const LocalIndexType subentitiesCount = SubentityTraits< Dimension >::count;
   TNL_ASSERT( 0 <= index && index < subentitiesCount, );

   typedef MeshSubentityStorageLayers< MeshConfig, EntityTopology >       SubentityStorageLayers;
   return SubentityStorageLayers::subentityOrientation( MeshDimensionsTag< Dimensions >(), index );
}

/****
 * Mesh initialization method
 */

template< typename MeshConfig,
          typename EntityTopology >
   template< int Subdimensions >
void
MeshEntity< MeshConfig, EntityTopology >::
setSubentityIndex( const LocalIndexType& localIndex,
                   const GlobalIndexType& globalIndex )
{
   static_assert( SubentityTraits< Subdimensions >::storageEnabled, "You try to set subentity which is not configured for storage." );
   Assert( 0 <= localIndex && localIndex < SubentityTraits< Subdimensions >::count,
              std::cerr << "localIndex = " << localIndex
                        << " subentitiesCount = "
                        << SubentityTraits< Subdimensions >::count );
   typedef MeshSubentityStorageLayers< MeshConfig, EntityTopology >  SubentityBaseType;
   SubentityBaseType::setSubentityIndex( MeshDimensionTag< Subdimensions >(),
                                         localIndex,
                                         globalIndex );
}

template< typename MeshConfig,
          typename EntityTopology >
   template< int Subdimensions >
typename MeshEntity< MeshConfig, EntityTopology >::template SubentityTraits< Subdimensions >::IdArrayType&
MeshEntity< MeshConfig, EntityTopology >::
subentityIdsArray()
{
   typedef MeshSubentityStorageLayers< MeshConfig, EntityTopology >       SubentityStorageLayers;
   return SubentityStorageLayers::subentityIdsArray( MeshDimensionsTag< Subdimensions >() );
}

template< typename MeshConfig,
          typename EntityTopology >
   template< int Subdimensions >
typename MeshEntity< MeshConfig, EntityTopology >::template SubentityTraits< Subdimensions >::OrientationArrayType&
MeshEntity< MeshConfig, EntityTopology >::
subentityOrientationsArray()
{
   typedef MeshSubentityStorageLayers< MeshConfig, EntityTopology >       SubentityStorageLayers;
   return SubentityStorageLayers::subentityOrientationsArray( MeshDimensionsTag< Subdimensions >() );
}

/****
 * Vertex entity specialization
 */
template< typename MeshConfig >
String
MeshEntity< MeshConfig, MeshVertexTopology >::
getType()
{
   return String( "Mesh< ... >" );
}

template< typename MeshConfig >
String
MeshEntity< MeshConfig, MeshVertexTopology >::
getTypeVirtual() const
{
   return this->getType();
}

template< typename MeshConfig >
bool
MeshEntity< MeshConfig, MeshVertexTopology >::
save( File& file ) const
{
   if( //! MeshSuperentityStorageLayers< MeshConfig, MeshVertexTopology >::save( file ) ||
       ! point.save( file ) )
      return false;
   return true;
}

template< typename MeshConfig >
bool
MeshEntity< MeshConfig, MeshVertexTopology >::
load( File& file )
{
   if( //! MeshSuperentityStorageLayers< MeshConfig, MeshVertexTopology >::load( file ) ||
       ! point.load( file ) )
      return false;
   return true;
}

template< typename MeshConfig >
void
MeshEntity< MeshConfig, MeshVertexTopology >::
print( std::ostream& str ) const
{
   str << "\t Mesh entity dimensions: " << MeshVertexTopology::dimensions << std::endl;
   str << "\t Coordinates = " << point << std::endl;
   MeshSuperentityAccess< MeshConfig, MeshVertexTopology >::print( str );
}

template< typename MeshConfig >
bool
MeshEntity< MeshConfig, MeshVertexTopology >::
operator==( const MeshEntity& entity ) const
{
   return ( MeshSuperentityAccess< MeshConfig, MeshVertexTopology >::operator==( entity ) &&
            MeshEntityId< typename MeshConfig::IdType,
                          typename MeshConfig::GlobalIndexType >::operator==( entity ) &&
            point == entity.point );
}

template< typename MeshConfig >
bool
MeshEntity< MeshConfig, MeshVertexTopology >::
operator!=( const MeshEntity& entity ) const
{
   return ! ( *this == entity );
}

template< typename MeshConfig >
constexpr int
MeshEntity< MeshConfig, MeshVertexTopology >::
getEntityDimension() const
{
   return EntityTopology::dimensions;
}

template< typename MeshConfig >
typename MeshEntity< MeshConfig, MeshVertexTopology >::PointType
MeshEntity< MeshConfig, MeshVertexTopology >::
getPoint() const
{
   return this->point;
}

template< typename MeshConfig >
void
MeshEntity< MeshConfig, MeshVertexTopology >::
setPoint( const PointType& point )
{
   this->point = point;
}

template< typename MeshConfig,
          typename EntityTopology >
std::ostream& operator<<( std::ostream& str, const MeshEntity< MeshConfig, EntityTopology >& entity )
{
   entity.print( str );
   return str;
}

} // namespace Meshes
} // namespace TNL

