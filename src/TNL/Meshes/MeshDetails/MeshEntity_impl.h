/***************************************************************************
                          MeshEntity_impl.h  -  description
                             -------------------
    begin                : Sep 8, 2015
    copyright            : (C) 2015 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Meshes/MeshEntity.h>

namespace TNL {
namespace Meshes {

template< typename MeshConfig,
          typename EntityTopology >
MeshEntity< MeshConfig, EntityTopology >::
MeshEntity( const SeedType& entitySeed )
{
   typedef typename SeedType::LocalIndexType LocalIndexType;
   for( LocalIndexType i = 0; i < entitySeed.getCornerIds().getSize(); i++ )
      this->template setSubentityIndex< 0 >( i, entitySeed.getCornerIds()[ i ] );
}


template< typename MeshConfig,
          typename EntityTopology >
MeshEntity< MeshConfig, EntityTopology >::
MeshEntity()
{
}

template< typename MeshConfig,
          typename EntityTopology >
MeshEntity< MeshConfig, EntityTopology >::
~MeshEntity()
{
   //cerr << "   Destroying entity with " << EntityTopology::dimensions << " dimensions..." << std::endl;
}

template< typename MeshConfig,
          typename EntityTopology >
String
MeshEntity< MeshConfig, EntityTopology >::
getType()
{
   return String( "Mesh< ... >" );
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
   str << "\t Mesh entity dimensions: " << EntityTopology::dimensions << std::endl;
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
constexpr int
MeshEntity< MeshConfig, EntityTopology >::
getEntityDimensions() const
{
   return EntityTopology::dimensions;
}

/****
 * Subentities
 */
template< typename MeshConfig,
          typename EntityTopology >
   template< int Subdimensions >
constexpr bool
MeshEntity< MeshConfig, EntityTopology >::
subentitiesAvailable() const
{
   return SubentityTraits< Subdimensions >::storageEnabled;
};

template< typename MeshConfig,
          typename EntityTopology >
   template< int Subdimensions >
constexpr typename MeshEntity< MeshConfig, EntityTopology >::LocalIndexType
MeshEntity< MeshConfig, EntityTopology >::
getNumberOfSubentities() const
{
   return SubentityTraits< Subdimensions >::count;
};

template< typename MeshConfig,
          typename EntityTopology >
   template< int Subdimensions >
typename MeshEntity< MeshConfig, EntityTopology >::GlobalIndexType
MeshEntity< MeshConfig, EntityTopology >::
getSubentityIndex( const LocalIndexType localIndex) const
{
   static_assert( SubentityTraits< Subdimensions >::storageEnabled, "You try to get subentity which is not configured for storage." );
   Assert( 0 <= localIndex &&
              localIndex < SubentityTraits< Subdimensions >::count,
              std::cerr << "localIndex = " << localIndex
                   << " subentitiesCount = "
                   << SubentityTraits< Subdimensions >::count );
   typedef MeshSubentityStorageLayers< MeshConfig, EntityTopology >  SubentityBaseType;
   return SubentityBaseType::getSubentityIndex( MeshDimensionsTag< Subdimensions >(),
                                                localIndex );
}

template< typename MeshConfig,
          typename EntityTopology >
   template< int Subdimensions >
typename MeshEntity< MeshConfig, EntityTopology >::template SubentityTraits< Subdimensions >::AccessArrayType&
MeshEntity< MeshConfig, EntityTopology >::
 getSubentitiesIndices()
{
   static_assert( SubentityTraits< Subdimensions >::storageEnabled, "You try to get subentities which are not configured for storage." );
   typedef MeshSubentityStorageLayers< MeshConfig, EntityTopology >  SubentityBaseType;
   return SubentityBaseType::getSubentitiesIndices( MeshDimensionsTag< Subdimensions >() );
}

template< typename MeshConfig,
          typename EntityTopology >
   template< int Subdimensions >
const typename MeshEntity< MeshConfig, EntityTopology >::template SubentityTraits< Subdimensions >::AccessArrayType&
MeshEntity< MeshConfig, EntityTopology >::
getSubentitiesIndices() const
{
   static_assert( SubentityTraits< Subdimensions >::storageEnabled, "You try to set subentities which are not configured for storage." );
   typedef MeshSubentityStorageLayers< MeshConfig, EntityTopology >  SubentityBaseType;
   return SubentityBaseType::getSubentitiesIndices( MeshDimensionsTag< Subdimensions >() );
}

template< typename MeshConfig,
          typename EntityTopology >
   template< int SuperDimensions >
typename MeshEntity< MeshConfig, EntityTopology >::LocalIndexType
MeshEntity< MeshConfig, EntityTopology >::
getNumberOfSuperentities() const
{
   static_assert( SuperentityTraits< SuperDimensions >::available, "You try to get number of superentities which are not configured for storage." );
   typedef MeshSuperentityAccess< MeshConfig, EntityTopology >  SuperentityBaseType;
   return SuperentityBaseType::getNumberOfSuperentities( MeshDimensionsTag< SuperDimensions >() );
}

template< typename MeshConfig,
          typename EntityTopology >
   template< int SuperDimensions >
typename MeshEntity< MeshConfig, EntityTopology >::GlobalIndexType
MeshEntity< MeshConfig, EntityTopology >::
getSuperentityIndex( const LocalIndexType localIndex ) const
{
   static_assert( SuperentityTraits< SuperDimensions >::storageEnabled, "You try to get superentity which is not configured for storage." );
   Assert( localIndex < this->getNumberOfSuperentities< SuperDimensions >(),
              std::cerr << " localIndex = " << localIndex
                   << " this->getNumberOfSuperentities< Dimensions >() = " << this->getNumberOfSuperentities< SuperDimensions >() << std::endl; );
   typedef MeshSuperentityAccess< MeshConfig, EntityTopology >  SuperentityBaseType;
   return SuperentityBaseType::getSuperentityIndex( MeshDimensionsTag< SuperDimensions >(),
                                                    localIndex );
}

template< typename MeshConfig,
          typename EntityTopology >
   template< int SuperDimensions >
typename MeshEntity< MeshConfig, EntityTopology >::template SuperentityTraits< SuperDimensions >::AccessArrayType&
MeshEntity< MeshConfig, EntityTopology >::
getSuperentitiesIndices()
{
   static_assert( SuperentityTraits< SuperDimensions >::storageEnabled, "You try to get superentities which are not configured for storage." );
   typedef MeshSuperentityAccess< MeshConfig, EntityTopology >  SuperentityBaseType;
   //return SuperentityBaseType::getSuperentitiesIndices( MeshDimensionsTag< Dimensions >() );
}

template< typename MeshConfig,
          typename EntityTopology >
   template< int SuperDimensions >
const typename MeshEntity< MeshConfig, EntityTopology >::template SuperentityTraits< SuperDimensions >::AccessArrayType&
MeshEntity< MeshConfig, EntityTopology >::
getSuperentitiesIndices() const
{
   static_assert( SuperentityTraits< SuperDimensions >::storageEnabled, "You try to get superentities which are not configured for storage." );
   typedef MeshSuperentityAccess< MeshConfig, EntityTopology >  SuperentityBaseType;
   return SuperentityBaseType::getSubentitiesIndices( MeshDimensionsTag< SuperDimensions >() );
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
   return this->getSubentitiesIndices< 0 >();
}

template< typename MeshConfig,
          typename EntityTopology >
const typename MeshEntity< MeshConfig, EntityTopology >::template SubentityTraits< 0 >::AccessArrayType&
MeshEntity< MeshConfig, EntityTopology >::
getVerticesIndices() const
{
   return this->getSubentitiesIndices< 0 >();
}

template< typename MeshConfig,
          typename EntityTopology >
   template< int Dimensions >
typename MeshEntity< MeshConfig, EntityTopology >::IdPermutationArrayAccessorType
MeshEntity< MeshConfig, EntityTopology >::
subentityOrientation( LocalIndexType index ) const
{
   static const LocalIndexType subentitiesCount = SubentityTraits< Dimensions >::count;
   Assert( 0 <= index && index < subentitiesCount, );

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
setSubentityIndex( const LocalIndexType localIndex,
                   const GlobalIndexType globalIndex )
{
   static_assert( SubentityTraits< Subdimensions >::storageEnabled, "You try to set subentity which is not configured for storage." );
   Assert( 0 <= localIndex &&
              localIndex < SubentityTraits< Subdimensions >::count,
              std::cerr << "localIndex = " << localIndex
                   << " subentitiesCount = "
                   << SubentityTraits< Subdimensions >::count );
   typedef MeshSubentityStorageLayers< MeshConfig, EntityTopology >  SubentityBaseType;
   SubentityBaseType::setSubentityIndex( MeshDimensionsTag< Subdimensions >(),
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
   return SubentityStorageLayers::subentityIdsArray( MeshDimensionsTag< Subdimensions >() );
}

template< typename MeshConfig,
          typename EntityTopology >
   template< int Superdimensions >
typename MeshEntity< MeshConfig, EntityTopology >::IdArrayAccessorType&
MeshEntity< MeshConfig, EntityTopology >::
superentityIdsArray()
{
   return SuperentityAccessBase::superentityIdsArray( MeshDimensionsTag< Superdimensions >());
}

template< typename MeshConfig,
          typename EntityTopology >
   template< int Subdimensions >
typename MeshEntity< MeshConfig, EntityTopology >::template SubentityTraits< Subdimensions >::OrientationArrayType&
MeshEntity< MeshConfig, EntityTopology >::
subentityOrientationsArray()
{
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
MeshEntity< MeshConfig, MeshVertexTopology >::
~MeshEntity()
{
   //cerr << "   Destroying entity with " << MeshVertexTopology::dimensions << " dimensions..." << std::endl;
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
   str << "\t Coordinates = ( " << point << " )";
   MeshSuperentityAccess< MeshConfig, MeshVertexTopology >::print( str );
}

template< typename MeshConfig >
bool
MeshEntity< MeshConfig, MeshVertexTopology >::
operator==( const MeshEntity& entity ) const
{
   return ( //MeshSuperentityAccess< MeshConfig, MeshVertexTopology >::operator==( entity ) &&
            MeshEntityId< typename MeshConfig::IdType,
                             typename MeshConfig::GlobalIndexType >::operator==( entity ) &&
            point == entity.point );
}

template< typename MeshConfig >
constexpr int
MeshEntity< MeshConfig, MeshVertexTopology >::
getEntityDimensions() const
{
   return EntityTopology::dimensions;
}

template< typename MeshConfig >
   template< int Superdimensions >
typename MeshEntity< MeshConfig, MeshVertexTopology >::LocalIndexType
MeshEntity< MeshConfig, MeshVertexTopology >::
getNumberOfSuperentities() const
{
   typedef MeshSuperentityAccess< MeshConfig, MeshVertexTopology >  SuperentityBaseType;
   return SuperentityBaseType::getNumberOfSuperentities( MeshDimensionsTag< Superdimensions >() );
}

template< typename MeshConfig >
   template< int Superdimensions >
typename MeshEntity< MeshConfig, MeshVertexTopology >::template SuperentityTraits< Superdimensions >::AccessArrayType&
MeshEntity< MeshConfig, MeshVertexTopology >::
getSuperentitiesIndices()
{
   typedef MeshSuperentityAccess< MeshConfig, MeshVertexTopology >  SuperentityBaseType;
   return SuperentityBaseType::getSuperentitiesIndices( MeshDimensionsTag< Superdimensions >() );
}

template< typename MeshConfig >
   template< int Superdimensions >
const typename MeshEntity< MeshConfig, MeshVertexTopology >::template SuperentityTraits< Superdimensions >::AccessArrayType&
MeshEntity< MeshConfig, MeshVertexTopology >::
getSuperentitiesIndeces() const
{
   typedef MeshSuperentityAccess< MeshConfig, MeshVertexTopology >  SuperentityBaseType;
   return SuperentityBaseType::getSubentitiesIndices( MeshDimensionsTag< Superdimensions >() );
}

template< typename MeshConfig >
   template< int Dimensions >
typename MeshEntity< MeshConfig, MeshVertexTopology >::GlobalIndexType
MeshEntity< MeshConfig, MeshVertexTopology >::
getSuperentityIndex( const LocalIndexType localIndex ) const
{
   Assert( localIndex < this->getNumberOfSuperentities< Dimensions >(),
              std::cerr << " localIndex = " << localIndex
                   << " this->getNumberOfSuperentities< Dimensions >() = " << this->getNumberOfSuperentities< Dimensions >() << std::endl; );
   typedef MeshSuperentityAccess< MeshConfig, MeshVertexTopology >  SuperentityBaseType;
   return SuperentityBaseType::getSuperentityIndex( MeshDimensionsTag< Dimensions >(),
                                                    localIndex );
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

template< typename MeshConfig >
   template< int Superdimensions >
typename MeshEntity< MeshConfig, MeshVertexTopology >::MeshTraitsType::IdArrayAccessorType&
MeshEntity< MeshConfig, MeshVertexTopology >::
superentityIdsArray()
{
   return SuperentityAccessBase::superentityIdsArray( MeshDimensionsTag< Superdimensions >());
}

template< typename MeshConfig,
          typename EntityTopology >
std::ostream& operator <<( std::ostream& str, const MeshEntity< MeshConfig, EntityTopology >& entity )
{
   entity.print( str );
   return str;
}

} // namespace Meshes
} // namespace TNL
