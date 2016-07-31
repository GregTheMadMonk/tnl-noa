/***************************************************************************
                          tnlMeshEntity_impl.h  -  description
                             -------------------
    begin                : Sep 8, 2015
    copyright            : (C) 2015 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include "tnlMeshEntity.h"

namespace TNL {

template< typename MeshConfig,
          typename EntityTopology >
tnlMeshEntity< MeshConfig, EntityTopology >::
tnlMeshEntity( const SeedType& entitySeed )
{
   typedef typename SeedType::LocalIndexType LocalIndexType;
   for( LocalIndexType i = 0; i < entitySeed.getCornerIds().getSize(); i++ )
      this->template setSubentityIndex< 0 >( i, entitySeed.getCornerIds()[ i ] );
}


template< typename MeshConfig,
          typename EntityTopology >
tnlMeshEntity< MeshConfig, EntityTopology >::
tnlMeshEntity()
{
}

template< typename MeshConfig,
          typename EntityTopology >
tnlMeshEntity< MeshConfig, EntityTopology >::
~tnlMeshEntity()
{
   //cerr << "   Destroying entity with " << EntityTopology::dimensions << " dimensions..." << std::endl;
}

template< typename MeshConfig,
          typename EntityTopology >
String
tnlMeshEntity< MeshConfig, EntityTopology >::
getType()
{
   return String( "tnlMesh< ... >" );
}

template< typename MeshConfig,
          typename EntityTopology >
String
tnlMeshEntity< MeshConfig, EntityTopology >::
getTypeVirtual() const
{
   return this->getType();
}

template< typename MeshConfig,
          typename EntityTopology >
bool
tnlMeshEntity< MeshConfig, EntityTopology >::
save( File& file ) const
{
   if( ! tnlMeshSubentityStorageLayers< MeshConfig, EntityTopology >::save( file ) /*||
       ! tnlMeshSuperentityStorageLayers< MeshConfig, EntityTopology >::save( file )*/ )
      return false;
   return true;
}

template< typename MeshConfig,
          typename EntityTopology >
bool
tnlMeshEntity< MeshConfig, EntityTopology >::
load( File& file )
{
   if( ! tnlMeshSubentityStorageLayers< MeshConfig, EntityTopology >::load( file ) /*||
       ! tnlMeshSuperentityStorageLayers< MeshConfig, EntityTopology >::load( file ) */ )
      return false;
   return true;
}

template< typename MeshConfig,
          typename EntityTopology >
void
tnlMeshEntity< MeshConfig, EntityTopology >::
print( std::ostream& str ) const
{
   str << "\t Mesh entity dimensions: " << EntityTopology::dimensions << std::endl;
   tnlMeshSubentityStorageLayers< MeshConfig, EntityTopology >::print( str );
   tnlMeshSuperentityAccess< MeshConfig, EntityTopology >::print( str );
}

template< typename MeshConfig,
          typename EntityTopology >
bool
tnlMeshEntity< MeshConfig, EntityTopology >::
operator==( const tnlMeshEntity& entity ) const
{
   return ( tnlMeshSubentityStorageLayers< MeshConfig, EntityTopology >::operator==( entity ) &&
            tnlMeshSuperentityAccess< MeshConfig, EntityTopology >::operator==( entity ) &&
            tnlMeshEntityId< typename MeshConfig::IdType,
                             typename MeshConfig::GlobalIndexType >::operator==( entity ) );
}

template< typename MeshConfig,
          typename EntityTopology >
constexpr int
tnlMeshEntity< MeshConfig, EntityTopology >::
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
tnlMeshEntity< MeshConfig, EntityTopology >::
subentitiesAvailable() const
{
   return SubentityTraits< Subdimensions >::storageEnabled;
};

template< typename MeshConfig,
          typename EntityTopology >
   template< int Subdimensions >
constexpr typename tnlMeshEntity< MeshConfig, EntityTopology >::LocalIndexType
tnlMeshEntity< MeshConfig, EntityTopology >::
getNumberOfSubentities() const
{
   return SubentityTraits< Subdimensions >::count;
};

template< typename MeshConfig,
          typename EntityTopology >
   template< int Subdimensions >
typename tnlMeshEntity< MeshConfig, EntityTopology >::GlobalIndexType
tnlMeshEntity< MeshConfig, EntityTopology >::
getSubentityIndex( const LocalIndexType localIndex) const
{
   static_assert( SubentityTraits< Subdimensions >::storageEnabled, "You try to get subentity which is not configured for storage." );
   Assert( 0 <= localIndex &&
              localIndex < SubentityTraits< Subdimensions >::count,
              std::cerr << "localIndex = " << localIndex
                   << " subentitiesCount = "
                   << SubentityTraits< Subdimensions >::count );
   typedef tnlMeshSubentityStorageLayers< MeshConfig, EntityTopology >  SubentityBaseType;
   return SubentityBaseType::getSubentityIndex( tnlDimensionsTag< Subdimensions >(),
                                                localIndex );
}

template< typename MeshConfig,
          typename EntityTopology >
   template< int Subdimensions >
typename tnlMeshEntity< MeshConfig, EntityTopology >::template SubentityTraits< Subdimensions >::AccessArrayType&
tnlMeshEntity< MeshConfig, EntityTopology >::
 getSubentitiesIndices()
{
   static_assert( SubentityTraits< Subdimensions >::storageEnabled, "You try to get subentities which are not configured for storage." );
   typedef tnlMeshSubentityStorageLayers< MeshConfig, EntityTopology >  SubentityBaseType;
   return SubentityBaseType::getSubentitiesIndices( tnlDimensionsTag< Subdimensions >() );
}

template< typename MeshConfig,
          typename EntityTopology >
   template< int Subdimensions >
const typename tnlMeshEntity< MeshConfig, EntityTopology >::template SubentityTraits< Subdimensions >::AccessArrayType&
tnlMeshEntity< MeshConfig, EntityTopology >::
getSubentitiesIndices() const
{
   static_assert( SubentityTraits< Subdimensions >::storageEnabled, "You try to set subentities which are not configured for storage." );
   typedef tnlMeshSubentityStorageLayers< MeshConfig, EntityTopology >  SubentityBaseType;
   return SubentityBaseType::getSubentitiesIndices( tnlDimensionsTag< Subdimensions >() );
}

template< typename MeshConfig,
          typename EntityTopology >
   template< int SuperDimensions >
typename tnlMeshEntity< MeshConfig, EntityTopology >::LocalIndexType
tnlMeshEntity< MeshConfig, EntityTopology >::
getNumberOfSuperentities() const
{
   static_assert( SuperentityTraits< SuperDimensions >::available, "You try to get number of superentities which are not configured for storage." );
   typedef tnlMeshSuperentityAccess< MeshConfig, EntityTopology >  SuperentityBaseType;
   return SuperentityBaseType::getNumberOfSuperentities( tnlDimensionsTag< SuperDimensions >() );
}

template< typename MeshConfig,
          typename EntityTopology >
   template< int SuperDimensions >
typename tnlMeshEntity< MeshConfig, EntityTopology >::GlobalIndexType
tnlMeshEntity< MeshConfig, EntityTopology >::
getSuperentityIndex( const LocalIndexType localIndex ) const
{
   static_assert( SuperentityTraits< SuperDimensions >::storageEnabled, "You try to get superentity which is not configured for storage." );
   Assert( localIndex < this->getNumberOfSuperentities< SuperDimensions >(),
              std::cerr << " localIndex = " << localIndex
                   << " this->getNumberOfSuperentities< Dimensions >() = " << this->getNumberOfSuperentities< SuperDimensions >() << std::endl; );
   typedef tnlMeshSuperentityAccess< MeshConfig, EntityTopology >  SuperentityBaseType;
   return SuperentityBaseType::getSuperentityIndex( tnlDimensionsTag< SuperDimensions >(),
                                                    localIndex );
}

template< typename MeshConfig,
          typename EntityTopology >
   template< int SuperDimensions >
typename tnlMeshEntity< MeshConfig, EntityTopology >::template SuperentityTraits< SuperDimensions >::AccessArrayType&
tnlMeshEntity< MeshConfig, EntityTopology >::
getSuperentitiesIndices()
{
   static_assert( SuperentityTraits< SuperDimensions >::storageEnabled, "You try to get superentities which are not configured for storage." );
   typedef tnlMeshSuperentityAccess< MeshConfig, EntityTopology >  SuperentityBaseType;
   //return SuperentityBaseType::getSuperentitiesIndices( tnlDimensionsTag< Dimensions >() );
}

template< typename MeshConfig,
          typename EntityTopology >
   template< int SuperDimensions >
const typename tnlMeshEntity< MeshConfig, EntityTopology >::template SuperentityTraits< SuperDimensions >::AccessArrayType&
tnlMeshEntity< MeshConfig, EntityTopology >::
getSuperentitiesIndices() const
{
   static_assert( SuperentityTraits< SuperDimensions >::storageEnabled, "You try to get superentities which are not configured for storage." );
   typedef tnlMeshSuperentityAccess< MeshConfig, EntityTopology >  SuperentityBaseType;
   return SuperentityBaseType::getSubentitiesIndices( tnlDimensionsTag< SuperDimensions >() );
}

template< typename MeshConfig,
          typename EntityTopology >
constexpr typename tnlMeshEntity< MeshConfig, EntityTopology >::LocalIndexType
tnlMeshEntity< MeshConfig, EntityTopology >::
getNumberOfVertices() const
{
   return SubentityTraits< 0 >::count;
}

template< typename MeshConfig,
          typename EntityTopology >
typename tnlMeshEntity< MeshConfig, EntityTopology >::GlobalIndexType
tnlMeshEntity< MeshConfig, EntityTopology >::
getVertexIndex( const LocalIndexType localIndex ) const
{
   return this->getSubentityIndex< 0 >( localIndex  );
}

template< typename MeshConfig,
          typename EntityTopology >
typename tnlMeshEntity< MeshConfig, EntityTopology >::template SubentityTraits< 0 >::AccessArrayType&
tnlMeshEntity< MeshConfig, EntityTopology >::
getVerticesIndices()
{
   return this->getSubentitiesIndices< 0 >();
}

template< typename MeshConfig,
          typename EntityTopology >
const typename tnlMeshEntity< MeshConfig, EntityTopology >::template SubentityTraits< 0 >::AccessArrayType&
tnlMeshEntity< MeshConfig, EntityTopology >::
getVerticesIndices() const
{
   return this->getSubentitiesIndices< 0 >();
}

template< typename MeshConfig,
          typename EntityTopology >
   template< int Dimensions >
typename tnlMeshEntity< MeshConfig, EntityTopology >::IdPermutationArrayAccessorType
tnlMeshEntity< MeshConfig, EntityTopology >::
subentityOrientation( LocalIndexType index ) const
{
   static const LocalIndexType subentitiesCount = SubentityTraits< Dimensions >::count;
   Assert( 0 <= index && index < subentitiesCount, );

   return SubentityStorageLayers::subentityOrientation( tnlDimensionsTag< Dimensions >(), index );
}

/****
 * Mesh initialization method
 */

template< typename MeshConfig,
          typename EntityTopology >
   template< int Subdimensions >
void
tnlMeshEntity< MeshConfig, EntityTopology >::
setSubentityIndex( const LocalIndexType localIndex,
                   const GlobalIndexType globalIndex )
{
   static_assert( SubentityTraits< Subdimensions >::storageEnabled, "You try to set subentity which is not configured for storage." );
   Assert( 0 <= localIndex &&
              localIndex < SubentityTraits< Subdimensions >::count,
              std::cerr << "localIndex = " << localIndex
                   << " subentitiesCount = "
                   << SubentityTraits< Subdimensions >::count );
   typedef tnlMeshSubentityStorageLayers< MeshConfig, EntityTopology >  SubentityBaseType;
   SubentityBaseType::setSubentityIndex( tnlDimensionsTag< Subdimensions >(),
                                         localIndex,
                                         globalIndex );
}

template< typename MeshConfig,
          typename EntityTopology >
   template< int Subdimensions >
typename tnlMeshEntity< MeshConfig, EntityTopology >::template SubentityTraits< Subdimensions >::IdArrayType&
tnlMeshEntity< MeshConfig, EntityTopology >::
subentityIdsArray()
{
   return SubentityStorageLayers::subentityIdsArray( tnlDimensionsTag< Subdimensions >() );
}

template< typename MeshConfig,
          typename EntityTopology >
   template< int Superdimensions >
typename tnlMeshEntity< MeshConfig, EntityTopology >::IdArrayAccessorType&
tnlMeshEntity< MeshConfig, EntityTopology >::
superentityIdsArray()
{
   return SuperentityAccessBase::superentityIdsArray( tnlDimensionsTag< Superdimensions >());
}

template< typename MeshConfig,
          typename EntityTopology >
   template< int Subdimensions >
typename tnlMeshEntity< MeshConfig, EntityTopology >::template SubentityTraits< Subdimensions >::OrientationArrayType&
tnlMeshEntity< MeshConfig, EntityTopology >::
subentityOrientationsArray()
{
   return SubentityStorageLayers::subentityOrientationsArray( tnlDimensionsTag< Subdimensions >() );
}

/****
 * Vertex entity specialization
 */
template< typename MeshConfig >
String
tnlMeshEntity< MeshConfig, tnlMeshVertexTopology >::
getType()
{
   return String( "tnlMesh< ... >" );
}

template< typename MeshConfig >
String
tnlMeshEntity< MeshConfig, tnlMeshVertexTopology >::
getTypeVirtual() const
{
   return this->getType();
}

template< typename MeshConfig >
tnlMeshEntity< MeshConfig, tnlMeshVertexTopology >::
~tnlMeshEntity()
{
   //cerr << "   Destroying entity with " << tnlMeshVertexTopology::dimensions << " dimensions..." << std::endl;
}

template< typename MeshConfig >
bool
tnlMeshEntity< MeshConfig, tnlMeshVertexTopology >::
save( File& file ) const
{
   if( //! tnlMeshSuperentityStorageLayers< MeshConfig, tnlMeshVertexTopology >::save( file ) ||
       ! point.save( file ) )
      return false;
   return true;
}

template< typename MeshConfig >
bool
tnlMeshEntity< MeshConfig, tnlMeshVertexTopology >::
load( File& file )
{
   if( //! tnlMeshSuperentityStorageLayers< MeshConfig, tnlMeshVertexTopology >::load( file ) ||
       ! point.load( file ) )
      return false;
   return true;
}

template< typename MeshConfig >
void
tnlMeshEntity< MeshConfig, tnlMeshVertexTopology >::
print( std::ostream& str ) const
{
   str << "\t Mesh entity dimensions: " << tnlMeshVertexTopology::dimensions << std::endl;
   str << "\t Coordinates = ( " << point << " )";
   tnlMeshSuperentityAccess< MeshConfig, tnlMeshVertexTopology >::print( str );
}

template< typename MeshConfig >
bool
tnlMeshEntity< MeshConfig, tnlMeshVertexTopology >::
operator==( const tnlMeshEntity& entity ) const
{
   return ( //tnlMeshSuperentityAccess< MeshConfig, tnlMeshVertexTopology >::operator==( entity ) &&
            tnlMeshEntityId< typename MeshConfig::IdType,
                             typename MeshConfig::GlobalIndexType >::operator==( entity ) &&
            point == entity.point );
}

template< typename MeshConfig >
constexpr int
tnlMeshEntity< MeshConfig, tnlMeshVertexTopology >::
getEntityDimensions() const
{
   return EntityTopology::dimensions;
}

template< typename MeshConfig >
   template< int Superdimensions >
typename tnlMeshEntity< MeshConfig, tnlMeshVertexTopology >::LocalIndexType
tnlMeshEntity< MeshConfig, tnlMeshVertexTopology >::
getNumberOfSuperentities() const
{
   typedef tnlMeshSuperentityAccess< MeshConfig, tnlMeshVertexTopology >  SuperentityBaseType;
   return SuperentityBaseType::getNumberOfSuperentities( tnlDimensionsTag< Superdimensions >() );
}

template< typename MeshConfig >
   template< int Superdimensions >
typename tnlMeshEntity< MeshConfig, tnlMeshVertexTopology >::template SuperentityTraits< Superdimensions >::AccessArrayType&
tnlMeshEntity< MeshConfig, tnlMeshVertexTopology >::
getSuperentitiesIndices()
{
   typedef tnlMeshSuperentityAccess< MeshConfig, tnlMeshVertexTopology >  SuperentityBaseType;
   return SuperentityBaseType::getSuperentitiesIndices( tnlDimensionsTag< Superdimensions >() );
}

template< typename MeshConfig >
   template< int Superdimensions >
const typename tnlMeshEntity< MeshConfig, tnlMeshVertexTopology >::template SuperentityTraits< Superdimensions >::AccessArrayType&
tnlMeshEntity< MeshConfig, tnlMeshVertexTopology >::
getSuperentitiesIndeces() const
{
   typedef tnlMeshSuperentityAccess< MeshConfig, tnlMeshVertexTopology >  SuperentityBaseType;
   return SuperentityBaseType::getSubentitiesIndices( tnlDimensionsTag< Superdimensions >() );
}

template< typename MeshConfig >
   template< int Dimensions >
typename tnlMeshEntity< MeshConfig, tnlMeshVertexTopology >::GlobalIndexType
tnlMeshEntity< MeshConfig, tnlMeshVertexTopology >::
getSuperentityIndex( const LocalIndexType localIndex ) const
{
   Assert( localIndex < this->getNumberOfSuperentities< Dimensions >(),
              std::cerr << " localIndex = " << localIndex
                   << " this->getNumberOfSuperentities< Dimensions >() = " << this->getNumberOfSuperentities< Dimensions >() << std::endl; );
   typedef tnlMeshSuperentityAccess< MeshConfig, tnlMeshVertexTopology >  SuperentityBaseType;
   return SuperentityBaseType::getSuperentityIndex( tnlDimensionsTag< Dimensions >(),
                                                    localIndex );
}

template< typename MeshConfig >
typename tnlMeshEntity< MeshConfig, tnlMeshVertexTopology >::PointType
tnlMeshEntity< MeshConfig, tnlMeshVertexTopology >::
getPoint() const
{
   return this->point;
}

template< typename MeshConfig >
void
tnlMeshEntity< MeshConfig, tnlMeshVertexTopology >::
setPoint( const PointType& point )
{
   this->point = point;
}

template< typename MeshConfig >
   template< int Superdimensions >
typename tnlMeshEntity< MeshConfig, tnlMeshVertexTopology >::MeshTraits::IdArrayAccessorType&
tnlMeshEntity< MeshConfig, tnlMeshVertexTopology >::
superentityIdsArray()
{
   return SuperentityAccessBase::superentityIdsArray( tnlDimensionsTag< Superdimensions >());
}

template< typename MeshConfig,
          typename EntityTopology >
std::ostream& operator <<( std::ostream& str, const tnlMeshEntity< MeshConfig, EntityTopology >& entity )
{
   entity.print( str );
   return str;
}

} // namespace TNL

