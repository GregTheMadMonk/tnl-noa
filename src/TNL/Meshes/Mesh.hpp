/***************************************************************************
                          Mesh.hpp  -  description
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
#include <TNL/Meshes/MeshDetails/IndexPermutationApplier.h>
#include <TNL/Meshes/MeshDetails/initializer/Initializer.h>

namespace TNL {
namespace Meshes {

template< typename MeshConfig, typename Device, typename MeshType >
void
MeshInitializableBase< MeshConfig, Device, MeshType >::
init( typename MeshTraitsType::PointArrayType& points,
      typename MeshTraitsType::CellSeedArrayType& cellSeeds )
{
   MeshType* mesh = static_cast< MeshType* >( this );
   Initializer< typename MeshType::Config > initializer;
   initializer.createMesh( points, cellSeeds, *mesh );
   // init boundary tags
   static_cast< EntityTags::LayerFamily< MeshConfig, Device, MeshType >* >( mesh )->initLayer();
   // init dual graph
   mesh->initializeDualGraph( *mesh );
}


template< typename MeshConfig, typename Device >
Mesh< MeshConfig, Device >::
Mesh( const Mesh& mesh )
   : StorageBaseType( mesh ),
     EntityTagsLayerFamily( mesh )
{
}

template< typename MeshConfig, typename Device >
   template< typename Device_ >
Mesh< MeshConfig, Device >::
Mesh( const Mesh< MeshConfig, Device_ >& mesh )
   : StorageBaseType( mesh ),
     EntityTagsLayerFamily( mesh )
{
}

template< typename MeshConfig, typename Device >
Mesh< MeshConfig, Device >&
Mesh< MeshConfig, Device >::
operator=( const Mesh& mesh )
{
   StorageBaseType::operator=( mesh );
   EntityTagsLayerFamily::operator=( mesh );
   return *this;
}

template< typename MeshConfig, typename Device >
   template< typename Device_ >
Mesh< MeshConfig, Device >&
Mesh< MeshConfig, Device >::
operator=( const Mesh< MeshConfig, Device_ >& mesh )
{
   StorageBaseType::operator=( mesh );
   EntityTagsLayerFamily::operator=( mesh );
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
getSerializationType()
{
   return String( "Meshes::Mesh< ") + TNL::getType< MeshConfig >() + " >";
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
__cuda_callable__
typename Mesh< MeshConfig, Device >::GlobalIndexType
Mesh< MeshConfig, Device >::
getEntitiesCount() const
{
   return StorageBaseType::getEntitiesCount( DimensionTag< Dimension >() );
}

template< typename MeshConfig, typename Device >
   template< int Dimension >
__cuda_callable__
typename Mesh< MeshConfig, Device >::template EntityType< Dimension >
Mesh< MeshConfig, Device >::
getEntity( const GlobalIndexType entityIndex ) const
{
   TNL_ASSERT_LT( entityIndex, getEntitiesCount< Dimension >(), "invalid entity index" );
   return EntityType< Dimension >( *this, entityIndex );
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
Entity
Mesh< MeshConfig, Device >::
getEntity( const GlobalIndexType entityIndex ) const
{
   return getEntity< Entity::getEntityDimension() >( entityIndex );
}


template< typename MeshConfig, typename Device >
__cuda_callable__
const typename Mesh< MeshConfig, Device >::PointType&
Mesh< MeshConfig, Device >::
getPoint( const GlobalIndexType vertexIndex ) const
{
   TNL_ASSERT_GE( vertexIndex, 0, "invalid vertex index" );
   TNL_ASSERT_LT( vertexIndex, getEntitiesCount< 0 >(), "invalid vertex index" );
   return this->points[ vertexIndex ];
}

template< typename MeshConfig, typename Device >
__cuda_callable__
typename Mesh< MeshConfig, Device >::PointType&
Mesh< MeshConfig, Device >::
getPoint( const GlobalIndexType vertexIndex )
{
   TNL_ASSERT_GE( vertexIndex, 0, "invalid vertex index" );
   TNL_ASSERT_LT( vertexIndex, getEntitiesCount< 0 >(), "invalid vertex index" );
   return this->points[ vertexIndex ];
}


template< typename MeshConfig, typename Device >
   template< int EntityDimension, int SubentityDimension >
__cuda_callable__
constexpr typename Mesh< MeshConfig, Device >::LocalIndexType
Mesh< MeshConfig, Device >::
getSubentitiesCount( const GlobalIndexType entityIndex ) const
{
   return StorageBaseType::template getSubentitiesCount< EntityDimension, SubentityDimension >();
}

template< typename MeshConfig, typename Device >
   template< int EntityDimension, int SubentityDimension >
__cuda_callable__
typename Mesh< MeshConfig, Device >::GlobalIndexType
Mesh< MeshConfig, Device >::
getSubentityIndex( const GlobalIndexType entityIndex, const LocalIndexType subentityIndex ) const
{
   const auto& row = this->template getSubentitiesMatrix< EntityDimension, SubentityDimension >().getRow( entityIndex );
   TNL_ASSERT_GE( row.getColumnIndex( subentityIndex ), 0, "padding index returned for given subentity index" );
   return row.getColumnIndex( subentityIndex );
}

template< typename MeshConfig, typename Device >
   template< int EntityDimension, int SuperentityDimension >
__cuda_callable__
typename Mesh< MeshConfig, Device >::LocalIndexType
Mesh< MeshConfig, Device >::
getSuperentitiesCount( const GlobalIndexType entityIndex ) const
{
   return this->template getSuperentitiesCountsArray< EntityDimension, SuperentityDimension >()[ entityIndex ];
}

template< typename MeshConfig, typename Device >
   template< int EntityDimension, int SuperentityDimension >
__cuda_callable__
typename Mesh< MeshConfig, Device >::GlobalIndexType
Mesh< MeshConfig, Device >::
getSuperentityIndex( const GlobalIndexType entityIndex, const LocalIndexType superentityIndex ) const
{
   const auto row = this->template getSuperentitiesMatrix< EntityDimension, SuperentityDimension >().getRow( entityIndex );
   TNL_ASSERT_GE( row.getColumnIndex( superentityIndex ), 0, "padding index returned for given superentity index" );
   return row.getColumnIndex( superentityIndex );
}

template< typename MeshConfig, typename Device >
__cuda_callable__
typename Mesh< MeshConfig, Device >::LocalIndexType
Mesh< MeshConfig, Device >::
getCellNeighborsCount( const GlobalIndexType cellIndex ) const
{
   static_assert( MeshConfig::dualGraphStorage(),
                  "You try to access the dual graph which is disabled in the mesh configuration." );
   return this->getNeighborCounts()[ cellIndex ];
}

template< typename MeshConfig, typename Device >
__cuda_callable__
typename Mesh< MeshConfig, Device >::GlobalIndexType
Mesh< MeshConfig, Device >::
getCellNeighborIndex( const GlobalIndexType cellIndex, const LocalIndexType neighborIndex ) const
{
   static_assert( MeshConfig::dualGraphStorage(),
                  "You try to access the dual graph which is disabled in the mesh configuration." );
   TNL_ASSERT_GE( neighborIndex, 0, "Invalid cell neighbor index." );
   TNL_ASSERT_LT( neighborIndex, getCellNeighborsCount( cellIndex ), "Invalid cell neighbor index." );
   const auto row = this->getDualGraph().getRow( cellIndex );
   TNL_ASSERT_GE( row.getColumnIndex( neighborIndex ), 0, "padding index returned for given neighbor index" );
   return row.getColumnIndex( neighborIndex );
}


template< typename MeshConfig, typename Device >
   template< int Dimension >
void
Mesh< MeshConfig, Device >::
reorderEntities( const GlobalIndexArray& perm,
                 const GlobalIndexArray& iperm )
{
   const GlobalIndexType entitiesCount = getEntitiesCount< Dimension >();

   // basic sanity check
   if( perm.getSize() != entitiesCount || iperm.getSize() != entitiesCount ) {
      throw std::logic_error( "Wrong size of permutation vectors: "
                              "perm size = " + std::to_string( perm.getSize() ) + ", "
                              "iperm size = " + std::to_string( iperm.getSize() ) );
   }
#ifndef NDEBUG
   using View = Containers::VectorView< const GlobalIndexType, DeviceType, GlobalIndexType >;
   const View perm_view = perm.getConstView();
   const View iperm_view = iperm.getConstView();
   TNL_ASSERT( min( perm_view ) == 0 && max( perm_view ) == entitiesCount - 1,
               std::cerr << "Given array is not a permutation: min = " << min( perm_view )
                         << ", max = " << max( perm_view )
                         << ", number of entities = " << entitiesCount
                         << ", array = " << perm << std::endl; );
   TNL_ASSERT( min( iperm_view ) == 0 && max( iperm_view ) == entitiesCount - 1,
               std::cerr << "Given array is not a permutation: min = " << min( iperm_view )
                         << ", max = " << max( iperm_view )
                         << ", number of entities = " << entitiesCount
                         << ", array = " << iperm << std::endl; );
#endif

   IndexPermutationApplier< Mesh, Dimension >::exec( *this, perm, iperm );
   // update boundary tags
   static_cast< EntityTagsLayerFamily* >( this )->initLayer();
}


template< typename MeshConfig, typename Device >
void
Mesh< MeshConfig, Device >::
save( File& file ) const
{
   // saving via host is necessary due to segment-based sparse matrices
   if( std::is_same< Device, Devices::Cuda >::value ) {
      Mesh< MeshConfig, Devices::Host > hostMesh;
      hostMesh = *this;
      hostMesh.save( file );
   }
   else {
      Object::save( file );
      StorageBaseType::save( file );
      EntityTagsLayerFamily::save( file );
   }
}

template< typename MeshConfig, typename Device >
void
Mesh< MeshConfig, Device >::
load( File& file )
{
   // loading via host is necessary for the initialization of the dual graph (and due to segment-based sparse matrices)
   if( std::is_same< Device, Devices::Cuda >::value ) {
      Mesh< MeshConfig, Devices::Host > hostMesh;
      hostMesh.load( file );
      *this = hostMesh;
   }
   else {
      Object::load( file );
      StorageBaseType::load( file );
      EntityTagsLayerFamily::load( file );
      this->initializeDualGraph( *this );
   }
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
      EntityTagsLayerFamily::print( str );
   }
}

template< typename MeshConfig, typename Device >
bool
Mesh< MeshConfig, Device >::
operator==( const Mesh& mesh ) const
{
   return StorageBaseType::operator==( mesh ) &&
          EntityTagsLayerFamily::operator==( mesh );
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
writeProlog( Logger& logger ) const
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
