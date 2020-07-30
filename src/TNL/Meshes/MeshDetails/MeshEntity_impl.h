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
          typename Device,
          typename EntityTopology >
__cuda_callable__
MeshEntity< MeshConfig, Device, EntityTopology >::
MeshEntity( const MeshType& mesh, const GlobalIndexType index )
: meshPointer( &mesh ),
  index( index )
{
}

template< typename MeshConfig,
          typename Device,
          typename EntityTopology >
__cuda_callable__
bool
MeshEntity< MeshConfig, Device, EntityTopology >::
operator==( const MeshEntity& entity ) const
{
   return getIndex() == entity.getIndex();
}

template< typename MeshConfig,
          typename Device,
          typename EntityTopology >
__cuda_callable__
bool
MeshEntity< MeshConfig, Device, EntityTopology >::
operator!=( const MeshEntity& entity ) const
{
   return ! ( *this == entity );
}

template< typename MeshConfig,
          typename Device,
          typename EntityTopology >
constexpr int
MeshEntity< MeshConfig, Device, EntityTopology >::
getEntityDimension()
{
   return EntityTopology::dimension;
}

template< typename MeshConfig,
          typename Device,
          typename EntityTopology >
__cuda_callable__
const Mesh< MeshConfig, Device >&
MeshEntity< MeshConfig, Device, EntityTopology >::
getMesh() const
{
   return *meshPointer;
}

template< typename MeshConfig,
          typename Device,
          typename EntityTopology >
__cuda_callable__
typename Mesh< MeshConfig, Device >::GlobalIndexType
MeshEntity< MeshConfig, Device, EntityTopology >::
getIndex() const
{
   return index;
}

template< typename MeshConfig,
          typename Device,
          typename EntityTopology >
__cuda_callable__
typename MeshEntity< MeshConfig, Device, EntityTopology >::PointType
MeshEntity< MeshConfig, Device, EntityTopology >::
getPoint() const
{
   static_assert( getEntityDimension() == 0, "getPoint() can be used only on vertices" );
   return meshPointer->getPoint( getIndex() );
}

template< typename MeshConfig,
          typename Device,
          typename EntityTopology >
   template< int Subdimension >
__cuda_callable__
typename MeshEntity< MeshConfig, Device, EntityTopology >::LocalIndexType
MeshEntity< MeshConfig, Device, EntityTopology >::
getSubentitiesCount() const
{
   TNL_ASSERT_TRUE( meshPointer, "meshPointer was not set" );
   return meshPointer->template getSubentitiesCount< getEntityDimension(), Subdimension >( this->getIndex() );
}

template< typename MeshConfig,
          typename Device,
          typename EntityTopology >
   template< int Subdimension >
__cuda_callable__
typename MeshEntity< MeshConfig, Device, EntityTopology >::GlobalIndexType
MeshEntity< MeshConfig, Device, EntityTopology >::
getSubentityIndex( const LocalIndexType localIndex ) const
{
   TNL_ASSERT_TRUE( meshPointer, "meshPointer was not set" );
   return meshPointer->template getSubentityIndex< getEntityDimension(), Subdimension >( this->getIndex(), localIndex );
}

template< typename MeshConfig,
          typename Device,
          typename EntityTopology >
   template< int Subdimension >
__cuda_callable__
const typename MeshEntity< MeshConfig, Device, EntityTopology >::template SubentityTraits< Subdimension >::OrientationArrayType&
MeshEntity< MeshConfig, Device, EntityTopology >::
getSubentityOrientation( const LocalIndexType localIndex ) const
{
   TNL_ASSERT_TRUE( meshPointer, "meshPointer was not set" );
   return meshPointer->template getSubentityOrientation< getEntityDimension(), Subdimension >( this->getIndex(), localIndex );
}

template< typename MeshConfig,
          typename Device,
          typename EntityTopology >
   template< int Superdimension >
__cuda_callable__
typename MeshEntity< MeshConfig, Device, EntityTopology >::LocalIndexType
MeshEntity< MeshConfig, Device, EntityTopology >::
getSuperentitiesCount() const
{
   TNL_ASSERT_TRUE( meshPointer, "meshPointer was not set" );
   return meshPointer->template getSuperentitiesCount< getEntityDimension(), Superdimension >( this->getIndex() );
}

template< typename MeshConfig,
          typename Device,
          typename EntityTopology >
   template< int Superdimension >
__cuda_callable__
typename MeshEntity< MeshConfig, Device, EntityTopology >::GlobalIndexType
MeshEntity< MeshConfig, Device, EntityTopology >::
getSuperentityIndex( const LocalIndexType localIndex ) const
{
   TNL_ASSERT_TRUE( meshPointer, "meshPointer was not set" );
   return meshPointer->template getSuperentityIndex< getEntityDimension(), Superdimension >( this->getIndex(), localIndex );
}

template< typename MeshConfig,
          typename Device,
          typename EntityTopology >
std::ostream& operator<<( std::ostream& str, const MeshEntity< MeshConfig, Device, EntityTopology >& entity )
{
   return str << getType< decltype(entity) >() << "( <meshPointer>, " << entity.getIndex() << " )";
}

} // namespace Meshes
} // namespace TNL
