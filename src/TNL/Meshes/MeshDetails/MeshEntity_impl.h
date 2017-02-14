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
String
MeshEntity< MeshConfig, Device, EntityTopology >::
getType()
{
   return String( "MeshEntity< " ) +
          MeshConfig::getType() + ", " +
          EntityTopology::getType() + " >";
}

template< typename MeshConfig,
          typename Device,
          typename EntityTopology >
String
MeshEntity< MeshConfig, Device, EntityTopology >::
getTypeVirtual() const
{
   return this->getType();
}

template< typename MeshConfig,
          typename Device,
          typename EntityTopology >
bool
MeshEntity< MeshConfig, Device, EntityTopology >::
save( File& file ) const
{
   if( ! MeshSubentityAccess< MeshConfig, EntityTopology >::save( file ) )
      return false;
   return true;
}

template< typename MeshConfig,
          typename Device,
          typename EntityTopology >
bool
MeshEntity< MeshConfig, Device, EntityTopology >::
load( File& file )
{
   if( ! MeshSubentityAccess< MeshConfig, EntityTopology >::load( file ) )
      return false;
   return true;
}

template< typename MeshConfig,
          typename Device,
          typename EntityTopology >
void
MeshEntity< MeshConfig, Device, EntityTopology >::
print( std::ostream& str ) const
{
   str << "\t Mesh entity dimension: " << EntityTopology::dimension << std::endl;
   MeshSubentityAccess< MeshConfig, EntityTopology >::print( str );
   MeshSuperentityAccess< MeshConfig, EntityTopology >::print( str );
}

template< typename MeshConfig,
          typename Device,
          typename EntityTopology >
bool
MeshEntity< MeshConfig, Device, EntityTopology >::
operator==( const MeshEntity& entity ) const
{
   return ( MeshSubentityAccess< MeshConfig, EntityTopology >::operator==( entity ) &&
            MeshSuperentityAccess< MeshConfig, EntityTopology >::operator==( entity ) &&
            MeshEntityIndex< typename MeshConfig::IdType >::operator==( entity ) );
}

template< typename MeshConfig,
          typename Device,
          typename EntityTopology >
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

/****
 * Subentities
 */
template< typename MeshConfig,
          typename Device,
          typename EntityTopology >
constexpr typename MeshEntity< MeshConfig, Device, EntityTopology >::LocalIndexType
MeshEntity< MeshConfig, Device, EntityTopology >::
getVerticesCount()
{
   return SubentityTraits< 0 >::count;
}

template< typename MeshConfig,
          typename Device,
          typename EntityTopology >
typename MeshEntity< MeshConfig, Device, EntityTopology >::GlobalIndexType
MeshEntity< MeshConfig, Device, EntityTopology >::
getVertexIndex( const LocalIndexType localIndex ) const
{
   return this->template getSubentityIndex< 0 >( localIndex  );
}


/****
 * Vertex entity specialization
 */
template< typename MeshConfig, typename Device >
String
MeshEntity< MeshConfig, Device, MeshVertexTopology >::
getType()
{
   return String( "MeshEntity< ... >" );
}

template< typename MeshConfig, typename Device >
String
MeshEntity< MeshConfig, Device, MeshVertexTopology >::
getTypeVirtual() const
{
   return this->getType();
}

template< typename MeshConfig, typename Device >
bool
MeshEntity< MeshConfig, Device, MeshVertexTopology >::
save( File& file ) const
{
   if( //! MeshSuperentityStorageLayers< MeshConfig, Device, MeshVertexTopology >::save( file ) ||
       ! point.save( file ) )
      return false;
   return true;
}

template< typename MeshConfig, typename Device >
bool
MeshEntity< MeshConfig, Device, MeshVertexTopology >::
load( File& file )
{
   if( //! MeshSuperentityStorageLayers< MeshConfig, Device, MeshVertexTopology >::load( file ) ||
       ! point.load( file ) )
      return false;
   return true;
}

template< typename MeshConfig, typename Device >
void
MeshEntity< MeshConfig, Device, MeshVertexTopology >::
print( std::ostream& str ) const
{
   str << "\t Mesh entity dimension: " << MeshVertexTopology::dimension << std::endl;
   str << "\t Coordinates = " << point << std::endl;
   MeshSuperentityAccess< MeshConfig, MeshVertexTopology >::print( str );
}

template< typename MeshConfig, typename Device >
bool
MeshEntity< MeshConfig, Device, MeshVertexTopology >::
operator==( const MeshEntity& entity ) const
{
   return ( MeshSuperentityAccess< MeshConfig, MeshVertexTopology >::operator==( entity ) &&
            MeshEntityIndex< typename MeshConfig::IdType >::operator==( entity ) &&
            point == entity.point );
}

template< typename MeshConfig, typename Device >
bool
MeshEntity< MeshConfig, Device, MeshVertexTopology >::
operator!=( const MeshEntity& entity ) const
{
   return ! ( *this == entity );
}

template< typename MeshConfig, typename Device >
constexpr int
MeshEntity< MeshConfig, Device, MeshVertexTopology >::
getEntityDimension()
{
   return EntityTopology::dimension;
}

template< typename MeshConfig, typename Device >
typename MeshEntity< MeshConfig, Device, MeshVertexTopology >::PointType
MeshEntity< MeshConfig, Device, MeshVertexTopology >::
getPoint() const
{
   return this->point;
}

template< typename MeshConfig, typename Device >
void
MeshEntity< MeshConfig, Device, MeshVertexTopology >::
setPoint( const PointType& point )
{
   this->point = point;
}

template< typename MeshConfig,
          typename Device,
          typename EntityTopology >
std::ostream& operator<<( std::ostream& str, const MeshEntity< MeshConfig, Device, EntityTopology >& entity )
{
   entity.print( str );
   return str;
}

} // namespace Meshes
} // namespace TNL
