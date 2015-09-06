/***************************************************************************
                          tnlMesh_impl.h  -  description
                             -------------------
    begin                : Sep 5, 2015
    copyright            : (C) 2015 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/


#ifndef TNLMESH_IMPL_H
#define	TNLMESH_IMPL_H

#include "tnlMesh.h"


template< typename MeshConfig >
tnlString
tnlMesh< MeshConfig >::
getType()
{
   return tnlString( "tnlMesh< ") + MeshConfig::getType() + " >";
}

template< typename MeshConfig >
tnlString
tnlMesh< MeshConfig >::
getTypeVirtual() const
{
   return this->getType();
}

template< typename MeshConfig >   
constexpr int
tnlMesh< MeshConfig >::
getDimensions()
{
   return dimensions;
}

template< typename MeshConfig >   
   template< int Dimensions >
bool
tnlMesh< MeshConfig >::
entitiesAvalable() const
{
   return MeshTraits::template EntityTraits< Dimensions >::available;
}

template< typename MeshConfig >   
   template< int Dimensions >
typename tnlMesh< MeshConfig >::GlobalIndexType 
tnlMesh< MeshConfig >::
getNumberOfEntities() const
{
   return entitiesStorage.getNumberOfEntities( tnlDimensionsTag< Dimensions >() );
}

template< typename MeshConfig >   
typename tnlMesh< MeshConfig >::GlobalIndexType
tnlMesh< MeshConfig >::
getNumberOfCells() const
{
   return entitiesStorage.getNumberOfEntities( tnlDimensionsTag< dimensions >() );
}

template< typename MeshConfig >   
typename tnlMesh< MeshConfig >::CellType&
tnlMesh< MeshConfig >::
getCell( const GlobalIndexType cellIndex )
{
   return entitiesStorage.getEntity( tnlDimensionsTag< dimensions >(), cellIndex );
}

template< typename MeshConfig >   
const typename tnlMesh< MeshConfig >::CellType&
tnlMesh< MeshConfig >::
getCell( const GlobalIndexType cellIndex ) const
{
   return entitiesStorage.getEntity( tnlDimensionsTag< dimensions >(), cellIndex );
}

template< typename MeshConfig >
   template< int Dimensions >
typename tnlMesh< MeshConfig >::template EntityType< Dimensions >&
tnlMesh< MeshConfig >::
getEntity( const GlobalIndexType entityIndex )
{
   return entitiesStorage.getEntity( tnlDimensionsTag< Dimensions >(), entityIndex );
}

template< typename MeshConfig >
   template< int Dimensions >
const typename tnlMesh< MeshConfig >::template EntityType< Dimensions >&
tnlMesh< MeshConfig >::
getEntity( const GlobalIndexType entityIndex ) const
{
   return entitiesStorage.getEntity( tnlDimensionsTag< Dimensions >(), entityIndex );
}
   
template< typename MeshConfig >
bool
tnlMesh< MeshConfig >::
save( tnlFile& file ) const
{
   if( ! tnlObject::save( file ) ||
       ! entitiesStorage.save( file ) )
   {
      cerr << "Mesh saving failed." << endl;
      return false;
   }
   return true;
}

template< typename MeshConfig >
bool
tnlMesh< MeshConfig >::
load( tnlFile& file )
{
   if( ! tnlObject::load( file ) ||
       ! entitiesStorage.load( file ) )
   {
      cerr << "Mesh loading failed." << endl;
      return false;
   }
   return true;
}

template< typename MeshConfig >
void
tnlMesh< MeshConfig >::
print( ostream& str ) const
{
   entitiesStorage.print( str );
}

template< typename MeshConfig >
bool
tnlMesh< MeshConfig >::
operator==( const tnlMesh& mesh ) const
{
   return entitiesStorage.operator==( mesh.entitiesStorage );
}

template< typename MeshConfig >
   template< typename DimensionsTag >
typename tnlMesh< MeshConfig >::template EntityTraits< DimensionsTag::value >::StorageArrayType&
tnlMesh< MeshConfig >::
entitiesArray()
{
   return entitiesStorage.entitiesArray( DimensionsTag() ); 
}

template< typename MeshConfig >
   template< typename DimensionsTag, typename SuperDimensionsTag >
typename tnlMesh< MeshConfig >::MeshTraits::GlobalIdArrayType& 
tnlMesh< MeshConfig >::
superentityIdsArray()
{
   return entitiesStorage.template superentityIdsArray< SuperDimensionsTag >( DimensionsTag() ); 
}

template< typename MeshConfig >
bool 
tnlMesh< MeshConfig >::
init( const typename tnlMesh< MeshConfig >::MeshTraits::PointArrayType& points,
      const typename tnlMesh< MeshConfig >::MeshTraits::CellSeedArrayType& cellSeeds )
{
   tnlMeshInitializer< MeshConfig> meshInitializer;
   return meshInitializer.createMesh( points, cellSeeds, *this );
}




#endif	/* TNLMESH_IMPL_H */

