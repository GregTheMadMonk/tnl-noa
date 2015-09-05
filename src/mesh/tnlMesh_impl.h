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
typename tnlMesh< MeshConfig >::template EntityTraits< Dimensions >::GlobalIndexType 
tnlMesh< MeshConfig >::
getNumberOfEntities() const
{
   return entitiesStorage.getNumberOfEntities( tnlDimensionsTag< Dimensions >() );
}

template< typename MeshConfig >   
typename tnlMesh< MeshConfig >::template EntityTraits< tnlMesh< MeshConfig >::dimensions >::GlobalIndexType
tnlMesh< MeshConfig >::
getNumberOfCells() const
{
   return entitiesStorage.getNumberOfEntities( tnlDimensionsTag< dimensions >() );
}

template< typename MeshConfig >   
typename tnlMesh< MeshConfig >::CellType&
tnlMesh< MeshConfig >::
getCell( const typename MeshTraits::template EntityTraits< dimensions >::GlobalIndexType cellIndex )
{
   return entitiesStorage.getEntity( tnlDimensionsTag< dimensions >(), cellIndex );
}

template< typename MeshConfig >   
const typename tnlMesh< MeshConfig >::CellType&
tnlMesh< MeshConfig >::
getCell( const typename MeshTraits::template EntityTraits< dimensions >::GlobalIndexType cellIndex ) const
{
   return entitiesStorage.getEntity( tnlDimensionsTag< dimensions >(), cellIndex );
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


#endif	/* TNLMESH_IMPL_H */

