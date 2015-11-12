/***************************************************************************
                          tnlMeshFunction_impl.h  -  description
                             -------------------
    begin                : Nov 8, 2015
    copyright            : (C) 2015 by oberhuber
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

#include <core/tnlAssert.h>

#ifndef TNLMESHFUNCTION_IMPL_H
#define	TNLMESHFUNCTION_IMPL_H

template< typename Mesh,
          typename Vector,
          int MeshEntitiesDimensions >
tnlMeshFunction< Mesh, Vector, MeshEntitiesDimensions >::
tnlMeshFunction()
: mesh( 0 )
{
}
      
template< typename Mesh,
          typename Vector,
          int MeshEntitiesDimensions >
tnlMeshFunction< Mesh, Vector, MeshEntitiesDimensions >::
tnlMeshFunction( const MeshType* mesh )
{
   this->setMesh( mesh );
}

template< typename Mesh,
          typename Vector,
          int MeshEntitiesDimensions >
void 
tnlMeshFunction< Mesh, Vector, MeshEntitiesDimensions >::
setMesh( const MeshType* mesh )
{
   tnlAssert( mesh != NULL, );
   this->mesh = mesh;
   this->data.setSize( this->mesh->template getNumberOfEntities< MeshEntitiesDimensions >() );
}

template< typename Mesh,
          typename Vector,
          int MeshEntitiesDimensions >
const typename tnlMeshFunction< Mesh, Vector, MeshEntitiesDimensions >::MeshType& 
tnlMeshFunction< Mesh, Vector, MeshEntitiesDimensions >::
getMesh() const
{
   return * this->mesh;
}

template< typename Mesh,
          typename Vector,
          int MeshEntitiesDimensions >
const typename tnlMeshFunction< Mesh, Vector, MeshEntitiesDimensions >::VectorType& 
tnlMeshFunction< Mesh, Vector, MeshEntitiesDimensions >::
getData() const
{
   return this->data;
}

template< typename Mesh,
          typename Vector,
          int MeshEntitiesDimensions >
const typename tnlMeshFunction< Mesh, Vector, MeshEntitiesDimensions >::RealType& 
tnlMeshFunction< Mesh, Vector, MeshEntitiesDimensions >::
getValue( const IndexType meshEntityIndex )
{
   return this->data.getValue( meshEntityIndex );
}

template< typename Mesh,
          typename Vector,
          int MeshEntitiesDimensions >
void 
tnlMeshFunction< Mesh, Vector, MeshEntitiesDimensions >::
setValue( const IndexType meshEntityIndex,
          const RealType& value )
{
   this->data.setValue( meshEntityIndex, value );
}

template< typename Mesh,
          typename Vector,
          int MeshEntitiesDimensions >
typename tnlMeshFunction< Mesh, Vector, MeshEntitiesDimensions >::RealType& 
tnlMeshFunction< Mesh, Vector, MeshEntitiesDimensions >::
operator[]( const IndexType meshEntityIndex )
{
   return this->data[ meshEntityIndex ];
}

template< typename Mesh,
          typename Vector,
          int MeshEntitiesDimensions >
const typename tnlMeshFunction< Mesh, Vector, MeshEntitiesDimensions >::RealType& 
tnlMeshFunction< Mesh, Vector, MeshEntitiesDimensions >::
operator[]( const IndexType meshEntityIndex ) const
{
   return this->data[ meshEntityIndex ];
}

#endif	/* TNLMESHFUNCTION_IMPL_H */

