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


#endif	/* TNLMESH_IMPL_H */

