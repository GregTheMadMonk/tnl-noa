/***************************************************************************
                          tnlMPIMesh.h  -  description
                             -------------------
    begin                : Dec 15, 2010
    copyright            : (C) 2010 by Tomas Oberhuber
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

#ifndef TNLMPIMESH_H_
#define TNLMPIMESH_H_

#include <mesh/tnlGrid.h>

template< int Dimensions, typename Real = double, typename Device = tnlHost, typename Index = int >
class tnlMPIMesh
{
};

#include <diff/tnlMPIMesh2D.h>
#include <diff/tnlMPIMesh3D.h>

#endif /* TNLMPIMESH_H_ */
