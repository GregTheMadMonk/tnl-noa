/***************************************************************************
                          tnlFiniteDifferences.h  -  description
                             -------------------
    begin                : Jan 10, 2016
    copyright            : (C) 2016 by Tomas Oberhuber
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

#ifndef TNLFINITEDIFFERENCES_H
#define	TNLFINITEDIFFERENCES_H

template< typename Mesh,
          typename Real,
          typename Index,
          int XDifference,
          int YDifference,
          int ZDifference,
          int XDirection,
          int YDirection,
          int ZDirection >
class tnlFiniteDifferences
{   
};

#include <operators/fdm/tnlFiniteDifferences_1D.h>
#include <operators/fdm/tnlFiniteDifferences_2D.h>
#include <operators/fdm/tnlFiniteDifferences_3D.h>

#endif	/* TNLFINITEDIFFERENCES_H */

