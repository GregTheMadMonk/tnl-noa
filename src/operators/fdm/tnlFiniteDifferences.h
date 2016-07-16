/***************************************************************************
                          tnlFiniteDifferences.h  -  description
                             -------------------
    begin                : Jan 10, 2016
    copyright            : (C) 2016 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

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

