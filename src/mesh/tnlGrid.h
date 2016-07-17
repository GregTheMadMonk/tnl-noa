/***************************************************************************
                          tnlGrid.h  -  description
                             -------------------
    begin                : Jan 16, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once


#include <tnlObject.h>
#include <core/tnlHost.h>
#include <core/vectors/tnlStaticVector.h>
#include <core/vectors/tnlVector.h>

namespace TNL {

template< int Dimensions,
          typename Real = double,
          typename Device = tnlHost,
          typename Index = int >
class tnlGrid : public tnlObject
{
};

} // namespace TNL

#include <mesh/grids/tnlGrid1D.h>
#include <mesh/grids/tnlGrid2D.h>
#include <mesh/grids/tnlGrid3D.h>
