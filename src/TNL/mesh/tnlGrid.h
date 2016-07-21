/***************************************************************************
                          tnlGrid.h  -  description
                             -------------------
    begin                : Jan 16, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Object.h>
#include <TNL/core/tnlHost.h>
#include <TNL/Vectors/StaticVector.h>
#include <TNL/Vectors/Vector.h>

namespace TNL {

template< int Dimensions,
          typename Real = double,
          typename Device = tnlHost,
          typename Index = int >
class tnlGrid : public Object
{
};

} // namespace TNL

#include <TNL/mesh/grids/tnlGrid1D.h>
#include <TNL/mesh/grids/tnlGrid2D.h>
#include <TNL/mesh/grids/tnlGrid3D.h>
