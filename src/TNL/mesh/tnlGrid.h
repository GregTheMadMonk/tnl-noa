/***************************************************************************
                          tnlGrid.h  -  description
                             -------------------
    begin                : Jan 16, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/tnlObject.h>
#include <TNL/core/tnlHost.h>
#include <TNL/core/vectors/tnlStaticVector.h>
#include <TNL/core/vectors/tnlVector.h>

namespace TNL {

template< int Dimensions,
          typename Real = double,
          typename Device = tnlHost,
          typename Index = int >
class tnlGrid : public tnlObject
{
};

} // namespace TNL

#include <TNL/mesh/grids/tnlGrid1D.h>
#include <TNL/mesh/grids/tnlGrid2D.h>
#include <TNL/mesh/grids/tnlGrid3D.h>
