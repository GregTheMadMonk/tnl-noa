/***************************************************************************
                          Grid.h  -  description
                             -------------------
    begin                : Jan 16, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Object.h>
#include <TNL/Devices/Host.h>

namespace TNL {
namespace Meshes {

template< int Dimension,
          typename Real = double,
          typename Device = Devices::Host,
          typename Index = int >
class Grid : public Object
{
};

} // namespace Meshes
} // namespace TNL

#include <TNL/Meshes/GridDetails/Grid1D.h>
#include <TNL/Meshes/GridDetails/Grid2D.h>
#include <TNL/Meshes/GridDetails/Grid3D.h>
