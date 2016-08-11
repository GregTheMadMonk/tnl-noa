/***************************************************************************
                          FiniteDifferences.h  -  description
                             -------------------
    begin                : Jan 10, 2016
    copyright            : (C) 2016 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {
namespace Operators {   

template< typename Mesh,
          typename Real,
          typename Index,
          int XDifference,
          int YDifference,
          int ZDifference,
          int XDirection,
          int YDirection,
          int ZDirection >
class FiniteDifferences
{
};

} // namespace Operators
} // namespace TNL

#include <TNL/Operators/fdm/FiniteDifferences_1D.h>
#include <TNL/Operators/fdm/FiniteDifferences_2D.h>
#include <TNL/Operators/fdm/FiniteDifferences_3D.h>

