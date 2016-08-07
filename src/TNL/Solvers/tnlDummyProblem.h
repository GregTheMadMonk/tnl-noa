/***************************************************************************
                          tnlDummyProblem.h  -  description
                             -------------------
    begin                : Jul 10, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Devices/Host.h>
#include <TNL/Vectors/Vector.h>
#include <TNL/mesh/tnlGrid.h>

namespace TNL {
namespace Solvers {   

template< typename Real = double,
          typename Device = Devices::Host,
          typename Index = int >
class tnlDummyProblem
{
   public:

      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;
      typedef Vectors::Vector< Real, Device, Index > DofVectorType;
      typedef tnlGrid< 1, Real, Device, Index > MeshType;
      typedef DofVectorType MeshDependentDataType;
};

} // namespace Solvers
} // namespace TNL
