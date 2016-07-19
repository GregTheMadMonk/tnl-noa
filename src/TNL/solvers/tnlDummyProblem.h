/***************************************************************************
                          tnlDummyProblem.h  -  description
                             -------------------
    begin                : Jul 10, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/core/tnlHost.h>
#include <TNL/core/vectors/tnlVector.h>
#include <TNL/mesh/tnlGrid.h>

namespace TNL {

template< typename Real = double,
          typename Device = tnlHost,
          typename Index = int >
class tnlDummyProblem
{
   public:

      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;
      typedef tnlVector< Real, Device, Index > DofVectorType;
      typedef tnlGrid< 1, Real, Device, Index > MeshType;
      typedef DofVectorType MeshDependentDataType;
};

} // namespace TNL
