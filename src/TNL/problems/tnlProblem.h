/***************************************************************************
                          tnlProblem.h  -  description
                             -------------------
    begin                : Jan 10, 2015
    copyright            : (C) 2015 by oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/matrices/tnlSlicedEllpackMatrix.h>

namespace TNL {

template< typename Real,
          typename Device,
          typename Index >
class tnlProblem
{
   public:

      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;
};

} //namespace TNL
