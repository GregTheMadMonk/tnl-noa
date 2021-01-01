/***************************************************************************
                          MPI/Profiling.h  -  description
                             -------------------
    begin                : Jan 1, 2021
    copyright            : (C) 2021 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Timer.h>

namespace TNL {
namespace MPI {

inline Timer& getTimerAllreduce()
{
   static Timer t;
   return t;
}

} // namespace MPI
} // namespace TNL
