/***************************************************************************
                          ScopedInitializer.h  -  description
                             -------------------
    begin                : Sep 16, 2018
    copyright            : (C) 2005 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovský

#pragma once

#include "Wrappers.h"
#include "Utils.h"

namespace TNL {
namespace MPI {

struct ScopedInitializer
{
   ScopedInitializer( int& argc, char**& argv, int required_thread_level = MPI_THREAD_SINGLE )
   {
      Init( argc, argv );
   }

   ~ScopedInitializer()
   {
      restoreRedirection();
      Finalize();
   }
};

} // namespace MPI
} // namespace TNL
