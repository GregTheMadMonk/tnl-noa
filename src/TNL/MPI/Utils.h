/***************************************************************************
                          MPI/Wrappers.h  -  description
                             -------------------
    begin                : Apr 23, 2005
    copyright            : (C) 2005 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Debugging/OutputRedirection.h>

#include "Wrappers.h"

namespace TNL {
namespace MPI {

inline bool isInitialized()
{
   return Initialized() && ! Finalized();
}

inline void setupRedirection( std::string outputDirectory )
{
#ifdef HAVE_MPI
   if( GetSize() > 1 && GetRank() != 0 ) {
      const std::string stdoutFile = outputDirectory + "/stdout_" + std::to_string(GetRank()) + ".txt";
      const std::string stderrFile = outputDirectory + "/stderr_" + std::to_string(GetRank()) + ".txt";
      std::cout << GetRank() << ": Redirecting stdout and stderr to files " << stdoutFile << " and " << stderrFile << std::endl;
      Debugging::redirect_stdout_stderr( stdoutFile, stderrFile );
   }
#endif
}

// restore redirection (usually not necessary, it uses RAII internally...)
inline void restoreRedirection()
{
   if( GetSize() > 1 && GetRank() != 0 ) {
      Debugging::redirect_stdout_stderr( "", "", true );
   }
}

} // namespace MPI
} // namespace TNL
