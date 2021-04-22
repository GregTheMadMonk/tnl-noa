/***************************************************************************
                          MPI/Utils.h  -  description
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

/**
 * \brief Returns a local rank ID of the current process within a group of
 * processes running on a shared-memory node.
 *
 * The given MPI communicator is split into groups according to the
 * `MPI_COMM_TYPE_SHARED` type (from MPI-3) and the rank ID of the process
 * within the group is returned.
 */
inline int getRankOnNode( MPI_Comm group = AllGroup() )
{
#ifdef HAVE_MPI
   const int rank = GetRank(group);

   MPI_Info info;
   MPI_Info_create( &info );

   MPI_Comm local_comm;
   MPI_Comm_split_type( group, MPI_COMM_TYPE_SHARED, rank, info, &local_comm );

   const int local_rank = GetRank( local_comm );

   MPI_Comm_free(&local_comm);
   MPI_Info_free(&info);

   return local_rank;
#else
   return 0;
#endif
}

/**
 * \brief Applies the given reduction operation to the values among all ranks
 * in the given communicator.
 *
 * This is a collective operation which uses \ref Allreduce internally. It
 * provides nicer semantics than the wrapper function: the input value is passed
 * by value (heh) rather then by pointer, and the result is returned rather than
 * written to the output pointer.
 *
 * \param value Value of the current rank to be reduced.
 * \param op The reduction operation to be applied.
 * \param group The communicator comprising ranks that participate in the
 *              collective operation.
 * \return The reduced value (it is ensured that all ranks receive the same
 *         value).
 */
template< typename T >
T reduce( T value, const MPI_Op& op, MPI_Comm group = AllGroup() )
{
   // call the in-place variant of Allreduce
   Allreduce( &value, 1, op, group );
   // return the reduced value
   return value;
}

} // namespace MPI
} // namespace TNL
