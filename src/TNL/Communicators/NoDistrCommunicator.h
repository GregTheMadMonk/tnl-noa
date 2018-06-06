/***************************************************************************
                          NoDistrCommunicator.h  -  description
                             -------------------
    begin                : 2018/01/09
    copyright            : (C) 2018 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Logger.h>

namespace TNL {
namespace Communicators {
        
class NoDistrCommunicator
{


   public:

      typedef int Request;
      static Request NullRequest;

      static void configSetup( Config::ConfigDescription& config, const String& prefix = "" ){};
 
      static bool setup( const Config::ParameterContainer& parameters,
                         const String& prefix = "" )
      {
         return true;
      }
      
      static void Init(int argc, char **argv, bool redirect=false)
      {
          NullRequest=-1;
      };

      static void Finalize()
      {
      };

      static bool IsInitialized()
      {   
          return true;
      };

      static bool isDistributed()
      {
          return false;
      };

      static int GetRank()
      {
          return 0;
      };

      static int GetSize()
      {
          return 1;
      };

      static void DimsCreate(int nproc, int dim, int *distr)
      {
          for(int i=0;i<dim;i++)
          {
              distr[i]=1;
          }
      };

      static void Barrier()
      {
      };

      template <typename T>
      static Request ISend( const T *data, int count, int dest)
      {
          return 1;
      }

      template <typename T>
      static Request IRecv( const T *data, int count, int src)
      {
          return 1;
      }

      static void WaitAll(Request *reqs, int length)
      {
      };

      template< typename T > 
      static void Bcast(  T& data, int count, int root)
      {
      }

     /* template< typename T >
      static void Allreduce( T& data,
                   T& reduced_data,
                   int count,
                   const MPI_Op &op)
      {
              MPI::COMM_WORLD.Allreduce((void*) &data, (void*) &reduced_data,count,MPIDataType(data),op);
      };

      template< typename T >
      static void Reduce( T& data,
                  T& reduced_data,
                  int count,
                  MPI_Op &op,
                  int root)
      {
           MPI::COMM_WORLD.Reduce((void*) &data, (void*) &reduced_data,count,MPIDataType(data),op,root);
      };*/

      static void writeProlog( Logger& logger ){};
};


  int NoDistrCommunicator::NullRequest;

} // namespace Communicators
} // namespace TNL


