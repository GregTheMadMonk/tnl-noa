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
#include <TNL/Communicators/MpiDefs.h>

namespace TNL {
namespace Communicators {
namespace {

class NoDistrCommunicator
{


   public:

      typedef int Request;
      typedef int CommunicationGroup;
      static constexpr Request NullRequest = -1;
      static constexpr CommunicationGroup AllGroup = 1;
      static constexpr CommunicationGroup NullGroup = 0;

      static void configSetup( Config::ConfigDescription& config, const String& prefix = "" ){};
 
      static bool setup( const Config::ParameterContainer& parameters,
                         const String& prefix = "" )
      {
         return true;
      }
      
      static void Init(int& argc, char**& argv) {}
      
      static void setRedirection( bool redirect_ ) {}
      
      static void setupRedirection(){}

      static void Finalize(){}

      static bool IsInitialized()
      {   
          return true;
      }

      static bool isDistributed()
      {
          return false;
      }

      static int GetRank(CommunicationGroup group = AllGroup )
      {
          return 0;
      }

      static int GetSize(CommunicationGroup group = AllGroup )
      {
          return 1;
      }

      static void DimsCreate(int nproc, int dim, int *distr)
      {
          for(int i=0;i<dim;i++)
          {
              distr[i]=1;
          }
      }

      static void Barrier(CommunicationGroup group)
      {
      };

      template <typename T>
      static Request ISend( const T *data, int count, int dest, int tag, CommunicationGroup group)
      {
          return 1;
      }

      template <typename T>
      static Request IRecv( const T *data, int count, int src, int tag, CommunicationGroup group)
      {
          return 1;
      }

      static void WaitAll(Request *reqs, int length)
      {
      }

      template< typename T >
      static void Bcast( T* data, int count, int root, CommunicationGroup group)
      {
      }

      template< typename T >
      static void Allreduce( const T* data,
                             T* reduced_data,
                             int count,
                             const MPI_Op &op,
                             CommunicationGroup group )
      {
         memcpy( ( void* ) reduced_data, ( const void* ) data, count * sizeof( T ) );
      }

      // in-place variant of Allreduce
      template< typename T >
      static void Allreduce( T* data,
                             int count,
                             const MPI_Op &op,
                             CommunicationGroup group )
      {
      }

      template< typename T >
      static void Reduce( T* data,
                          T* reduced_data,
                          int count,
                          MPI_Op &op,
                          int root,
                          CommunicationGroup group )
      {
         memcpy( ( void* ) reduced_data, ( void* ) data, count * sizeof( T ) );
      }

      template< typename T >
      static void Alltoall( const T* sendData,
                            int sendCount,
                            T* receiveData,
                            int receiveCount,
                            CommunicationGroup group )
      {
      }

      static void CreateNewGroup(bool meToo, int myRank, CommunicationGroup &oldGroup, CommunicationGroup &newGroup)
      {
         newGroup=oldGroup;
      }

      static void writeProlog( Logger& logger ){};
};

} // namespace <unnamed>
} // namespace Communicators
} // namespace TNL
