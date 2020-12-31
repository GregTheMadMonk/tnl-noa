/***************************************************************************
                          MpiCommunicator.h  -  description
                             -------------------
    begin                : Apr 23, 2005
    copyright            : (C) 2005 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/MPI/Wrappers.h>
#include <TNL/MPI/DummyDefs.h>
#include <TNL/MPI/Utils.h>
#include <TNL/MPI/Config.h>
#include <TNL/Exceptions/MPIDimsCreateError.h>

namespace TNL {
//! \brief Namespace for TNL communicators.
namespace Communicators {
namespace {

//! \brief MPI communicator.
class MpiCommunicator
{
   public:
#ifdef HAVE_MPI
      using Request = MPI_Request;
      using CommunicationGroup = MPI_Comm;
#else
      using Request = int;
      using CommunicationGroup = int;
#endif

      static bool isDistributed()
      {
#ifdef HAVE_MPI
         return GetSize(AllGroup)>1;
#else
         return false;
#endif
      }

      static void configSetup( Config::ConfigDescription& config, const String& prefix = "" )
      {
         MPI::configSetup( config, prefix );
      }

      static bool setup( const Config::ParameterContainer& parameters,
                         const String& prefix = "" )
      {
         return MPI::setup( parameters, prefix );
      }

      static void Init( int& argc, char**& argv, int required_thread_level = MPI_THREAD_SINGLE )
      {
         MPI::Init( argc, argv, required_thread_level );

         // silence warnings about (potentially) unused variables
         (void) NullGroup;
      }

      static void Finalize()
      {
         MPI::Finalize();
      }

      static bool IsInitialized()
      {
         return MPI::isInitialized();
      }

      static int GetRank(CommunicationGroup group = AllGroup )
      {
         return MPI::GetRank( group );
      }

      static int GetSize(CommunicationGroup group = AllGroup )
      {
         return MPI::GetSize( group );
      }

      static void Barrier( CommunicationGroup group = AllGroup )
      {
         MPI::Barrier( group );
      }

      template <typename T>
      static void Send( const T* data, int count, int dest, int tag, CommunicationGroup group = AllGroup )
      {
         MPI::Send( data, count, dest, tag, group );
      }

      template <typename T>
      static void Recv( T* data, int count, int src, int tag, CommunicationGroup group = AllGroup )
      {
         MPI::Recv( data, count, src, tag, group );
      }

      template <typename T>
      static Request ISend( const T* data, int count, int dest, int tag, CommunicationGroup group = AllGroup )
      {
         return MPI::Isend( data, count, dest, tag, group );
      }

      template <typename T>
      static Request IRecv( T* data, int count, int src, int tag, CommunicationGroup group = AllGroup )
      {
         return MPI::Irecv( data, count, src, tag, group );
      }

      static void WaitAll(Request *reqs, int length)
      {
         MPI::Waitall( reqs, length );
      }

      template< typename T >
      static void Bcast( T* data, int count, int root, CommunicationGroup group)
      {
         MPI::Bcast( data, count, root, group );
      }

      template< typename T >
      static void Allreduce( const T* data,
                             T* reduced_data,
                             int count,
                             const MPI_Op &op,
                             CommunicationGroup group)
      {
         MPI::Allreduce( data, reduced_data, count, op, group );
      }

      // in-place variant of Allreduce
      template< typename T >
      static void Allreduce( T* data,
                             int count,
                             const MPI_Op &op,
                             CommunicationGroup group)
      {
         MPI::Allreduce( data, count, op, group );
      }

      template< typename T >
      static void Reduce( const T* data,
                          T* reduced_data,
                          int count,
                          const MPI_Op &op,
                          int root,
                          CommunicationGroup group)
      {
         MPI::Reduce( data, reduced_data, count, op, root, group );
      }

      template< typename T >
      static void SendReceive( const T* sendData,
                               int sendCount,
                               int destination,
                               int sendTag,
                               T* receiveData,
                               int receiveCount,
                               int source,
                               int receiveTag,
                               CommunicationGroup group )
      {
         MPI::Sendrecv( sendData, sendCount, destination, sendTag, receiveData, receiveCount, source, receiveTag, group );
      }

      template< typename T >
      static void Alltoall( const T* sendData,
                            int sendCount,
                            T* receiveData,
                            int receiveCount,
                            CommunicationGroup group )
      {
         MPI::Alltoall( sendData, sendCount, receiveData, receiveCount, group );
      }


      //dim-number of dimensions, distr array of guess distr - 0 for computation
      //distr array will be filled by computed distribution
      //more information in MPI documentation
      static void DimsCreate(int nproc, int dim, int *distr)
      {
#ifdef HAVE_MPI
         int sum = 0, prod = 1;
         for( int i = 0;i < dim; i++ ) {
            sum += distr[ i ];
            prod *= distr[ i ];
         }
         if( prod != 0 && prod != GetSize( AllGroup ) )
            throw Exceptions::MPIDimsCreateError();
         if(sum==0) {
            for(int i=0;i<dim-1;i++)
               distr[i]=1;
            distr[dim-1]=0;
         }

         MPI_Dims_create(nproc, dim, distr);
#else
         for(int i=0;i<dim;i++)
            distr[i]=1;
#endif
      }

      static void CreateNewGroup( bool meToo, int myRank, CommunicationGroup &oldGroup, CommunicationGroup &newGroup )
      {
#ifdef HAVE_MPI
         if(meToo)
            MPI_Comm_split(oldGroup, 1, myRank, &newGroup);
         else
            MPI_Comm_split(oldGroup, MPI_UNDEFINED, GetRank(oldGroup), &newGroup);
#else
         newGroup=oldGroup;
#endif
      }

#ifdef HAVE_MPI
      static MPI_Comm AllGroup;
      static MPI_Comm NullGroup;
#else
      static constexpr int AllGroup = 1;
      static constexpr int NullGroup = 0;
#endif
   private:
};

#ifdef HAVE_MPI
MPI_Comm MpiCommunicator::AllGroup = MPI_COMM_WORLD;
MPI_Comm MpiCommunicator::NullGroup = MPI_COMM_NULL;
#endif

} // namespace <unnamed>
} // namespace Communicators
} // namespace TNL
