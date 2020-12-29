/***************************************************************************
                          MpiCommunicator.h  -  description
                             -------------------
    begin                : Apr 23, 2005
    copyright            : (C) 2005 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <iostream>

#ifdef HAVE_MPI
#ifdef OMPI_MAJOR_VERSION
   // header specific to OpenMPI (needed for CUDA-aware detection)
   #include <mpi-ext.h>
#endif

#include <unistd.h>  // getpid
#endif

#include <TNL/String.h>
#include <TNL/Logger.h>
#include <TNL/MPI/Wrappers.h>
#include <TNL/MPI/DummyDefs.h>
#include <TNL/MPI/Utils.h>
#include <TNL/Config/ConfigDescription.h>
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
#ifdef HAVE_MPI
         config.addEntry< bool >( "redirect-mpi-output", "Only process with rank 0 prints to console. Other processes are redirected to files.", true );
         config.addEntry< String >( "redirect-mpi-output-dir", "Directory where ranks will store the files if their output is redirected.", "." );
         config.addEntry< bool >( "mpi-gdb-debug", "Wait for GDB to attach the master MPI process.", false );
         config.addEntry< int >( "mpi-process-to-attach", "Number of the MPI process to be attached by GDB. Set -1 for all processes.", 0 );
#endif
      }

      static bool setup( const Config::ParameterContainer& parameters,
                         const String& prefix = "" )
      {
#ifdef HAVE_MPI
         if(IsInitialized())//i.e. - isUsed
         {
            const bool redirect = parameters.getParameter< bool >( "redirect-mpi-output" );
            const String outputDirectory = parameters.getParameter< String >( "redirect-mpi-output-dir" );
            if( redirect )
               MPI::setupRedirection( outputDirectory );
#ifdef HAVE_CUDA
            int size;
            MPI_Comm_size( MPI_COMM_WORLD, &size );
            if( size > 1 )
            {
   #if defined( MPIX_CUDA_AWARE_SUPPORT ) && MPIX_CUDA_AWARE_SUPPORT
               std::cout << "CUDA-aware MPI detected on this system ... " << std::endl;
   #elif defined( MPIX_CUDA_AWARE_SUPPORT ) && !MPIX_CUDA_AWARE_SUPPORT
               std::cerr << "MPI is not CUDA-aware. Please install correct version of MPI." << std::endl;
               return false;
   #else
               std::cerr << "WARNING: TNL cannot detect if you have CUDA-aware MPI. Some problems may occur." << std::endl;
   #endif
            }
#endif // HAVE_CUDA
            bool gdbDebug = parameters.getParameter< bool >( "mpi-gdb-debug" );
            int processToAttach = parameters.getParameter< int >( "mpi-process-to-attach" );

            if( gdbDebug )
            {
               int rank = GetRank( MPI_COMM_WORLD );
               int pid = getpid();

               volatile int tnlMPIDebugAttached = 0;
               MPI_Send( &pid, 1, MPI_INT, 0, 0, MPI_COMM_WORLD );
               MPI_Barrier( MPI_COMM_WORLD );
               if( rank == 0 )
               {
                  std::cout << "Attach GDB to MPI process(es) by entering:" << std::endl;
                  for( int i = 0; i < GetSize( MPI_COMM_WORLD ); i++ )
                  {
                     MPI_Status status;
                     int recvPid;
                     MPI_Recv( &recvPid, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &status );

                     if( i == processToAttach || processToAttach == -1 )
                     {
                        std::cout << "  For MPI process " << i << ": gdb -q -ex \"attach " << recvPid << "\""
                                  << " -ex \"set variable tnlMPIDebugAttached=1\""
                                  << " -ex \"continue\"" << std::endl;
                     }
                  }
                  std::cout << std::flush;
               }
               if( rank == processToAttach || processToAttach == -1 )
                  while( ! tnlMPIDebugAttached );
               MPI_Barrier( MPI_COMM_WORLD );
            }
         }
#endif // HAVE_MPI
         return true;
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


      static void writeProlog( Logger& logger )
      {
         if( isDistributed() )
         {
            logger.writeParameter( "MPI processes:", GetSize(AllGroup) );
         }
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
