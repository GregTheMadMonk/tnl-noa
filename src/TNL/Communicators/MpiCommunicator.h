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
#include <fstream>
#include <cstring>

#ifdef HAVE_MPI
#include <mpi.h>
#ifdef OMPI_MAJOR_VERSION
   // header specific to OpenMPI (needed for CUDA-aware detection)
   #include <mpi-ext.h>
#endif

#include <unistd.h>  // getpid

#ifdef HAVE_CUDA
    #include <TNL/Cuda/CheckDevice.h>

    typedef struct __attribute__((__packed__))  {
       char name[MPI_MAX_PROCESSOR_NAME];
    } procName;
#endif

#endif

#include <TNL/String.h>
#include <TNL/Logger.h>
#include <TNL/Debugging/OutputRedirection.h>
#include <TNL/Communicators/MpiDefs.h>
#include <TNL/Config/ConfigDescription.h>
#include <TNL/Exceptions/MPISupportMissing.h>
#include <TNL/Exceptions/MPIDimsCreateError.h>
#include <TNL/Communicators/MPITypeResolver.h>


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
            if( redirect )
               setupRedirection();
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

      static void Init(int& argc, char**& argv )
      {
#ifdef HAVE_MPI
         MPI_Init( &argc, &argv );
         selectGPU();
#endif

         // silence warnings about (potentially) unused variables
         (void) NullGroup;
         (void) NullRequest;
      }

      static void setupRedirection()
      {
#ifdef HAVE_MPI
         if(isDistributed() )
         {
            if(GetRank(AllGroup)!=0)
            {
               const std::string stdoutFile = std::string("./stdout_") + std::to_string(GetRank(AllGroup)) + ".txt";
               const std::string stderrFile = std::string("./stderr_") + std::to_string(GetRank(AllGroup)) + ".txt";
               std::cout << GetRank(AllGroup) << ": Redirecting stdout and stderr to files " << stdoutFile << " and " << stderrFile << std::endl;
               Debugging::redirect_stdout_stderr( stdoutFile, stderrFile );
            }
         }
#endif
      }

      static void Finalize()
      {
#ifdef HAVE_MPI
         if(isDistributed())
         {
            if(GetRank(AllGroup)!=0)
            {
               // restore redirection (not necessary, it uses RAII internally...)
               Debugging::redirect_stdout_stderr( "", "", true );
            }
         }
         MPI_Finalize();
#endif
      }

      static bool IsInitialized()
      {
#ifdef HAVE_MPI
         int initialized, finalized;
         MPI_Initialized(&initialized);
         MPI_Finalized(&finalized);
         return initialized && !finalized;
#else
         return true;
#endif
      }

      static int GetRank(CommunicationGroup group = AllGroup )
      {
#ifdef HAVE_MPI
         TNL_ASSERT_TRUE(IsInitialized(), "Fatal Error - MPI communicator is not initialized");
         TNL_ASSERT_NE(group, NullGroup, "GetRank cannot be called with NullGroup");
         int rank;
         MPI_Comm_rank(group,&rank);
         return rank;
#else
         return 0;
#endif
      }

      static int GetSize(CommunicationGroup group = AllGroup )
      {
#ifdef HAVE_MPI
         TNL_ASSERT_TRUE(IsInitialized(), "Fatal Error - MPI communicator is not initialized");
         TNL_ASSERT_NE(group, NullGroup, "GetSize cannot be called with NullGroup");
         int size;
         MPI_Comm_size(group,&size);
         return size;
#else
         return 1;
#endif
      }

#ifdef HAVE_MPI
      template< typename T >
      static MPI_Datatype getDataType( const T& t )
      {
         return MPITypeResolver< T >::getType();
      }
#endif

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
#ifdef HAVE_MPI
         TNL_ASSERT_TRUE(IsInitialized(), "Fatal Error - MPI communicator is not initialized");
         TNL_ASSERT_NE(group, NullGroup, "Barrier cannot be called with NullGroup");
         MPI_Barrier(group);
#endif
      }

      template <typename T>
      static void Send( const T* data, int count, int dest, int tag, CommunicationGroup group = AllGroup )
      {
#ifdef HAVE_MPI
         TNL_ASSERT_TRUE(IsInitialized(), "Fatal Error - MPI communicator is not initialized");
         TNL_ASSERT_NE(group, NullGroup, "Send cannot be called with NullGroup");
         MPI_Send( const_cast< void* >( ( const void* ) data ), count, MPITypeResolver< T >::getType(), dest, tag, group );
#endif
      }

      template <typename T>
      static void Recv( T* data, int count, int src, int tag, CommunicationGroup group = AllGroup )
      {
#ifdef HAVE_MPI
         TNL_ASSERT_TRUE(IsInitialized(), "Fatal Error - MPI communicator is not initialized");
         TNL_ASSERT_NE(group, NullGroup, "Recv cannot be called with NullGroup");
         MPI_Status status;
         MPI_Recv( const_cast< void* >( ( const void* ) data ), count, MPITypeResolver< T >::getType() , src, tag, group, &status );
#endif
     }

      template <typename T>
      static Request ISend( const T* data, int count, int dest, int tag, CommunicationGroup group = AllGroup )
      {
#ifdef HAVE_MPI
         TNL_ASSERT_TRUE(IsInitialized(), "Fatal Error - MPI communicator is not initialized");
         TNL_ASSERT_NE(group, NullGroup, "ISend cannot be called with NullGroup");
         Request req;
         MPI_Isend( const_cast< void* >( ( const void* ) data ), count, MPITypeResolver< T >::getType(), dest, tag, group, &req);
         return req;
#else
         return 1;
#endif
      }

      template <typename T>
      static Request IRecv( T* data, int count, int src, int tag, CommunicationGroup group = AllGroup )
      {
#ifdef HAVE_MPI
         TNL_ASSERT_TRUE(IsInitialized(), "Fatal Error - MPI communicator is not initialized");
         TNL_ASSERT_NE(group, NullGroup, "IRecv cannot be called with NullGroup");
         Request req;
         MPI_Irecv( const_cast< void* >( ( const void* ) data ), count, MPITypeResolver< T >::getType() , src, tag, group, &req);
         return req;
#else
         return 1;
#endif
      }

      static void WaitAll(Request *reqs, int length)
      {
#ifdef HAVE_MPI
         TNL_ASSERT_TRUE(IsInitialized(), "Fatal Error - MPI communicator is not initialized");
         MPI_Waitall(length, reqs, MPI_STATUSES_IGNORE);
#endif
      }

      template< typename T >
      static void Bcast( T* data, int count, int root, CommunicationGroup group)
      {
#ifdef HAVE_MPI
         TNL_ASSERT_TRUE(IsInitialized(), "Fatal Error - MPI communicator is not initialized");
         TNL_ASSERT_NE(group, NullGroup, "BCast cannot be called with NullGroup");
         MPI_Bcast((void*) data, count, MPITypeResolver< T >::getType(), root, group);
#endif
      }

      template< typename T >
      static void Allreduce( const T* data,
                             T* reduced_data,
                             int count,
                             const MPI_Op &op,
                             CommunicationGroup group)
      {
#ifdef HAVE_MPI
         TNL_ASSERT_NE(group, NullGroup, "Allreduce cannot be called with NullGroup");
         MPI_Allreduce( const_cast< void* >( ( void* ) data ), (void*) reduced_data,count,MPITypeResolver< T >::getType(),op,group);
#else
         memcpy( ( void* ) reduced_data, ( const void* ) data, count * sizeof( T ) );
#endif
      }

      // in-place variant of Allreduce
      template< typename T >
      static void Allreduce( T* data,
                             int count,
                             const MPI_Op &op,
                             CommunicationGroup group)
      {
#ifdef HAVE_MPI
         TNL_ASSERT_NE(group, NullGroup, "Allreduce cannot be called with NullGroup");
         MPI_Allreduce( MPI_IN_PLACE, (void*) data,count,MPITypeResolver< T >::getType(),op,group);
#endif
      }


      template< typename T >
      static void Reduce( const T* data,
                          T* reduced_data,
                          int count,
                          MPI_Op &op,
                          int root,
                          CommunicationGroup group)
      {
#ifdef HAVE_MPI
         TNL_ASSERT_NE(group, NullGroup, "Reduce cannot be called with NullGroup");
         MPI_Reduce( const_cast< void* >( ( void*) data ), (void*) reduced_data,count,MPITypeResolver< T >::getType(),op,root,group);
#else
         memcpy( ( void* ) reduced_data, ( void* ) data, count * sizeof( T ) );
#endif
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
#ifdef HAVE_MPI
         TNL_ASSERT_NE(group, NullGroup, "SendReceive cannot be called with NullGroup");
         MPI_Status status;
         MPI_Sendrecv( const_cast< void* >( ( void* ) sendData ),
                       sendCount,
                       MPITypeResolver< T >::getType(),
                       destination,
                       sendTag,
                       ( void* ) receiveData,
                       receiveCount,
                       MPITypeResolver< T >::getType(),
                       source,
                       receiveTag,
                       group,
                       &status );
#else
         throw Exceptions::MPISupportMissing();
#endif
      }

      template< typename T >
      static void Alltoall( const T* sendData,
                            int sendCount,
                            T* receiveData,
                            int receiveCount,
                            CommunicationGroup group )
      {
#ifdef HAVE_MPI
         TNL_ASSERT_NE(group, NullGroup, "SendReceive cannot be called with NullGroup");
         MPI_Alltoall( const_cast< void* >( ( void* ) sendData ),
                       sendCount,
                       MPITypeResolver< T >::getType(),
                       ( void* ) receiveData,
                       receiveCount,
                       MPITypeResolver< T >::getType(),
                       group );
#else
         TNL_ASSERT_EQ( sendCount, receiveCount, "sendCount must be equal to receiveCount when running without MPI." );
         memcpy( (void*) receiveData, (const void*) sendData, sendCount * sizeof( T ) );
#endif
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
      static MPI_Request NullRequest;
      static MPI_Comm AllGroup;
      static MPI_Comm NullGroup;
#else
      static constexpr int NullRequest = -1;
      static constexpr int AllGroup = 1;
      static constexpr int NullGroup = 0;
#endif
   private:

      static void selectGPU(void)
      {
#ifdef HAVE_MPI
    #ifdef HAVE_CUDA
         const int count = GetSize(AllGroup);
         const int rank = GetRank(AllGroup);
         int gpuCount;
         cudaGetDeviceCount(&gpuCount);

         procName names[count];

         int i=0;
         int len;
         MPI_Get_processor_name(names[rank].name, &len);

         for(i=0;i<count;i++)
            std::memcpy(names[i].name,names[rank].name,len+1);

         MPI_Alltoall( (void*)names ,MPI_MAX_PROCESSOR_NAME,MPI_CHAR,
            (void*)names,MPI_MAX_PROCESSOR_NAME,MPI_CHAR,
                     MPI_COMM_WORLD);

         int nodeRank=0;
         for(i=0;i<rank;i++)
         {
            if(std::strcmp(names[rank].name,names[i].name)==0)
               nodeRank++;
         }

         const int gpuNumber = nodeRank % gpuCount;

         cudaSetDevice(gpuNumber);
         TNL_CHECK_CUDA_DEVICE;

         //std::cout<<"Node: " << rank << " gpu: " << gpuNumber << std::endl;
    #endif
#endif
      }
};

#ifdef HAVE_MPI
MPI_Request MpiCommunicator::NullRequest = MPI_REQUEST_NULL;
MPI_Comm MpiCommunicator::AllGroup = MPI_COMM_WORLD;
MPI_Comm MpiCommunicator::NullGroup = MPI_COMM_NULL;
#endif

} // namespace <unnamed>
} // namespace Communicators
} // namespace TNL
