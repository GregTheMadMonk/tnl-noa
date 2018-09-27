/***************************************************************************
                          MpiCommunicator.h  -  description
                             -------------------
    begin                : 2005/04/23
    copyright            : (C) 2005 by Tomas Oberhuber
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

#ifdef HAVE_CUDA
    #include <TNL/Devices/Cuda.h>

    typedef struct __attribute__((__packed__))  {
       char name[MPI_MAX_PROCESSOR_NAME];
    } procName;
#endif

#endif

#include <TNL/String.h>
#include <TNL/Logger.h>
#include <TNL/Communicators/MpiDefs.h>
#include <TNL/Config/ConfigDescription.h>
#include <TNL/Exceptions/MPISupportMissing.h>



namespace TNL {
namespace Communicators {
namespace {

class MpiCommunicator
{

   public: // TODO: this was private
#ifdef HAVE_MPI
      inline static MPI_Datatype MPIDataType( const signed char* ) { return MPI_CHAR; };
      inline static MPI_Datatype MPIDataType( const signed short int* ) { return MPI_SHORT; };
      inline static MPI_Datatype MPIDataType( const signed int* ) { return MPI_INT; };
      inline static MPI_Datatype MPIDataType( const signed long int* ) { return MPI_LONG; };
      inline static MPI_Datatype MPIDataType( const unsigned char *) { return MPI_UNSIGNED_CHAR; };
      inline static MPI_Datatype MPIDataType( const unsigned short int* ) { return MPI_UNSIGNED_SHORT; };
      inline static MPI_Datatype MPIDataType( const unsigned int* ) { return MPI_UNSIGNED; };
      inline static MPI_Datatype MPIDataType( const unsigned long int* ) { return MPI_UNSIGNED_LONG; };
      inline static MPI_Datatype MPIDataType( const float* ) { return MPI_FLOAT; };
      inline static MPI_Datatype MPIDataType( const double* ) { return MPI_DOUBLE; };
      inline static MPI_Datatype MPIDataType( const long double* ) { return MPI_LONG_DOUBLE; };

      // TODO: tested with MPI_LOR and MPI_LAND, but there should probably be unit tests for all operations
      inline static MPI_Datatype MPIDataType( const bool* )
      {
         // sizeof(bool) is implementation-defined: https://stackoverflow.com/a/4897859
         static_assert( sizeof(bool) == 1, "The programmer did not count with systems where sizeof(bool) != 1." );
         return MPI_CHAR;
      };

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
         config.addEntry< int >( "mpi-process-to-attach", "Number of the MPI process to be attached by GDB.", 0 );
#endif
      }

      static bool setup( const Config::ParameterContainer& parameters,
                         const String& prefix = "" )
      {
#ifdef HAVE_MPI
         if(IsInitialized())//i.e. - isUsed
         {
            redirect = parameters.getParameter< bool >( "redirect-mpi-output" );
            setupRedirection();
#ifdef HAVE_CUDA
   #if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
            std::cout << "CUDA-aware MPI detected on this system ... " << std::endl;
   #elif defined(MPIX_CUDA_AWARE_SUPPORT) && !MPIX_CUDA_AWARE_SUPPORT
            std::cerr << "MPI is not CUDA-aware. Please install correct version of MPI." << std::endl;
            return false;
   #else
            std::cerr << "WARNING: TNL cannot detect if you have CUDA-aware MPI. Some problems may occur." << std::endl;
   #endif
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
                                  << " -ex \"finish\"" << std::endl;
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
      }

      static void setRedirection( bool redirect_ )
      {
         redirect = redirect_;
      }

      static void setupRedirection()
      {
#ifdef HAVE_MPI
         if(isDistributed() && redirect )
         {
            //redirect all stdout to files, only 0 take to go to console
            backup=std::cout.rdbuf();

            //redirect output to files...
            if(GetRank(AllGroup)!=0)
            {
               std::cout << GetRank(AllGroup) << ": Redirecting std::cout to file" << std::endl;
               const String stdoutFile = String("./stdout-") + convertToString(GetRank(AllGroup)) + String(".txt");
               filestr.open(stdoutFile.getString());
               psbuf = filestr.rdbuf();
               std::cout.rdbuf(psbuf);
            }
         }
#else
         throw Exceptions::MPISupportMissing();
#endif
      }

      static void Finalize()
      {
#ifdef HAVE_MPI
         if(isDistributed())
         {
            if(GetRank(AllGroup)!=0)
            {
               std::cout.rdbuf(backup);
               filestr.close();
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
         throw Exceptions::MPISupportMissing();
#endif
      }

      static int GetRank(CommunicationGroup group)
      {
#ifdef HAVE_MPI
        TNL_ASSERT_TRUE(IsInitialized(), "Fatal Error - MPI communicator is not initialized");
        TNL_ASSERT_NE(group, NullGroup, "GetRank cannot be called with NullGroup");
        int rank;
        MPI_Comm_rank(group,&rank);
        return rank;
#else
         throw Exceptions::MPISupportMissing();
#endif
      }

      static int GetSize(CommunicationGroup group)
      {
#ifdef HAVE_MPI
         TNL_ASSERT_TRUE(IsInitialized(), "Fatal Error - MPI communicator is not initialized");
         TNL_ASSERT_NE(group, NullGroup, "GetSize cannot be called with NullGroup");
         int size;
         MPI_Comm_size(group,&size);
         return size;
#else
         throw Exceptions::MPISupportMissing();
#endif
      }

        //dim-number of dimesions, distr array of guess distr - 0 for computation
        //distr array will be filled by computed distribution
        //more information in MPI documentation
        static void DimsCreate(int nproc, int dim, int *distr)
        {
#ifdef HAVE_MPI
            /***HACK for linear distribution***/
           int sum=0;
           for(int i=0;i<dim;i++)
                sum+=distr[i];
           if(sum==0)
           {
               for(int i=0;i<dim-1;i++)
               {
                    distr[i]=1;
               }
               distr[dim-1]=0;
            }
            /***END OF HACK***/

            MPI_Dims_create(nproc, dim, distr);
#else
            throw Exceptions::MPISupportMissing();
#endif
        }

         static void Barrier(CommunicationGroup group)
         {
#ifdef HAVE_MPI
            TNL_ASSERT_TRUE(IsInitialized(), "Fatal Error - MPI communicator is not initialized");
            TNL_ASSERT_NE(group, NullGroup, "Barrier cannot be called with NullGroup");
            MPI_Barrier(group);
#else
            throw Exceptions::MPISupportMissing();
#endif
        }

         template <typename T>
         static Request ISend( const T* data, int count, int dest, CommunicationGroup group)
         {
#ifdef HAVE_MPI
            TNL_ASSERT_TRUE(IsInitialized(), "Fatal Error - MPI communicator is not initialized");
            TNL_ASSERT_NE(group, NullGroup, "ISend cannot be called with NullGroup");
            Request req;
            MPI_Isend((const void*) data, count, MPIDataType(data) , dest, 0, group, &req);
            return req;
#else
            throw Exceptions::MPISupportMissing();
#endif
        }

         template <typename T>
         static Request IRecv( T* data, int count, int src, CommunicationGroup group)
         {
#ifdef HAVE_MPI
            TNL_ASSERT_TRUE(IsInitialized(), "Fatal Error - MPI communicator is not initialized");
            TNL_ASSERT_NE(group, NullGroup, "IRecv cannot be called with NullGroup");
            Request req;
            MPI_Irecv((void*) data, count, MPIDataType(data) , src, 0, group, &req);
            return req;
#else
            throw Exceptions::MPISupportMissing();
#endif
        }

         static void WaitAll(Request *reqs, int length)
         {
#ifdef HAVE_MPI
            TNL_ASSERT_TRUE(IsInitialized(), "Fatal Error - MPI communicator is not initialized");
            MPI_Waitall(length, reqs, MPI_STATUSES_IGNORE);
#else
            throw Exceptions::MPISupportMissing();
#endif
        }

        template< typename T >
        static void Bcast( T* data, int count, int root, CommunicationGroup group)
        {
#ifdef HAVE_MPI
           TNL_ASSERT_TRUE(IsInitialized(), "Fatal Error - MPI communicator is not initialized");
           TNL_ASSERT_NE(group, NullGroup, "BCast cannot be called with NullGroup");
           MPI_Bcast((void*) data, count, MPIDataType(data), root, group);
#else
           throw Exceptions::MPISupportMissing();
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
            MPI_Allreduce( (const void*) data, (void*) reduced_data,count,MPIDataType(data),op,group);
#else
            throw Exceptions::MPISupportMissing();
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
            MPI_Reduce( (const void*) data, (void*) reduced_data,count,MPIDataType(data),op,root,group);
#else
            throw Exceptions::MPISupportMissing();
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
            MPI_Sendrecv( ( const void* ) sendData,
                          sendCount,
                          MPIDataType( sendData ),
                          destination,
                          sendTag,
                          ( void* ) receiveData,
                          receiveCount,
                          MPIDataType( receiveData ),
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
            MPI_Alltoall( ( const void* ) sendData,
                          sendCount,
                          MPIDataType( sendData ),
                          ( void* ) receiveData,
                          receiveCount,
                          MPIDataType( receiveData ),
                          group );
#else
            throw Exceptions::MPISupportMissing();
#endif
         }


      static void writeProlog( Logger& logger )
      {
         if( isDistributed() )
         {
            logger.writeParameter( "MPI processes:", GetSize(AllGroup) );
         }
      }

      static void CreateNewGroup(bool meToo,int myRank, CommunicationGroup &oldGroup, CommunicationGroup &newGroup)
      {
#ifdef HAVE_MPI
        if(meToo)
        {
            MPI_Comm_split(oldGroup, 1, myRank, &newGroup);
        }
        else
        {
            MPI_Comm_split(oldGroup, MPI_UNDEFINED, GetRank(oldGroup), &newGroup);
        }
#else
         throw Exceptions::MPISupportMissing();
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
    private :
      static std::streambuf* psbuf;
      static std::streambuf* backup;
      static std::ofstream filestr;
      static bool redirect;

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
std::streambuf* MpiCommunicator::psbuf = nullptr;
std::streambuf* MpiCommunicator::backup = nullptr;
std::ofstream MpiCommunicator::filestr;
bool MpiCommunicator::redirect = true;

#ifdef HAVE_MPI
// TODO: this duplicates MpiCommunicator::MPIDataType
template<typename Type>
struct MPITypeResolver
{
    static inline MPI_Datatype getType()
    {
        TNL_ASSERT_TRUE(false, "Fatal Error - Unknown MPI Type");
        return MPI_INT;
    };
};

template<> struct MPITypeResolver<char>
{
    static inline MPI_Datatype getType(){return MPI_CHAR;};
};

template<> struct MPITypeResolver<short int>
{
    static inline MPI_Datatype getType(){return MPI_SHORT;};
};

template<> struct MPITypeResolver<long int>
{
    static inline MPI_Datatype getType(){return MPI_LONG;};
};

template<> struct MPITypeResolver<unsigned char>
{
    static inline MPI_Datatype getType(){return MPI_UNSIGNED_CHAR;};
};

template<> struct MPITypeResolver<unsigned short int>
{
    static inline MPI_Datatype getType(){return MPI_UNSIGNED_SHORT;};
};

template<> struct MPITypeResolver<unsigned int>
{
    static inline MPI_Datatype getType(){return MPI_UNSIGNED;};
};

template<> struct MPITypeResolver<unsigned long int>
{
    static inline MPI_Datatype getType(){return MPI_UNSIGNED_LONG;};
};

template<> struct MPITypeResolver<float>
{
    static inline MPI_Datatype getType(){return MPI_FLOAT;};
};

template<> struct MPITypeResolver<double>
{
    static inline MPI_Datatype getType(){return MPI_DOUBLE;};
};

template<> struct MPITypeResolver<long double>
{
    static inline MPI_Datatype getType(){return MPI_LONG_DOUBLE;};
};
#endif

} // namespace <unnamed>
} // namespace Communicators
} // namespace TNL

#define TNL_MPI_PRINT( message )                                                                                         \
for( int j = 0; j < TNL::Communicators::MpiCommunicator::GetSize( TNL::Communicators::MpiCommunicator::AllGroup ); j++ ) \
   {                                                                                                                     \
      if( j == TNL::Communicators::MpiCommunicator::GetRank( TNL::Communicators::MpiCommunicator::AllGroup ) )           \
      {                                                                                                                  \
         std::cerr << "Node " << j << " of "                                                                             \
                   << TNL::Communicators::MpiCommunicator::GetSize( TNL::Communicators::MpiCommunicator::AllGroup )      \
                   << " : " << message << std::endl;                                                                     \
      }                                                                                                                  \
      TNL::Communicators::MpiCommunicator::Barrier( TNL::Communicators::MpiCommunicator::AllGroup );                     \
   }

