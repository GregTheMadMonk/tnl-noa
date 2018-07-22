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
#include <mpi-ext.h>

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

      using Request = MPI_Request;
      using CommunicationGroup = MPI_Comm;
#else
      using Request = int;
      using CommunicationGroup = int;
#endif

      static bool isDistributed()
      {
         return GetSize(AllGroup)>1;
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
               if( rank == 0 )
                  std::cerr << "Attach GDB to MPI process(es) by entering:" << std::endl;
               for( int i = 0; i < GetSize( MPI_COMM_WORLD ); i++ )
               {
                  MPI_Status status;
                  int recvPid;
                  MPI_Recv( &recvPid, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &status );
               
                  if( i == processToAttach || processToAttach == -1 )
                  {
                     std::cerr << "  For MPI process " << i << ": gdb -q -ex \"attach " << recvPid << "\"" 
                               << " -ex \"set variable tnlMPIDebugAttached=1\"" 
                               << " -ex \"finish\"" << std::endl;
                  }
               }
               if( rank == processToAttach || processToAttach == -1 )
                  while( ! tnlMPIDebugAttached );
               MPI_Barrier( MPI_COMM_WORLD );
            }
         }
#endif // HAVE_MPI
         return true;
      }

      static void Init(int argc, char **argv )
      {
#ifdef HAVE_MPI
         MPI_Init( &argc, &argv );
         NullRequest=MPI_REQUEST_NULL;
         AllGroup=MPI_COMM_WORLD;
         redirect = true;

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
               std::cout<< GetRank(AllGroup) <<": Redirecting std::out to file" <<std::endl;
               String stdoutFile;
               stdoutFile=String( "./stdout-")+convertToString(MPI::COMM_WORLD.Get_rank())+String(".txt");
               filestr.open (stdoutFile.getString()); 
               psbuf = filestr.rdbuf(); 
               std::cout.rdbuf(psbuf);
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
         int inicialized, finalized;
         MPI_Initialized(&inicialized);
         MPI_Finalized(&finalized);
         return inicialized && !finalized;
#else
        return false;
#endif
      }

      static int GetRank(CommunicationGroup group)
      {
#ifdef HAVE_MPI
        TNL_ASSERT_TRUE(IsInitialized(), "Fatal Error - MPI communicator is not initialized");
        int rank;
        MPI_Comm_rank(group,&rank);
        return rank;
#else
        return 1;
#endif
      }

      static int GetSize(CommunicationGroup group)
      {
#ifdef HAVE_MPI
        TNL_ASSERT_TRUE(IsInitialized(), "Fatal Error - MPI communicator is not initialized");
        int size;
        MPI_Comm_size(group,&size);
        return size;
#else
        return 1;
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
#endif
        }

         static void Barrier(CommunicationGroup comm)
         {
#ifdef HAVE_MPI
            TNL_ASSERT_TRUE(IsInitialized(), "Fatal Error - MPI communicator is not inicialized");
            MPI_Barrier(comm);
#else
            throw Exceptions::MPISupportMissing();
#endif
        }

         template <typename T>
         static Request ISend( const T *data, int count, int dest, CommunicationGroup group)
         {
#ifdef HAVE_MPI
            TNL_ASSERT_TRUE(IsInitialized(), "Fatal Error - MPI communicator is not inicialized");
            Request req;
            MPI_Isend((void*) data, count, MPIDataType(data) , dest, 0, group, &req);
            return req;
#else
            throw Exceptions::MPISupportMissing();
#endif
        }

         template <typename T>
         static Request IRecv( const T *data, int count, int src, CommunicationGroup group)
         {
#ifdef HAVE_MPI
            TNL_ASSERT_TRUE(IsInitialized(), "Fatal Error - MPI communicator is not inicialized");
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
            TNL_ASSERT_TRUE(IsInitialized(), "Fatal Error - MPI communicator is not inicialized");
            MPI_Waitall(length, reqs, MPI_STATUSES_IGNORE);
#else
            throw Exceptions::MPISupportMissing();
#endif
        }

        template< typename T > 
        static void Bcast(  T& data, int count, int root,CommunicationGroup group)
        {
#ifdef HAVE_MPI
        TNL_ASSERT_TRUE(IsInitialized(), "Fatal Error - MPI communicator is not inicialized");
        MPI_Bcast((void*) &data, count,  MPIDataType(data), root, group);
#else
        throw Exceptions::MPISupportMissing();
#endif
        }

        template< typename T >
        static void Allreduce( T* data,
                               T* reduced_data,
                               int count,
                               const MPI_Op &op,
                               CommunicationGroup group)
        {
#ifdef HAVE_MPI
            MPI_Allreduce( (void*) data, (void*) reduced_data,count,MPIDataType(data),op,group);
#else
            throw Exceptions::MPISupportMissing();
#endif
        }


         template< typename T >
         static void Reduce( T* data,
                    T* reduced_data,
                    int count,
                    MPI_Op &op,
                    int root,
                    CommunicationGroup group)
         {
#ifdef HAVE_MPI
            MPI_Reduce( (void*) data, (void*) reduced_data,count,MPIDataType(data),op,root,group);
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
        newGroup=oldGroup;
#endif         
      }

#ifdef HAVE_MPI
      static MPI_Request NullRequest;
      static MPI_Comm AllGroup;
#else
      static int NullRequest;
      static int AllGroup;
#endif
    private :
      static std::streambuf *psbuf;
      static std::streambuf *backup;
      static std::ofstream filestr;
      static bool redirect;
      static bool inited;

      static void selectGPU(void)
      {
#ifdef HAVE_MPI
    #ifdef HAVE_CUDA
        	int count,rank, gpuCount, gpuNumber;
         MPI_Comm_size(MPI_COMM_WORLD,&count);
         MPI_Comm_rank(MPI_COMM_WORLD,&rank);

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

         gpuNumber=nodeRank % gpuCount;

         cudaSetDevice(gpuNumber);
         TNL_CHECK_CUDA_DEVICE;

         //std::cout<<"Node: " << rank << " gpu: " << gpuNumber << std::endl;

    #endif
#endif
      }


};

#ifdef HAVE_MPI
MPI_Request MpiCommunicator::NullRequest;
MPI_Comm MpiCommunicator::AllGroup;
#else
int MpiCommunicator::NullRequest;
int MpiCommunicator::AllGroup;
#endif
std::streambuf *MpiCommunicator::psbuf;
std::streambuf *MpiCommunicator::backup;
std::ofstream MpiCommunicator::filestr;
bool MpiCommunicator::redirect;
bool MpiCommunicator::inited;

}//namespace Communicators
} // namespace TNL



