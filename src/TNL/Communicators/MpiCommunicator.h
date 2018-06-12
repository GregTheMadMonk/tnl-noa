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

#ifdef HAVE_MPI
#include <mpi.h>   
#endif

#include <TNL/String.h>
#include <TNL/Logger.h>
#include <TNL/Config/ConfigDescription.h>

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

      using Request = MPI::Request;
#else
      using Request = int;
#endif

      static bool isDistributed()
      {
         return GetSize()>1;
      };
      
      static void configSetup( Config::ConfigDescription& config, const String& prefix = "" )
      {
#ifdef HAVE_MPI         
         config.addEntry< bool >( "redirect-mpi-output", "Only process with rank 0 prints to console. Other processes are redirected to files.", true );
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
         }
#endif         
         return true;
      }

      static void Init(int argc, char **argv )
      {
#ifdef HAVE_MPI         
         MPI::Init( argc, argv );
         NullRequest=MPI::REQUEST_NULL;
         redirect = true;
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
            if(MPI::COMM_WORLD.Get_rank()!=0)
            {
               std::cout<< GetRank() <<": Redirecting std::out to file" <<std::endl;
               String stdoutFile;
               stdoutFile=String( "./stdout-")+convertToString(MPI::COMM_WORLD.Get_rank())+String(".txt");
               filestr.open (stdoutFile.getString()); 
               psbuf = filestr.rdbuf(); 
               std::cout.rdbuf(psbuf);
            }
         }
#endif
      };

      static void Finalize()
      {
#ifdef HAVE_MPI
         if(isDistributed())
         {
            if(MPI::COMM_WORLD.Get_rank()!=0)
            { 
               std::cout.rdbuf(backup);
               filestr.close(); 
            }
         }
         MPI::Finalize();
#endif
      };

      static bool IsInitialized()
      {
#ifdef HAVE_MPI 
         return MPI::Is_initialized() && !MPI::Is_finalized();
#else
        return false;
#endif
      };

      static int GetRank()
      {
         //CHECK_INICIALIZED_RET(MPI::COMM_WORLD.Get_rank());
#ifdef HAVE_MPI
        TNL_ASSERT_TRUE(IsInitialized(), "Fatal Error - MPI communicator is not inicialized");
        return MPI::COMM_WORLD.Get_rank();
#else
        TNL_ASSERT_TRUE(false, "Fatal Error - MPI in not compiled");
        return 0;
#endif
      };

      static int GetSize()
      {
#ifdef HAVE_MPI
        TNL_ASSERT_TRUE(IsInitialized(), "Fatal Error - MPI communicator is not inicialized");
        return MPI::COMM_WORLD.Get_size();
#else
        TNL_ASSERT_TRUE(false, "Fatal Error - MPI in not compiled");
        return 0;
#endif
      };

        //dim-number of dimesions, distr array of guess distr - 0 for computation
        //distr array will be filled by computed distribution
        //more information in MPI documentation
        static void DimsCreate(int nproc, int dim, int *distr)
        {
#ifdef HAVE_MPI
            /***HACK for linear distribution***/
      /*     int sum=0;
           for(int i=0;i<dim;i++)
                sum+=distr[i];
           if(sum==0) //uživatel neovlivňuje distribuci
           {
               std::cout << "vynucuji distribuci" <<std::endl;
               for(int i=0;i<dim-1;i++)
               {
                    distr[i]=1;
               }
               distr[dim-1]=0;
            }*/
            /***END OF HACK***/

            MPI_Dims_create(nproc, dim, distr);
#endif
        };

        static void Barrier()
        {
#ifdef HAVE_MPI
        TNL_ASSERT_TRUE(IsInitialized(), "Fatal Error - MPI communicator is not inicialized");
        MPI::COMM_WORLD.Barrier();;
#else
        TNL_ASSERT_TRUE(false, "Fatal Error - MPI in not compiled");
#endif     
        };

        template <typename T>
        static Request ISend( const T *data, int count, int dest)
        {
#ifdef HAVE_MPI
        TNL_ASSERT_TRUE(IsInitialized(), "Fatal Error - MPI communicator is not inicialized");
        return MPI::COMM_WORLD.Isend((void*) data, count, MPIDataType(data) , dest, 0);
#else
        TNL_ASSERT_TRUE(false, "Fatal Error - MPI in not compiled");
        return 0;
#endif  
        }    

        template <typename T>
        static Request IRecv( const T *data, int count, int src)
        {
#ifdef HAVE_MPI
        TNL_ASSERT_TRUE(IsInitialized(), "Fatal Error - MPI communicator is not inicialized");
        return MPI::COMM_WORLD.Irecv((void*) data, count, MPIDataType(data) , src, 0);
#else
        TNL_ASSERT_TRUE(false, "Fatal Error - MPI in not compiled");
        return 0;
#endif  
        }

        static void WaitAll(Request *reqs, int length)
        {
#ifdef HAVE_MPI
        TNL_ASSERT_TRUE(IsInitialized(), "Fatal Error - MPI communicator is not inicialized");
        MPI::Request::Waitall(length, reqs);
#else
        TNL_ASSERT_TRUE(false, "Fatal Error - MPI in not compiled");
#endif  

        };

        template< typename T > 
        static void Bcast(  T& data, int count, int root)
        {
#ifdef HAVE_MPI
        TNL_ASSERT_TRUE(IsInitialized(), "Fatal Error - MPI communicator is not inicialized");
        MPI::COMM_WORLD.Bcast((void*) &data, count,  MPIDataType(data), root);
#else
        TNL_ASSERT_TRUE(false, "Fatal Error - MPI in not compiled");
#endif  
        }

      /*  template< typename T >
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

      static void writeProlog( Logger& logger ) 
      {
         if( isDistributed() )
         {
            logger.writeParameter( "MPI processes:", GetSize() );
         }
      }
      
#ifdef HAVE_MPI
      static MPI::Request NullRequest;
#else
      static int NullRequest;
#endif

      static std::streambuf *psbuf;
      static std::streambuf *backup;
      static std::ofstream filestr;
      static bool redirect;
      static bool inited;
   
};
   
#ifdef HAVE_MPI 
MPI::Request MpiCommunicator::NullRequest;
#else
int MpiCommunicator::NullRequest;
#endif
std::streambuf *MpiCommunicator::psbuf;
std::streambuf *MpiCommunicator::backup;
std::ofstream MpiCommunicator::filestr;
bool MpiCommunicator::redirect;
bool MpiCommunicator::inited;

}//namespace Communicators
} // namespace TNL



