/***************************************************************************
                          MpiCommunicator.h  -  description
                             -------------------
    begin                : 2005/04/23
    copyright            : (C) 2005 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#ifdef HAVE_MPI

#include <mpi.h>

namespace TNL {
namespace Communicators {

    class MpiCommunicator
    {

        private:
        inline static MPI_Datatype MPIDataType( const signed char ) { return MPI_CHAR; };
        inline static MPI_Datatype MPIDataType( const signed short int ) { return MPI_SHORT; };
        inline static MPI_Datatype MPIDataType( const signed int ) { return MPI_INT; };
        inline static MPI_Datatype MPIDataType( const signed long int ) { return MPI_LONG; };
        inline static MPI_Datatype MPIDataType( const unsigned char ) { return MPI_UNSIGNED_CHAR; };
        inline static MPI_Datatype MPIDataType( const unsigned short int ) { return MPI_UNSIGNED_SHORT; };
        inline static MPI_Datatype MPIDataType( const unsigned int ) { return MPI_UNSIGNED; };
        inline static MPI_Datatype MPIDataType( const unsigned long int ) { return MPI_UNSIGNED_LONG; };
        inline static MPI_Datatype MPIDataType( const float ) { return MPI_FLOAT; };
        inline static MPI_Datatype MPIDataType( const double ) { return MPI_DOUBLE; };
        inline static MPI_Datatype MPIDataType( const long double ) { return MPI_LONG_DOUBLE; };
        
        public:

        typedef MPI::Request Request;
        static MPI::Request NullRequest;

        static void Init(int argc, char **argv)
        {
            MPI::Init(argc,argv);
            NullRequest=MPI::REQUEST_NULL;
        };

        static void Finalize()
        {
            MPI::Finalize();
        };

        static bool IsInitialized()
        {
            return MPI::Is_initialized();
        };

        static int GetRank()
        {
            return MPI::COMM_WORLD.Get_rank();
        };

        static int GetSize()
        {
            return MPI::COMM_WORLD.Get_size();
        };

        //dim-number of dimesions, distr array of guess distr - 0 for computation
        //distr array will be filled by computed distribution
        //more information in MPI documentation
        static void DimsCreate(int nproc, int dim, int *distr)
        {
            MPI_Dims_create(nproc, dim, distr);
        };

        static void Barrier()
        {
            MPI::COMM_WORLD.Barrier();
        };

        template <typename T>
        static Request ISend( const T *data, int count, int dest)
        {
                return MPI::COMM_WORLD.Isend((void*) data, count, MPIDataType(*data) , dest, 0);
        };    

        template <typename T>
        static Request IRecv( const T *data, int count, int src)
        {
                return MPI::COMM_WORLD.Irecv((void*) data, count, MPIDataType(*data) , src, 0);
        };

        static void WaitAll(Request *reqs, int length)
        {
                MPI::Request::Waitall(length, reqs);
        };

        template< typename T > 
        static void Bcast(  T& data, int count, int root)
        {
                MPI::COMM_WORLD.Bcast((void*) &data, count,  MPIDataType(data), root);
        };

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
    };

    MPI::Request MpiCommunicator::NullRequest;

}//namespace Communicators
} // namespace TNL

#endif
