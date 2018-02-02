/***************************************************************************
                          NoDistrCommunicator.h  -  description
                             -------------------
    begin                : 2018/01/09
    copyright            : (C) 2018 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {
namespace Communicators {
        
    class NoDistrCommunicator
    {


        public:

        typedef int Request;
        Request NullRequest;

        void Init(int argc, char **argv)
        {
        };

        void Finalize()
        {
        };

        bool IsInitialized()
        {   
            return true;
        };

        int GetRank()
        {
            return 0;
        };

        int GetSize()
        {
            return 1;
        };

        void DimsCreate(int nproc, int dim, int *distr)
        {
            for(int i=0;i<dim;i++)
            {
                distr[i]=1;
            }
        };

        void Barrier()
        {
        };

        template <typename T>
        Request ISend( const T *data, int count, int dest)
        {
            return 1;
        };    

        template <typename T>
        Request IRecv( const T *data, int count, int src)
        {
            return 1;
        };

        void WaitAll(Request *reqs, int length)
        {
        };

        template< typename T > 
        void Bcast(  T& data, int count, int root)
        {
        };

       /* template< typename T >
        void Allreduce( T& data,
                     T& reduced_data,
                     int count,
                     const MPI_Op &op)
        {
                MPI::COMM_WORLD.Allreduce((void*) &data, (void*) &reduced_data,count,MPIDataType(data),op);
        };

        template< typename T >
        void Reduce( T& data,
                    T& reduced_data,
                    int count,
                    MPI_Op &op,
                    int root)
        {
             MPI::COMM_WORLD.Reduce((void*) &data, (void*) &reduced_data,count,MPIDataType(data),op,root);
        };*/
    };

} // namespace Communicators
} // namespace TNL


