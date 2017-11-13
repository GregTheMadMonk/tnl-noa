/***************************************************************************
                          mpi-supp.h  -  description
                             -------------------
    begin                : 2005/04/23
    copyright            : (C) 2005 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#ifdef USE_MPI
   #include <mpi.h>

namespace TNL {
    namespace TNLMPI{
        
    inline MPI_Datatype MPIDataType( const signed char ) { return MPI_CHAR; };
    inline MPI_Datatype MPIDataType( const signed short int ) { return MPI_SHORT; };
    inline MPI_Datatype MPIDataType( const signed int ) { return MPI_INT; };
    inline MPI_Datatype MPIDataType( const signed long int ) { return MPI_LONG; };
    inline MPI_Datatype MPIDataType( const unsigned char ) { return MPI_UNSIGNED_CHAR; };
    inline MPI_Datatype MPIDataType( const unsigned short int ) { return MPI_UNSIGNED_SHORT; };
    inline MPI_Datatype MPIDataType( const unsigned int ) { return MPI_UNSIGNED; };
    inline MPI_Datatype MPIDataType( const unsigned long int ) { return MPI_UNSIGNED_LONG; };
    inline MPI_Datatype MPIDataType( const float ) { return MPI_FLOAT; };
    inline MPI_Datatype MPIDataType( const double ) { return MPI_DOUBLE; };
    inline MPI_Datatype MPIDataType( const long double ) { return MPI_LONG_DOUBLE; };
    
    template <typename T>
    MPI::Request ISend( const T *data, int count, int dest)
    {
            return MPI::COMM_WORLD.Isend((void*) data, count, MPIDataType(*data) , dest, 0);
    }     

    template <typename T>
    MPI::Request IRecv( const T *data, int count, int src)
    {
            return MPI::COMM_WORLD.Irecv((void*) data, count, MPIDataType(*data) , src, 0);
    }

    template< typename T > 
    void Bcast(  T& data, int count, int root)
    {
            MPI::COMM_WORLD.Bcast((void*) &data, count,  MPIDataType(data), root);
    };

    template< typename T >
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
    };

}//namespace MPI
} // namespace TNL

#endif
