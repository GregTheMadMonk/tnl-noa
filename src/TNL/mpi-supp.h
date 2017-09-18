/***************************************************************************
                          mpi-supp.h  -  description
                             -------------------
    begin                : 2005/04/23
    copyright            : (C) 2005 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <iostream>
#include <cstdlib>

#ifdef USE_MPI
   #include <mpi.h>
#else
   typedef int MPI_Comm;
   typedef int MPI_Op;
   #define MPI_COMM_WORLD  0
   #define MPI_MAX 0
   #define MPI_SUM 0

template< typename T > 
void MPIAllreduce( T& data,
                 T& reduced_data,
                 int,
                 MPI_Op,
                 MPI_Comm )
{
    reduced_data = data;
};

template< typename T >
void MPIReduce( T& data,
                T& reduced_data,
                int,
                MPI_Op,
                int,
                MPI_Comm )
{
   reduced_data = data;
};

template< typename T > 
void MPIBcast(  T&,
                int,
                int,
                MPI_Comm = MPI_COMM_WORLD )
{
};
   
#endif

namespace TNL {
    namespace TNLMPI{

#ifdef USE_MPI
        
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

#endif

}//namespace MPI
} // namespace TNL
