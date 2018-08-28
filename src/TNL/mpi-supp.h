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

#ifdef HAVE_MPI
   #include <mpi.h>
#else
   typedef int MPI_Comm;
   typedef int MPI_Op;
   #define MPI_COMM_WORLD  0
   #define MPI_MAX 0
   #define MPI_MIN 0
   #define MPI_SUM 0
#endif

namespace TNL {
   
class String;

#ifdef HAVE_MPI
inline MPI_Datatype MPIDataType( const signed char ) { return MPI_CHAR; };
inline MPI_Datatype MPIDataType( const signed short int ) { return MPI_SHORT; };
inline MPI_Datatype MPIDataType( const signed int ) { return MPI_INT; };
inline MPI_Datatype MPIDataType( const signed long int ) { return MPI_LONG; };
inline MPI_Datatype MPIDataType( const unsigned char ) { return MPI_UNSIGNED_CHAR; };
inline MPI_Datatype MPIDataType( const unsigned short int ) { return MPI_UNSIGNED_SHORT; };
inline MPI_Datatype MPIDataType( const unsigned int ) { return MPI_UNSIGNED; };
inline MPI_Datatype MPIDataType( const unsigned long int ) { return MPI_UNSIGNED_LONG; };
inline MPI_Datatype MPIDataType( const float& ) { return MPI_FLOAT; };
inline MPI_Datatype MPIDataType( const double& ) { return MPI_DOUBLE; };
inline MPI_Datatype MPIDataType( const long double& ) { return MPI_LONG_DOUBLE; };
#endif


void MPIInit( int* argc, char** argv[] );

void MPIFinalize();

bool HaveMPI();

int MPIGetRank( MPI_Comm comm = MPI_COMM_WORLD );

int MPIGetSize( MPI_Comm comm = MPI_COMM_WORLD );

void MPIBarrier( MPI_Comm comm = MPI_COMM_WORLD );

#ifdef HAVE_MPI
template< class T > void MPISend( const T& data,
                                  int count,
                                  int dest,
                                  MPI_Comm comm = MPI_COMM_WORLD )
{
   MPI_Send( &data, count, MPIDataType( data ), dest, 0, comm );
};
#else
template< class T > void MPISend( const T&,
                                  int,
                                  int,
                                  MPI_Comm  )
{
};
#endif

#ifdef HAVE_MPI
template< class T > void MPIRecv( T& data,
                                  int count,
                                  int src,
                                  MPI_Comm comm = MPI_COMM_WORLD )
{
   MPI_Status stat;
   MPI_Recv( data, count, MPIDataType( data ), src, 0, comm, &stat );
};
#else
template< class T > void MPIRecv( T&,
                                  int,
                                  int,
                                  MPI_Comm = MPI_COMM_WORLD )
{};
#endif

#ifdef HAVE_MPI
template< class T > void MPIBcast( T& data,
                                   int count,
                                   int root,
                                   MPI_Comm comm = MPI_COMM_WORLD )
{
   MPI_Bcast( &data, count, MPIDataType( data ), root, comm );
};

inline void MPIBcast( String& data, intstd::cout, int root, MPI_Comm comm = MPI_COMM_WORLD )
{
   std::cerr << "Call method MPIBcast of mString instead of function MPIBcast( mString&, ... ) " << std::endl;
   abort();
}
#else
template< class T > void MPIBcast( T&,
                                   int,
                                   int,
                                   MPI_Comm = MPI_COMM_WORLD )
{
}
#endif

#ifdef HAVE_MPI
template< typename T > void MPIReduce( T& data,
                                       T& reduced_data,
                                       int count,
                                       MPI_Op op,
                                       int root,
                                       MPI_Comm comm )
{
   MPI_Reduce( &data,
               &reduced_data,
               count,
               MPIDataType( data ),
               op,
               root,
               comm );
};
#else
template< typename T > void MPIReduce( T& data,
                                       T& reduced_data,
                                       int,
                                       MPI_Op,
                                       int,
                                       MPI_Comm )
{
   reduced_data = data;
};
#endif

#ifdef HAVE_MPI
template< typename T > void MPIAllreduce( T& data,
                                          T& reduced_data,
                                          int count,
                                          MPI_Op op,
                                          MPI_Comm comm )
{
   MPI_Allreduce( &data,
                  &reduced_data,
                  count,
                  MPIDataType( data ),
                  op,
                  comm );
};
#else
template< typename T > void MPIAllreduce( T& data,
                                          T& reduced_data,
                                          int,
                                          MPI_Op,
                                          MPI_Comm )
{
   reduced_data = data;
};
#endif

} // namespace TNL
