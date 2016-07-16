/***************************************************************************
                          mpi-supp.cpp  -  description
                             -------------------
    begin                : 2007/06/21
    copyright            : (C) 2007 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <core/mpi-supp.h>

//--------------------------------------------------------------------------
void MPIInit( int* argc, char** argv[] )
{
#ifdef HAVE_MPI
   MPI_Init( argc, argv );
#endif
}

void MPIFinalize()
{
#ifdef HAVE_MPI
   MPI_Finalize();
#endif
}

bool HaveMPI()
{
#ifdef HAVE_MPI
   return true;
#else
   return false;
#endif
}

int MPIGetRank( MPI_Comm comm )
{
#ifdef HAVE_MPI
   int rank;
   MPI_Comm_rank( MPI_COMM_WORLD, &rank );
   return rank;
#else
   return 0;
#endif
}

int MPIGetSize( MPI_Comm comm )
{
#ifdef HAVE_MPI
   int size;
   MPI_Comm_size( comm, &size );
   return size;
#else
   return 0;
#endif
}


void MPIBarrier( MPI_Comm comm )
{
#ifdef HAVE_MPI
   MPI_Barrier( comm );
#endif
}

