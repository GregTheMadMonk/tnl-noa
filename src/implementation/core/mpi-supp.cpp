/***************************************************************************
                          mpi-supp.cpp  -  description
                             -------------------
    begin                : 2007/06/21
    copyright            : (C) 2007 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

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

