/***************************************************************************
                          mMPIHandler.cpp  -  description
                             -------------------
    begin                : 2007/06/19
    copyright            : (C) 2007 by Tomas Oberhuber
    email                : oberhuber@seznam.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include <assert.h>
#include <iostream.h>
#include "mMPIHandler.h"


bool mMPIHandler :: mpi_initialised( false );

//--------------------------------------------------------------------------
mMPIHandler :: mMPIHandler()
{
}
//--------------------------------------------------------------------------
int mMPIHandler :: InitMPI( int* argc, char** argv[] )
{
#ifdef HAVE_MPI
   int r = MPI_Init( argc, argv );
   mpi_initialised = true;
   return r;
#else
   return 1;
#endif
}
//--------------------------------------------------------------------------
int mMPIHandler :: FinalizeMPI()
{
#ifdef HAVE_MPI
   return MPI_Finalize();
#endif
   return 0;
}
//--------------------------------------------------------------------------
bool mMPIHandler :: HaveMPI() const
{
#ifdef HAVE_MPI
   return true;
#else
   return false;
#endif
}
//--------------------------------------------------------------------------
int mMPIHandler :: GetRank() const
{
#ifdef HAVE_MPI
   assert( mpi_initialised );
   int rank;
   MPI_Comm_rank( MPI_COMM_WORLD, &rank );
   return rank;
#else
   return 0;
#endif
}
//--------------------------------------------------------------------------
int mMPIHandler :: GetSize() const
{
#ifdef HAVE_MPI
   assert( mpi_initialised );
   int size;
   MPI_Comm_size( MPI_COMM_WORLD, &size );
   return size;
#else
   return 0;
#endif
}
//--------------------------------------------------------------------------
bool mMPIHandler :: GetAlone( int _rank )
{
#ifdef HAVE_MPI
   int rank = GetRank();
   if( rank == _rank ) return true;
   char stat;
   MPI_Bcast( &stat, 1, MPI_CHAR, _rank, MPI_COMM_WORLD );
   if( stat == 0 )
   {
      cout << "Finilizing MPI and aborting on node " << GetRank() << "." << endl;
      MPI_Finalize();
      abort();
   }
   return false;
#endif
   return true;
}
//--------------------------------------------------------------------------
int mMPIHandler :: GetBackOK( int ret_val )
{
#ifdef HAVE_MPI
   char stat( 1 );
   MPI_Bcast( &stat, 1, MPI_CHAR, GetRank(), MPI_COMM_WORLD );
   return ret_val;
#else
   return ret_val;
#endif
}
//--------------------------------------------------------------------------
int mMPIHandler :: GetBackQuit( int ret_val )
{
#ifdef HAVE_MPI
   char stat( 0 );
   MPI_Bcast( &stat, 1, MPI_CHAR, GetRank(), MPI_COMM_WORLD );
   MPI_Finalize();
   return ret_val;
#else
   return ret_val;
#endif
}
//--------------------------------------------------------------------------
