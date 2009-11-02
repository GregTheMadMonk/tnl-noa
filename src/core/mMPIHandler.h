/***************************************************************************
                          mMPIHandler.h  -  description
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

#ifndef mMPIHandlerH
#define mMPIHandlerH
#ifdef HAVE_MPI
   #include<mpi.h>
#endif

class mMPIHandler
{
   public:

   mMPIHandler();

   int InitMPI( int* argc, char** argv[] );
   
   int FinalizeMPI();

   bool HaveMPI() const;

   int GetRank() const;

   int GetSize() const;

   bool GetAlone( int rank = 0 );

   int GetBackOK( int ret_val );

   int GetBackQuit( int ret_val );

   protected:
   static bool mpi_initialised;
};

#endif
