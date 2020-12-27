/***************************************************************************
                          MpiCommunicator.h  -  description
                             -------------------
    begin                : 2005/04/23
    copyright            : (C) 2005 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#ifndef HAVE_MPI
enum MPI_Op {
   MPI_MAX,
   MPI_MIN,
   MPI_SUM,
   MPI_PROD,
   MPI_LAND,
   MPI_BAND,
   MPI_LOR,
   MPI_BOR,
   MPI_LXOR,
   MPI_BXOR,
   MPI_MINLOC,
   MPI_MAXLOC,
};

// MPI_Init_thread constants
enum {
  MPI_THREAD_SINGLE,
  MPI_THREAD_FUNNELED,
  MPI_THREAD_SERIALIZED,
  MPI_THREAD_MULTIPLE
};
#endif
