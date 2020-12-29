/***************************************************************************
                          MPI/DummyDefs.h  -  description
                             -------------------
    begin                : Dec 29, 2020
    copyright            : (C) 2020 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#ifndef HAVE_MPI
using MPI_Request = int;
using MPI_Comm = int;

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
