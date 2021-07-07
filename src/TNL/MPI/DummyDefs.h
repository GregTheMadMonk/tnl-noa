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

// Miscellaneous constants
#define MPI_ANY_SOURCE         -1                      /* match any source rank */
#define MPI_PROC_NULL          -2                      /* rank of null process */
#define MPI_ROOT               -4                      /* special value for intercomms */
#define MPI_ANY_TAG            -1                      /* match any message tag */
#define MPI_UNDEFINED          -32766                  /* undefined stuff */
#define MPI_DIST_GRAPH         3                       /* dist graph topology */
#define MPI_CART               1                       /* cartesian topology */
#define MPI_GRAPH              2                       /* graph topology */
#define MPI_KEYVAL_INVALID     -1                      /* invalid key value */
#define MPI_COMM_WORLD         0

#endif
