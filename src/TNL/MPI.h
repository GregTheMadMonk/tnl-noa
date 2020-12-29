/***************************************************************************
                          MPI.h  -  description
                             -------------------
    begin                : Dec 29, 2020
    copyright            : (C) 2020 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

/**
 * \brief A convenient header file which includes all headers from the
 * `TNL/MPI/` subdirectory.
 *
 * Users may use this to avoid having to include many header files in their
 * projects. On the other hand, parts of the TNL library should generally
 * include only the specific headers they need, in order to avoid cycles in
 * the header inclusion.
 */

#include "MPI/DummyDefs.h"
#include "MPI/getDataType.h"
#include "MPI/selectGPU.h"
#include "MPI/Wrappers.h"
#include "MPI/Utils.h"
#include "MPI/ScopedInitializer.h"
#include "MPI/Print.h"
