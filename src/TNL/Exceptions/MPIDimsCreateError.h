/***************************************************************************
                          MPIDimsCreateError.h  -  description
                             -------------------
    begin                : Jan 30, 2019
    copyright            : (C) 2019 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <stdexcept>

namespace TNL {
namespace Exceptions {

struct MPIDimsCreateError
   : public std::runtime_error
{
   MPIDimsCreateError()
   : std::runtime_error( "The program tries to call MPI_Dims_create with wrong dimensions."
                         "Non of the dimensions is zero and product of all dimensions does not fit with number of MPI processes." )
   {}
};

} // namespace Exceptions
} // namespace TNL
