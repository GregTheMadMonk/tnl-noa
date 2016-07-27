/***************************************************************************
                          tnlSORSolver_impl.cpp  -  description
                             -------------------
    begin                : Jan 20, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/solvers/linear/stationary/tnlSORSolver.h>
#include <TNL/matrices/tnlCSRMatrix.h>
#include <TNL/matrices/tnlEllpackMatrix.h>
#include <TNL/matrices/tnlMultidiagonalMatrix.h>

namespace TNL {

template class tnlSORSolver< tnlCSRMatrix< float,  Devices::Host, int > >;
template class tnlSORSolver< tnlCSRMatrix< double, Devices::Host, int > >;
template class tnlSORSolver< tnlCSRMatrix< float,  Devices::Host, long int > >;
template class tnlSORSolver< tnlCSRMatrix< double, Devices::Host, long int > >;

template class tnlSORSolver< tnlEllpackMatrix< float,  Devices::Host, int > >;
template class tnlSORSolver< tnlEllpackMatrix< double, Devices::Host, int > >;
template class tnlSORSolver< tnlEllpackMatrix< float,  Devices::Host, long int > >;
template class tnlSORSolver< tnlEllpackMatrix< double, Devices::Host, long int > >;

template class tnlSORSolver< tnlMultidiagonalMatrix< float,  Devices::Host, int > >;
template class tnlSORSolver< tnlMultidiagonalMatrix< double, Devices::Host, int > >;
template class tnlSORSolver< tnlMultidiagonalMatrix< float,  Devices::Host, long int > >;
template class tnlSORSolver< tnlMultidiagonalMatrix< double, Devices::Host, long int > >;


#ifdef HAVE_CUDA
template class tnlSORSolver< tnlCSRMatrix< float,  Devices::Cuda, int > >;
template class tnlSORSolver< tnlCSRMatrix< double, Devices::Cuda, int > >;
template class tnlSORSolver< tnlCSRMatrix< float,  Devices::Cuda, long int > >;
template class tnlSORSolver< tnlCSRMatrix< double, Devices::Cuda, long int > >;

template class tnlSORSolver< tnlEllpackMatrix< float,  Devices::Cuda, int > >;
template class tnlSORSolver< tnlEllpackMatrix< double, Devices::Cuda, int > >;
template class tnlSORSolver< tnlEllpackMatrix< float,  Devices::Cuda, long int > >;
template class tnlSORSolver< tnlEllpackMatrix< double, Devices::Cuda, long int > >;

template class tnlSORSolver< tnlMultidiagonalMatrix< float,  Devices::Cuda, int > >;
template class tnlSORSolver< tnlMultidiagonalMatrix< double, Devices::Cuda, int > >;
template class tnlSORSolver< tnlMultidiagonalMatrix< float,  Devices::Cuda, long int > >;
template class tnlSORSolver< tnlMultidiagonalMatrix< double, Devices::Cuda, long int > >;
#endif

} // namespace TNL


