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

template class tnlSORSolver< tnlCSRMatrix< float,  tnlHost, int > >;
template class tnlSORSolver< tnlCSRMatrix< double, tnlHost, int > >;
template class tnlSORSolver< tnlCSRMatrix< float,  tnlHost, long int > >;
template class tnlSORSolver< tnlCSRMatrix< double, tnlHost, long int > >;

template class tnlSORSolver< tnlEllpackMatrix< float,  tnlHost, int > >;
template class tnlSORSolver< tnlEllpackMatrix< double, tnlHost, int > >;
template class tnlSORSolver< tnlEllpackMatrix< float,  tnlHost, long int > >;
template class tnlSORSolver< tnlEllpackMatrix< double, tnlHost, long int > >;

template class tnlSORSolver< tnlMultidiagonalMatrix< float,  tnlHost, int > >;
template class tnlSORSolver< tnlMultidiagonalMatrix< double, tnlHost, int > >;
template class tnlSORSolver< tnlMultidiagonalMatrix< float,  tnlHost, long int > >;
template class tnlSORSolver< tnlMultidiagonalMatrix< double, tnlHost, long int > >;


#ifdef HAVE_CUDA
template class tnlSORSolver< tnlCSRMatrix< float,  tnlCuda, int > >;
template class tnlSORSolver< tnlCSRMatrix< double, tnlCuda, int > >;
template class tnlSORSolver< tnlCSRMatrix< float,  tnlCuda, long int > >;
template class tnlSORSolver< tnlCSRMatrix< double, tnlCuda, long int > >;

template class tnlSORSolver< tnlEllpackMatrix< float,  tnlCuda, int > >;
template class tnlSORSolver< tnlEllpackMatrix< double, tnlCuda, int > >;
template class tnlSORSolver< tnlEllpackMatrix< float,  tnlCuda, long int > >;
template class tnlSORSolver< tnlEllpackMatrix< double, tnlCuda, long int > >;

template class tnlSORSolver< tnlMultidiagonalMatrix< float,  tnlCuda, int > >;
template class tnlSORSolver< tnlMultidiagonalMatrix< double, tnlCuda, int > >;
template class tnlSORSolver< tnlMultidiagonalMatrix< float,  tnlCuda, long int > >;
template class tnlSORSolver< tnlMultidiagonalMatrix< double, tnlCuda, long int > >;
#endif

} // namespace TNL


