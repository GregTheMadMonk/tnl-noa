/***************************************************************************
                          tnlGMRESSolver_impl.cpp  -  description
                             -------------------
    begin                : Jan 20, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <solvers/linear/krylov/tnlGMRESSolver.h>
#include <matrices/tnlCSRMatrix.h>
#include <matrices/tnlEllpackMatrix.h>
#include <matrices/tnlMultidiagonalMatrix.h>

namespace TNL {

template class tnlGMRESSolver< tnlCSRMatrix< float,  tnlHost, int > >;
template class tnlGMRESSolver< tnlCSRMatrix< double, tnlHost, int > >;
template class tnlGMRESSolver< tnlCSRMatrix< float,  tnlHost, long int > >;
template class tnlGMRESSolver< tnlCSRMatrix< double, tnlHost, long int > >;

/*template class tnlGMRESSolver< tnlEllpackMatrix< float,  tnlHost, int > >;
template class tnlGMRESSolver< tnlEllpackMatrix< double, tnlHost, int > >;
template class tnlGMRESSolver< tnlEllpackMatrix< float,  tnlHost, long int > >;
template class tnlGMRESSolver< tnlEllpackMatrix< double, tnlHost, long int > >;

template class tnlGMRESSolver< tnlMultiDiagonalMatrix< float,  tnlHost, int > >;
template class tnlGMRESSolver< tnlMultiDiagonalMatrix< double, tnlHost, int > >;
template class tnlGMRESSolver< tnlMultiDiagonalMatrix< float,  tnlHost, long int > >;
template class tnlGMRESSolver< tnlMultiDiagonalMatrix< double, tnlHost, long int > >;*/


#ifdef HAVE_CUDA
template class tnlGMRESSolver< tnlCSRMatrix< float,  tnlCuda, int > >;
template class tnlGMRESSolver< tnlCSRMatrix< double, tnlCuda, int > >;
template class tnlGMRESSolver< tnlCSRMatrix< float,  tnlCuda, long int > >;
template class tnlGMRESSolver< tnlCSRMatrix< double, tnlCuda, long int > >;

/*template class tnlGMRESSolver< tnlEllpackMatrix< float,  tnlCuda, int > >;
template class tnlGMRESSolver< tnlEllpackMatrix< double, tnlCuda, int > >;
template class tnlGMRESSolver< tnlEllpackMatrix< float,  tnlCuda, long int > >;
template class tnlGMRESSolver< tnlEllpackMatrix< double, tnlCuda, long int > >;*/

/*template class tnlGMRESSolver< tnlMultiDiagonalMatrix< float,  tnlCuda, int > >;
template class tnlGMRESSolver< tnlMultiDiagonalMatrix< double, tnlCuda, int > >;
template class tnlGMRESSolver< tnlMultiDiagonalMatrix< float,  tnlCuda, long int > >;
template class tnlGMRESSolver< tnlMultiDiagonalMatrix< double, tnlCuda, long int > >;*/
#endif

} // namespace TNL
