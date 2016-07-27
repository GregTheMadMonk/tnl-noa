/***************************************************************************
                          tnlGMRESSolver_impl.cpp  -  description
                             -------------------
    begin                : Jan 20, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/solvers/linear/krylov/tnlGMRESSolver.h>
#include <TNL/matrices/tnlCSRMatrix.h>
#include <TNL/matrices/tnlEllpackMatrix.h>
#include <TNL/matrices/tnlMultidiagonalMatrix.h>

namespace TNL {

template class tnlGMRESSolver< tnlCSRMatrix< float,  Devices::Host, int > >;
template class tnlGMRESSolver< tnlCSRMatrix< double, Devices::Host, int > >;
template class tnlGMRESSolver< tnlCSRMatrix< float,  Devices::Host, long int > >;
template class tnlGMRESSolver< tnlCSRMatrix< double, Devices::Host, long int > >;

/*template class tnlGMRESSolver< tnlEllpackMatrix< float,  Devices::Host, int > >;
template class tnlGMRESSolver< tnlEllpackMatrix< double, Devices::Host, int > >;
template class tnlGMRESSolver< tnlEllpackMatrix< float,  Devices::Host, long int > >;
template class tnlGMRESSolver< tnlEllpackMatrix< double, Devices::Host, long int > >;

template class tnlGMRESSolver< tnlMultiDiagonalMatrix< float,  Devices::Host, int > >;
template class tnlGMRESSolver< tnlMultiDiagonalMatrix< double, Devices::Host, int > >;
template class tnlGMRESSolver< tnlMultiDiagonalMatrix< float,  Devices::Host, long int > >;
template class tnlGMRESSolver< tnlMultiDiagonalMatrix< double, Devices::Host, long int > >;*/


#ifdef HAVE_CUDA
template class tnlGMRESSolver< tnlCSRMatrix< float,  Devices::Cuda, int > >;
template class tnlGMRESSolver< tnlCSRMatrix< double, Devices::Cuda, int > >;
template class tnlGMRESSolver< tnlCSRMatrix< float,  Devices::Cuda, long int > >;
template class tnlGMRESSolver< tnlCSRMatrix< double, Devices::Cuda, long int > >;

/*template class tnlGMRESSolver< tnlEllpackMatrix< float,  Devices::Cuda, int > >;
template class tnlGMRESSolver< tnlEllpackMatrix< double, Devices::Cuda, int > >;
template class tnlGMRESSolver< tnlEllpackMatrix< float,  Devices::Cuda, long int > >;
template class tnlGMRESSolver< tnlEllpackMatrix< double, Devices::Cuda, long int > >;*/

/*template class tnlGMRESSolver< tnlMultiDiagonalMatrix< float,  Devices::Cuda, int > >;
template class tnlGMRESSolver< tnlMultiDiagonalMatrix< double, Devices::Cuda, int > >;
template class tnlGMRESSolver< tnlMultiDiagonalMatrix< float,  Devices::Cuda, long int > >;
template class tnlGMRESSolver< tnlMultiDiagonalMatrix< double, Devices::Cuda, long int > >;*/
#endif

} // namespace TNL
