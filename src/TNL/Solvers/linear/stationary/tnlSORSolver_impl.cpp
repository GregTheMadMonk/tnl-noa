/***************************************************************************
                          tnlSORSolver_impl.cpp  -  description
                             -------------------
    begin                : Jan 20, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/Solvers/linear/stationary/tnlSORSolver.h>
#include <TNL/Matrices/CSRMatrix.h>
#include <TNL/Matrices/EllpackMatrix.h>
#include <TNL/Matrices/MultidiagonalMatrix.h>

namespace TNL {

template class tnlSORSolver< Matrices::CSRMatrix< float,  Devices::Host, int > >;
template class tnlSORSolver< Matrices::CSRMatrix< double, Devices::Host, int > >;
template class tnlSORSolver< Matrices::CSRMatrix< float,  Devices::Host, long int > >;
template class tnlSORSolver< Matrices::CSRMatrix< double, Devices::Host, long int > >;

template class tnlSORSolver< Matrices::EllpackMatrix< float,  Devices::Host, int > >;
template class tnlSORSolver< Matrices::EllpackMatrix< double, Devices::Host, int > >;
template class tnlSORSolver< Matrices::EllpackMatrix< float,  Devices::Host, long int > >;
template class tnlSORSolver< Matrices::EllpackMatrix< double, Devices::Host, long int > >;

template class tnlSORSolver< Matrices::MultidiagonalMatrix< float,  Devices::Host, int > >;
template class tnlSORSolver< Matrices::MultidiagonalMatrix< double, Devices::Host, int > >;
template class tnlSORSolver< Matrices::MultidiagonalMatrix< float,  Devices::Host, long int > >;
template class tnlSORSolver< Matrices::MultidiagonalMatrix< double, Devices::Host, long int > >;


#ifdef HAVE_CUDA
template class tnlSORSolver< Matrices::CSRMatrix< float,  Devices::Cuda, int > >;
template class tnlSORSolver< Matrices::CSRMatrix< double, Devices::Cuda, int > >;
template class tnlSORSolver< Matrices::CSRMatrix< float,  Devices::Cuda, long int > >;
template class tnlSORSolver< Matrices::CSRMatrix< double, Devices::Cuda, long int > >;

template class tnlSORSolver< Matrices::EllpackMatrix< float,  Devices::Cuda, int > >;
template class tnlSORSolver< Matrices::EllpackMatrix< double, Devices::Cuda, int > >;
template class tnlSORSolver< Matrices::EllpackMatrix< float,  Devices::Cuda, long int > >;
template class tnlSORSolver< Matrices::EllpackMatrix< double, Devices::Cuda, long int > >;

template class tnlSORSolver< Matrices::MultidiagonalMatrix< float,  Devices::Cuda, int > >;
template class tnlSORSolver< Matrices::MultidiagonalMatrix< double, Devices::Cuda, int > >;
template class tnlSORSolver< Matrices::MultidiagonalMatrix< float,  Devices::Cuda, long int > >;
template class tnlSORSolver< Matrices::MultidiagonalMatrix< double, Devices::Cuda, long int > >;
#endif

} // namespace TNL


