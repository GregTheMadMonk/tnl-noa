/***************************************************************************
                          GMRES_impl.cpp  -  description
                             -------------------
    begin                : Jan 20, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/Solvers/Linear/GMRES.h>
#include <TNL/Matrices/CSRMatrix.h>
#include <TNL/Matrices/EllpackMatrix.h>
#include <TNL/Matrices/MultidiagonalMatrix.h>

namespace TNL {
namespace Solvers {
namespace Linear {

template class GMRES< Matrices::CSRMatrix< float,  Devices::Host, int > >;
template class GMRES< Matrices::CSRMatrix< double, Devices::Host, int > >;
template class GMRES< Matrices::CSRMatrix< float,  Devices::Host, long int > >;
template class GMRES< Matrices::CSRMatrix< double, Devices::Host, long int > >;

/*template class GMRES< Matrices::EllpackMatrix< float,  Devices::Host, int > >;
template class GMRES< Matrices::EllpackMatrix< double, Devices::Host, int > >;
template class GMRES< Matrices::EllpackMatrix< float,  Devices::Host, long int > >;
template class GMRES< Matrices::EllpackMatrix< double, Devices::Host, long int > >;

template class GMRES< Matrices::MultidiagonalMatrix< float,  Devices::Host, int > >;
template class GMRES< Matrices::MultidiagonalMatrix< double, Devices::Host, int > >;
template class GMRES< Matrices::MultidiagonalMatrix< float,  Devices::Host, long int > >;
template class GMRES< Matrices::MultidiagonalMatrix< double, Devices::Host, long int > >;*/


#ifdef HAVE_CUDA
template class GMRES< Matrices::CSRMatrix< float,  Devices::Cuda, int > >;
template class GMRES< Matrices::CSRMatrix< double, Devices::Cuda, int > >;
template class GMRES< Matrices::CSRMatrix< float,  Devices::Cuda, long int > >;
template class GMRES< Matrices::CSRMatrix< double, Devices::Cuda, long int > >;

/*template class GMRES< Matrices::EllpackMatrix< float,  Devices::Cuda, int > >;
template class GMRES< Matrices::EllpackMatrix< double, Devices::Cuda, int > >;
template class GMRES< Matrices::EllpackMatrix< float,  Devices::Cuda, long int > >;
template class GMRES< Matrices::EllpackMatrix< double, Devices::Cuda, long int > >;*/

/*template class GMRES< Matrices::MultidiagonalMatrix< float,  Devices::Cuda, int > >;
template class GMRES< Matrices::MultidiagonalMatrix< double, Devices::Cuda, int > >;
template class GMRES< Matrices::MultidiagonalMatrix< float,  Devices::Cuda, long int > >;
template class GMRES< Matrices::MultidiagonalMatrix< double, Devices::Cuda, long int > >;*/
#endif

} // namespace Linear
} // namespace Solvers
} // namespace TNL
