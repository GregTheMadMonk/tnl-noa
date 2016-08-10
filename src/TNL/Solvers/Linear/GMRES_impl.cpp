/***************************************************************************
                          GMRES_impl.cpp  -  description
                             -------------------
    begin                : Jan 20, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/Solvers/Linear/GMRES.h>
#include <TNL/Matrices/CSR.h>
#include <TNL/Matrices/Ellpack.h>
#include <TNL/Matrices/Multidiagonal.h>

namespace TNL {
namespace Solvers {
namespace Linear {

template class GMRES< Matrices::CSR< float,  Devices::Host, int > >;
template class GMRES< Matrices::CSR< double, Devices::Host, int > >;
template class GMRES< Matrices::CSR< float,  Devices::Host, long int > >;
template class GMRES< Matrices::CSR< double, Devices::Host, long int > >;

/*template class GMRES< Matrices::Ellpack< float,  Devices::Host, int > >;
template class GMRES< Matrices::Ellpack< double, Devices::Host, int > >;
template class GMRES< Matrices::Ellpack< float,  Devices::Host, long int > >;
template class GMRES< Matrices::Ellpack< double, Devices::Host, long int > >;

template class GMRES< Matrices::Multidiagonal< float,  Devices::Host, int > >;
template class GMRES< Matrices::Multidiagonal< double, Devices::Host, int > >;
template class GMRES< Matrices::Multidiagonal< float,  Devices::Host, long int > >;
template class GMRES< Matrices::Multidiagonal< double, Devices::Host, long int > >;*/


#ifdef HAVE_CUDA
template class GMRES< Matrices::CSR< float,  Devices::Cuda, int > >;
template class GMRES< Matrices::CSR< double, Devices::Cuda, int > >;
template class GMRES< Matrices::CSR< float,  Devices::Cuda, long int > >;
template class GMRES< Matrices::CSR< double, Devices::Cuda, long int > >;

/*template class GMRES< Matrices::Ellpack< float,  Devices::Cuda, int > >;
template class GMRES< Matrices::Ellpack< double, Devices::Cuda, int > >;
template class GMRES< Matrices::Ellpack< float,  Devices::Cuda, long int > >;
template class GMRES< Matrices::Ellpack< double, Devices::Cuda, long int > >;*/

/*template class GMRES< Matrices::Multidiagonal< float,  Devices::Cuda, int > >;
template class GMRES< Matrices::Multidiagonal< double, Devices::Cuda, int > >;
template class GMRES< Matrices::Multidiagonal< float,  Devices::Cuda, long int > >;
template class GMRES< Matrices::Multidiagonal< double, Devices::Cuda, long int > >;*/
#endif

} // namespace Linear
} // namespace Solvers
} // namespace TNL
